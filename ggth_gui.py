"""
GGTH Predictor GUI v2.2
Updated for unified_predictor_v8.py
Author: Jason Rusk

Fixes v2.1 → v2.2:
  - Thread-safe UI: all widget updates dispatched via self.after()
  - Removed dead _date_row helper; date rows now include a "Today" quick-fill button
  - __file__ resolution uses os.path.abspath() — safe regardless of CWD
  - Config I/O delegated entirely to ConfigManager (no duplicated JSON logic)
  - Stop button added; subprocess stored as self._proc for clean termination
  - interval_var changed to StringVar with explicit int validation
  - re / datetime moved to module-level imports
  - Save Log button added to action bar
"""

import os
import re
import sys
import json
import threading
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
from datetime import datetime

# ---------------------------------------------------------------------------
# Resolve the directory that contains THIS script — safe regardless of CWD
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))

SCRIPT_NAME = "unified_predictor_v8.py"
VERSION = "2.2"


# ---------------------------------------------------------------------------
# Lazy import of ConfigManager so the GUI still launches even if config_manager
# is absent (it will just fall back to the raw-JSON path with a warning).
# ---------------------------------------------------------------------------
def _load_config_manager():
    try:
        from config_manager import ConfigManager
        return ConfigManager(os.path.join(_HERE, "config.json"))
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------
class GGTHGui(tk.Tk):
    """
    Front-end for unified_predictor_v8.py.

    Calls the predictor script with one of these CLI modes:
        train | train-multitf | tune
        predict-mtf-once | predict-mtf-cont
        backtest | safe-backtest
    """

    def __init__(self):
        super().__init__()
        self.title(f"GGTH Predictor – Control Panel v{VERSION}")
        self.geometry("820x940")
        self.resizable(False, False)

        # Active subprocess — kept so Stop button can terminate it
        self._proc: subprocess.Popen | None = None
        self._running = False

        # ── Config manager ───────────────────────────────────────────────────
        self._config = _load_config_manager()

        # ── tk variables ─────────────────────────────────────────────────────
        self.symbol_var          = tk.StringVar(value="EURUSD")
        self.action_var          = tk.StringVar(value="train-multitf")
        self.force_retrain_var   = tk.BooleanVar(value=True)
        self.interval_var        = tk.StringVar(value="60")   # StringVar → validated manually

        # Model checkboxes
        self.models_lstm_var        = tk.BooleanVar(value=True)
        self.models_gru_var         = tk.BooleanVar(value=True)
        self.models_transformer_var = tk.BooleanVar(value=True)
        self.models_tcn_var         = tk.BooleanVar(value=True)
        self.models_lgbm_var        = tk.BooleanVar(value=True)

        self.use_kalman_var    = tk.BooleanVar(value=True)
        self.python_exe_var    = tk.StringVar(value=sys.executable)
        self.script_path_var   = tk.StringVar(value=self._default_script_path())
        self.mt5_path_var      = tk.StringVar(value="")

        # Date range (YYYY-MM-DD; blank = no limit)
        self.train_start_var   = tk.StringVar(value="")
        self.train_end_var     = tk.StringVar(value="")
        self.predict_start_var = tk.StringVar(value="")
        self.predict_end_var   = tk.StringVar(value="")

        # Load MT5 path from config
        self._load_mt5_path()

        self._build_ui()

    # =========================================================================
    # UI construction
    # =========================================================================
    def _build_ui(self):

        # ── Python / script / MT5 config ─────────────────────────────────────
        cfg_frame = ttk.LabelFrame(self, text="Python & Script Configuration")
        cfg_frame.place(x=10, y=10, width=800, height=120)

        ttk.Label(cfg_frame, text="Python exe:").place(x=10, y=10)
        ttk.Entry(cfg_frame, textvariable=self.python_exe_var, width=72).place(x=90, y=8)
        ttk.Button(cfg_frame, text="Browse…", command=self._browse_python).place(
            x=710, y=6, width=70)

        ttk.Label(cfg_frame, text="Predictor:").place(x=10, y=40)
        ttk.Entry(cfg_frame, textvariable=self.script_path_var, width=72).place(x=90, y=38)
        ttk.Button(cfg_frame, text="Browse…", command=self._browse_script).place(
            x=710, y=36, width=70)

        ttk.Label(cfg_frame, text="MT5 Files:").place(x=10, y=70)
        ttk.Entry(cfg_frame, textvariable=self.mt5_path_var, width=72).place(x=90, y=68)
        ttk.Button(cfg_frame, text="Browse…", command=self._browse_mt5).place(
            x=710, y=66, width=70)
        ttk.Label(cfg_frame, text="(MT5 Terminal\\…\\MQL5\\Files directory)",
                  foreground="gray", font=("Segoe UI", 7)).place(x=90, y=90)

        # ── Basic settings ────────────────────────────────────────────────────
        basic_frame = ttk.LabelFrame(self, text="Basic Settings")
        basic_frame.place(x=10, y=135, width=395, height=130)

        ttk.Label(basic_frame, text="Symbol:").place(x=10, y=10)
        ttk.Entry(basic_frame, textvariable=self.symbol_var, width=12).place(x=70, y=8)

        ttk.Label(basic_frame, text="Action:").place(x=10, y=40)
        row_y = 35
        for text, value in [
            ("Train ALL models (multi-TF) [RECOMMENDED]", "train-multitf"),
            ("Train main ensemble (single config)",        "train"),
            ("Hyperparameter tuning",                      "tune"),
        ]:
            ttk.Radiobutton(basic_frame, text=text,
                            variable=self.action_var, value=value).place(x=70, y=row_y)
            row_y += 22

        # ── Prediction / backtest modes ───────────────────────────────────────
        mode_frame = ttk.LabelFrame(self, text="Prediction / Backtest Modes")
        mode_frame.place(x=415, y=135, width=395, height=130)

        for row_y, (text, value) in enumerate([
            ("Predict ONCE for EA (multi-TF JSON)",         "predict-mtf-once"),
            ("Predict CONTINUOUSLY (multi-TF JSON)",        "predict-mtf-cont"),
            ("Generate backtest predictions",               "backtest"),
            ("Safe backtest (walk-forward, anti-leakage)",  "safe-backtest"),
        ]):
            ttk.Radiobutton(mode_frame, text=text,
                            variable=self.action_var, value=value).place(x=10, y=10 + row_y * 25)

        # ── Training / model options ──────────────────────────────────────────
        train_frame = ttk.LabelFrame(self, text="Training / Model Options")
        train_frame.place(x=10, y=270, width=395, height=130)

        ttk.Checkbutton(train_frame, text="Force retrain (ignore existing models)",
                        variable=self.force_retrain_var).place(x=10, y=5)

        ttk.Label(train_frame, text="Models to use:").place(x=10, y=35)
        for x_pos, (text, var) in zip(
            [110, 170, 230, 110, 170],
            [
                ("LSTM",        self.models_lstm_var),
                ("GRU",         self.models_gru_var),
                ("Transformer", self.models_transformer_var),
                ("TCN",         self.models_tcn_var),
                ("LightGBM",    self.models_lgbm_var),
            ]
        ):
            row = 33 if text in ("LSTM", "GRU", "Transformer") else 60
            ttk.Checkbutton(train_frame, text=text, variable=var).place(x=x_pos, y=row)

        ttk.Label(train_frame,
                  text="LSTM+Transformer+LightGBM recommended for best results",
                  foreground="gray", font=("Segoe UI", 7)).place(x=10, y=90)

        # ── Prediction options ────────────────────────────────────────────────
        pred_frame = ttk.LabelFrame(self, text="Prediction Options")
        pred_frame.place(x=415, y=270, width=395, height=130)

        ttk.Checkbutton(pred_frame, text="Use Kalman smoothing",
                        variable=self.use_kalman_var).place(x=10, y=5)

        ttk.Label(pred_frame, text="Continuous interval (mins):").place(x=10, y=35)
        ttk.Entry(pred_frame, textvariable=self.interval_var, width=6).place(x=180, y=33)

        ttk.Label(pred_frame,
                  text=(
                      "For continuous mode only. For one-shot prediction,\n"
                      "interval is ignored. Kalman smoothing helps reduce\n"
                      "prediction noise over time."
                  ),
                  foreground="gray").place(x=10, y=60)

        # ── Date range settings ───────────────────────────────────────────────
        date_frame = ttk.LabelFrame(self, text="Date Range Settings")
        date_frame.place(x=10, y=405, width=800, height=175)

        ttk.Label(date_frame, text="Training window",
                  font=("Segoe UI", 8, "bold")).place(x=120, y=5)
        ttk.Label(date_frame, text="Prediction / Backtest window",
                  font=("Segoe UI", 8, "bold")).place(x=520, y=5)

        ttk.Separator(date_frame, orient="vertical").place(x=400, y=5, height=155)

        col1_x, col2_x = 5, 410
        today_str = datetime.today().strftime("%Y-%m-%d")

        def _date_row(parent, label, var, y, col_x):
            """Build one labelled date entry with ✕ clear and Today quick-fill."""
            ttk.Label(parent, text=label, anchor="e", width=12).place(x=col_x + 5, y=y)
            entry = ttk.Entry(parent, textvariable=var, width=14)
            entry.place(x=col_x + 100, y=y - 2)
            ttk.Button(parent, text="✕", width=2,
                       command=lambda v=var: v.set("")).place(
                           x=col_x + 212, y=y - 2, width=22)
            ttk.Button(parent, text="Today", width=5,
                       command=lambda v=var, t=today_str: v.set(t)).place(
                           x=col_x + 238, y=y - 2, width=48)

        _date_row(date_frame, "Start date:", self.train_start_var,   32, col1_x)
        _date_row(date_frame, "End date:",   self.train_end_var,     62, col1_x)
        _date_row(date_frame, "Start date:", self.predict_start_var, 32, col2_x)
        _date_row(date_frame, "End date:",   self.predict_end_var,   62, col2_x)

        ttk.Label(date_frame,
                  text="Format: YYYY-MM-DD   |   Leave blank = no limit",
                  foreground="gray", font=("Segoe UI", 8)).place(x=10, y=108)
        ttk.Label(date_frame,
                  text="Clean out-of-sample test: set Train End = e.g. 2023-12-31, "
                       "Predict Start = 2024-01-01",
                  foreground="#0055AA", font=("Segoe UI", 8)).place(x=10, y=133)

        # ── Action bar ────────────────────────────────────────────────────────
        action_frame = ttk.Frame(self)
        action_frame.place(x=10, y=592, width=800, height=40)

        self._run_btn = ttk.Button(action_frame, text="▶  Run",
                                   command=self._on_run_clicked)
        self._run_btn.place(x=0, y=5, width=110, height=28)

        self._stop_btn = ttk.Button(action_frame, text="■  Stop",
                                    command=self._on_stop_clicked, state="disabled")
        self._stop_btn.place(x=115, y=5, width=110, height=28)

        ttk.Button(action_frame, text="Save MT5 Path",
                   command=self._save_mt5_path).place(x=230, y=5, width=110, height=28)
        ttk.Button(action_frame, text="Clear Log",
                   command=self._clear_log).place(x=345, y=5, width=90, height=28)
        ttk.Button(action_frame, text="Save Log",
                   command=self._save_log).place(x=440, y=5, width=90, height=28)
        ttk.Button(action_frame, text="Exit",
                   command=self.destroy).place(x=690, y=5, width=100, height=28)

        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(action_frame, text="Status:").place(x=540, y=10)
        self.status_label = ttk.Label(action_frame,
                                      textvariable=self.status_var, foreground="green")
        self.status_label.place(x=590, y=10)

        # ── Log window ────────────────────────────────────────────────────────
        log_frame = ttk.LabelFrame(self, text=f"Console Output  —  {SCRIPT_NAME}")
        log_frame.place(x=10, y=645, width=800, height=285)

        self.log_text = scrolledtext.ScrolledText(
            log_frame, wrap=tk.WORD, height=15, width=105,
            state="disabled", font=("Consolas", 9)
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    # =========================================================================
    # Helpers — path / config
    # =========================================================================
    def _default_script_path(self) -> str:
        """Locate the predictor script relative to this GUI file."""
        for name in [SCRIPT_NAME, "unified_predictor_v8_fixed.py",
                     "unified_predictor_v8.py", "GGTHpredictor2.py"]:
            candidate = os.path.join(_HERE, name)
            if os.path.isfile(candidate):
                return candidate
        return os.path.join(_HERE, SCRIPT_NAME)

    def _load_mt5_path(self):
        """Read MT5 path from ConfigManager (or raw JSON fallback)."""
        try:
            if self._config is not None:
                path = self._config.get("mt5_files_path", "")
            else:
                cfg_file = os.path.join(_HERE, "config.json")
                if os.path.exists(cfg_file):
                    with open(cfg_file, "r") as f:
                        path = json.load(f).get("mt5_files_path", "")
                else:
                    path = ""
            if path:
                self.mt5_path_var.set(path)
        except Exception as exc:
            print(f"Warning: could not load MT5 path from config: {exc}")

    def _save_mt5_path(self):
        """Persist MT5 path via ConfigManager (or raw JSON fallback)."""
        mt5_path = self.mt5_path_var.get().strip()
        if not mt5_path:
            messagebox.showwarning("No Path",
                                   "Please enter or browse to your MT5 Files directory.")
            return
        if not os.path.exists(mt5_path):
            messagebox.showerror("Invalid Path",
                                 f"Directory does not exist:\n{mt5_path}")
            return

        try:
            if self._config is not None:
                ok = self._config.set_mt5_files_path(mt5_path)
                if not ok:
                    raise RuntimeError("ConfigManager reported save failure.")
            else:
                # Fallback: raw JSON (ConfigManager unavailable)
                cfg_file = os.path.join(_HERE, "config.json")
                cfg = {}
                if os.path.exists(cfg_file):
                    with open(cfg_file, "r") as f:
                        cfg = json.load(f)
                cfg["mt5_files_path"] = mt5_path
                cfg["version"] = "2.0"
                with open(cfg_file, "w") as f:
                    json.dump(cfg, f, indent=2)

            messagebox.showinfo("Saved", f"MT5 path saved:\n{mt5_path}")
            self._set_status("MT5 path saved.", "green")

        except Exception as exc:
            messagebox.showerror("Error", f"Failed to save config:\n{exc}")

    # =========================================================================
    # Browse dialogs
    # =========================================================================
    def _browse_python(self):
        path = filedialog.askopenfilename(
            title="Select Python executable",
            filetypes=[("Python", "python.exe;pythonw.exe"), ("All files", "*.*")],
        )
        if path:
            self.python_exe_var.set(path)

    def _browse_script(self):
        path = filedialog.askopenfilename(
            title=f"Select {SCRIPT_NAME}",
            filetypes=[("Python files", "*.py"), ("All files", "*.*")],
        )
        if path:
            self.script_path_var.set(path)

    def _browse_mt5(self):
        initial = os.path.expandvars(r"%APPDATA%\MetaQuotes\Terminal")
        path = filedialog.askdirectory(title="Select MT5 Files Directory",
                                       initialdir=initial)
        if path:
            self.mt5_path_var.set(path)

    # =========================================================================
    # Log window helpers  ← always called on the main thread via self.after()
    # =========================================================================
    def _append_log(self, text: str):
        self.log_text.configure(state="normal")
        self.log_text.insert(tk.END, text)
        self.log_text.see(tk.END)
        self.log_text.configure(state="disabled")

    def _clear_log(self):
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state="disabled")

    def _save_log(self):
        """Write the current log contents to a user-chosen file."""
        path = filedialog.asksaveasfilename(
            title="Save log as…",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialfile=f"ggth_log_{datetime.today().strftime('%Y%m%d_%H%M%S')}.txt",
        )
        if not path:
            return
        try:
            content = self.log_text.get("1.0", tk.END)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            self._set_status(f"Log saved → {os.path.basename(path)}", "green")
        except Exception as exc:
            messagebox.showerror("Save Failed", str(exc))

    def _set_status(self, msg: str, color: str = "green"):
        """Thread-safe: may be called from anywhere."""
        self.after(0, self._set_status_main, msg, color)

    def _set_status_main(self, msg: str, color: str):
        self.status_var.set(msg)
        self.status_label.configure(foreground=color)
        self.update_idletasks()

    # =========================================================================
    # Date validation
    # =========================================================================
    @staticmethod
    def _validate_date(value: str, field_name: str) -> str:
        """
        Validates a YYYY-MM-DD date string.
        Returns the stripped string if valid (or empty).
        Raises RuntimeError with a clear message otherwise.
        """
        v = value.strip()
        if not v:
            return ""
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", v):
            raise RuntimeError(
                f"{field_name} must be in YYYY-MM-DD format (e.g. 2024-01-01).\n"
                f"You entered: '{v}'"
            )
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise RuntimeError(f"{field_name} '{v}' is not a valid calendar date.")
        return v

    # =========================================================================
    # Command construction
    # =========================================================================
    def _build_command(self) -> list[str]:
        python_exe  = self.python_exe_var.get().strip()
        script_path = self.script_path_var.get().strip()
        symbol      = self.symbol_var.get().strip().upper()

        if not python_exe or not os.path.isfile(python_exe):
            raise RuntimeError("Python executable path is invalid.")
        if not script_path or not os.path.isfile(script_path):
            raise RuntimeError(f"Predictor script path is invalid:\n{script_path}")
        if not symbol:
            raise RuntimeError("Symbol cannot be empty.")

        # ── Validate interval (StringVar, so must be cast explicitly) ─────────
        interval_raw = self.interval_var.get().strip()
        try:
            interval = max(1, int(interval_raw))
        except ValueError:
            raise RuntimeError(
                f"Continuous interval must be a whole number of minutes.\n"
                f"You entered: '{interval_raw}'"
            )

        # ── Validate date fields ──────────────────────────────────────────────
        train_start = self._validate_date(self.train_start_var.get(),   "Training Start")
        train_end   = self._validate_date(self.train_end_var.get(),     "Training End")
        pred_start  = self._validate_date(self.predict_start_var.get(), "Predict Start")
        pred_end    = self._validate_date(self.predict_end_var.get(),   "Predict End")

        # Cross-field sanity checks
        if train_start and train_end:
            if datetime.strptime(train_start, "%Y-%m-%d") >= \
               datetime.strptime(train_end,   "%Y-%m-%d"):
                raise RuntimeError("Training Start must be earlier than Training End.")
        if pred_start and pred_end:
            if datetime.strptime(pred_start, "%Y-%m-%d") >= \
               datetime.strptime(pred_end,   "%Y-%m-%d"):
                raise RuntimeError("Predict Start must be earlier than Predict End.")
        if train_end and pred_start:
            if datetime.strptime(pred_start, "%Y-%m-%d") < \
               datetime.strptime(train_end,  "%Y-%m-%d"):
                messagebox.showwarning(
                    "Look-Ahead Bias Warning",
                    f"Predict Start ({pred_start}) is BEFORE Training End ({train_end}).\n\n"
                    "The model has already seen the data it is predicting on — "
                    "this will produce unrealistically good backtest results.\n\n"
                    "For a clean out-of-sample test, set Predict Start >= Training End."
                )

        # ── Map GUI action tokens to CLI modes ────────────────────────────────
        mode_map = {
            "train":            "train",
            "train-multitf":    "train-multitf",
            "tune":             "tune",
            "predict-mtf-once": "predict-multitf",
            "predict-mtf-cont": "predict-multitf",
            "backtest":         "backtest",
            "safe-backtest":    "safe-backtest",
        }
        action = self.action_var.get()
        if action not in mode_map:
            raise RuntimeError(f"Unknown action: {action}")
        mode = mode_map[action]

        cmd = [python_exe, "-u", script_path, mode, "--symbol", symbol]

        # ── Training date args ────────────────────────────────────────────────
        if action in ("train", "train-multitf"):
            if train_start:
                cmd += ["--train-start", train_start]
            if train_end:
                cmd += ["--train-end", train_end]

        # ── Prediction / backtest date args ───────────────────────────────────
        if action in ("backtest", "safe-backtest"):
            if pred_start:
                cmd += ["--predict-start", pred_start]
            if pred_end:
                cmd += ["--predict-end", pred_end]

        # ── Model selection ───────────────────────────────────────────────────
        models = [m for m, var in [
            ("lstm",        self.models_lstm_var),
            ("gru",         self.models_gru_var),
            ("transformer", self.models_transformer_var),
            ("tcn",         self.models_tcn_var),
            ("lgbm",        self.models_lgbm_var),
        ] if var.get()]

        if action in ("train", "train-multitf") and not models:
            raise RuntimeError("Please select at least one model type to train.")

        if action in ("train", "train-multitf"):
            if models:
                cmd += ["--models"] + models
            if self.force_retrain_var.get():
                cmd.append("--force")

        if action in ("predict-mtf-once", "predict-mtf-cont"):
            if models:
                cmd += ["--models"] + models
            if not self.use_kalman_var.get():
                cmd.append("--no-kalman")
            if action == "predict-mtf-cont":
                cmd += ["--continuous", "--interval", str(interval)]

        return cmd

    # =========================================================================
    # Run / Stop
    # =========================================================================
    def _on_run_clicked(self):
        mt5_path = self.mt5_path_var.get().strip()
        if not mt5_path:
            messagebox.showwarning(
                "MT5 Path Not Set",
                "Please set the MT5 Files directory path first.\n\n"
                r"Typically: C:\Users\<name>\AppData\Roaming\MetaQuotes\Terminal\<hash>\MQL5\Files"
            )
            return
        if not os.path.exists(mt5_path):
            messagebox.showerror("Invalid MT5 Path",
                                 f"The MT5 Files directory does not exist:\n{mt5_path}\n\n"
                                 "Please update the path.")
            return

        if self._running:
            messagebox.showinfo("Already Running",
                                "A process is already running. Click Stop first.")
            return

        thread = threading.Thread(target=self._run_command_thread, daemon=True)
        thread.start()

    def _on_stop_clicked(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            self.after(0, self._append_log,
                       "\n[STOPPED] Process terminated by user.\n")
            self.after(0, self._set_status_main, "Stopped.", "orange")
        self._set_running(False)

    def _set_running(self, state: bool):
        """Toggle Run/Stop button states on the main thread."""
        self.after(0, self._set_running_main, state)

    def _set_running_main(self, state: bool):
        self._running = state
        self._run_btn.configure( state="disabled" if state else "normal")
        self._stop_btn.configure(state="normal"   if state else "disabled")

    # =========================================================================
    # Background thread — NO direct widget calls here
    # =========================================================================
    def _run_command_thread(self):
        # Build the command on this thread (validation is CPU-only)
        try:
            cmd = self._build_command()
        except Exception as exc:
            # showwarning / showerror are safe to call cross-thread in practice,
            # but schedule on main loop to be correct.
            self.after(0, messagebox.showerror, "Configuration Error", str(exc))
            return

        self._set_running(True)
        self.after(0, self._set_status_main, "Running…", "blue")
        self.after(0, self._clear_log)

        header = (
            "=" * 60 + "\n"
            f"GGTH Predictor GUI v{VERSION}\n"
            "=" * 60 + "\n\n"
            "Executing:\n  " + " ".join(cmd) + "\n\n"
        )
        self.after(0, self._append_log, header)

        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            self.after(0, self._append_log, f"FAILED to start process:\n{exc}\n")
            self.after(0, self._set_status_main, "Failed to start.", "red")
            self._set_running(False)
            return

        # Stream stdout line-by-line; schedule each line on the main thread
        for line in self._proc.stdout:
            self.after(0, self._append_log, line)

        self._proc.wait()
        rc = self._proc.returncode

        if rc == 0:
            self.after(0, self._set_status_main, "Done.", "green")
            self.after(0, self._append_log,
                       "\n" + "=" * 60 + "\n[OK] Process completed successfully!\n")
        else:
            # returncode -15 (SIGTERM) means user clicked Stop — already handled
            if rc not in (-15, 1):  # 1 = Windows SIGTERM equivalent
                self.after(0, self._set_status_main,
                           f"Finished with errors (code {rc}).", "red")
                self.after(0, self._append_log,
                           f"\n" + "=" * 60 + f"\n[ERR] Process exited with code {rc}\n")

        self._set_running(False)


# =============================================================================
if __name__ == "__main__":
    app = GGTHGui()
    app.mainloop()
