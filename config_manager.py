"""
GGTH Predictor Configuration Manager v2.1
Handles loading and saving configuration, especially MT5 Files path.
Updated for unified_predictor_v8.py

Fixes v2.0 → v2.1:
  - set() renamed to set_value() — 'set' shadowed the Python built-in
  - get_default_models() fallback now references DEFAULT_CONFIG instead of a
    hardcoded list that contradicted it (was all-5 models vs 3 in DEFAULT_CONFIG)
  - auto_detect_mt5_path() sorts candidates by mtime so the most-recently-used
    terminal wins when multiple MT5 installations exist
  - __file__ wrapped in os.path.abspath() so the config path is stable
    regardless of the working directory at launch time
  - Singleton exposes reload() so callers can refresh after an on-disk change
  - _load_config() surfaces a clear ValueError on malformed JSON instead of
    silently falling back to defaults and masking the real problem
"""

import os
import glob
import json
from typing import Optional, Dict, Any

# Stable path to this file's directory — safe regardless of CWD at launch time
_HERE = os.path.dirname(os.path.abspath(__file__))


class ConfigManager:
    """Manages configuration for GGTH Predictor."""

    DEFAULT_CONFIG: Dict[str, Any] = {
        "mt5_files_path":             "",
        "version":                    "2.0",
        "models_dir":                 "models",
        "use_kalman":                 True,
        "default_symbol":             "EURUSD",
        "prediction_interval_minutes": 60,
        "default_models":             ["lstm", "transformer", "lgbm"],
        "available_models":           ["lstm", "gru", "transformer", "tcn", "lgbm"],
    }

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialise the config manager.

        Args:
            config_path: Path to config.json.
                         Defaults to config.json in the same directory as this
                         script (resolved with abspath, so CWD doesn't matter).
        """
        if config_path is None:
            config_path = os.path.join(_HERE, "config.json")

        self.config_path = config_path
        self.config = self._load_config()

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from disk and merge with defaults.

        Raises:
            ValueError: If the file exists but contains invalid JSON.
                        (Callers should catch this and report it clearly rather
                        than silently falling back to defaults.)
        """
        if not os.path.exists(self.config_path):
            return self.DEFAULT_CONFIG.copy()

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                on_disk = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"config.json is malformed and cannot be parsed.\n"
                f"File: {self.config_path}\n"
                f"Detail: {exc}\n\n"
                "Fix or delete config.json, then restart."
            ) from exc
        except OSError as exc:
            print(f"Warning: could not read {self.config_path}: {exc}. "
                  "Using default configuration.")
            return self.DEFAULT_CONFIG.copy()

        # Merge: defaults first so new keys added to DEFAULT_CONFIG are picked up
        merged = self.DEFAULT_CONFIG.copy()
        merged.update(on_disk)
        return merged

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def reload(self) -> None:
        """Re-read config.json from disk. Useful when the file may have changed."""
        self.config = self._load_config()

    def save_config(self) -> bool:
        """
        Persist the current in-memory config to disk.

        Returns:
            True on success, False on failure (error is printed).
        """
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=2)
            return True
        except OSError as exc:
            print(f"Error: could not save config to {self.config_path}: {exc}")
            return False

    def get(self, key: str, default: Any = None) -> Any:
        """Return a configuration value."""
        return self.config.get(key, default)

    def set_value(self, key: str, value: Any) -> None:
        """
        Set a configuration value in memory.
        Call save_config() afterwards to persist.

        Note: previously named set(), which shadowed the Python built-in.
        """
        self.config[key] = value

    # -------------------------------------------------------------------------
    # MT5 path helpers
    # -------------------------------------------------------------------------
    def get_mt5_files_path(self) -> str:
        """
        Return the validated MT5 Files directory path.

        Raises:
            ValueError: If the path is not configured or does not exist on disk.
        """
        path = self.config.get("mt5_files_path", "")

        if not path:
            raise ValueError(
                "MT5 Files path is not configured.\n\n"
                "Run setup_wizard.bat, or manually create config.json with:\n"
                "{\n"
                '  "mt5_files_path": "C:\\\\Users\\\\YourName\\\\AppData\\\\Roaming\\\\'
                'MetaQuotes\\\\Terminal\\\\HASH\\\\MQL5\\\\Files"\n'
                "}"
            )

        if not os.path.exists(path):
            raise ValueError(
                f"MT5 Files path does not exist: {path}\n\n"
                "Please update config.json with the correct path."
            )

        return path

    def set_mt5_files_path(self, path: str) -> bool:
        """
        Validate, set, and immediately persist the MT5 Files directory path.

        Args:
            path: Absolute path to the MT5 Files directory.

        Returns:
            True if the path exists and was saved, False otherwise.
        """
        if not os.path.exists(path):
            print(f"Error: directory does not exist: {path}")
            return False

        self.config["mt5_files_path"] = path
        return self.save_config()

    def auto_detect_mt5_path(self) -> Optional[str]:
        """
        Attempt to auto-detect the MT5 Files directory.

        When multiple Terminal installations exist (e.g. demo + live accounts),
        the one modified most recently is returned — that is most likely the
        active one.

        Returns:
            Absolute path if found, None otherwise.
        """
        appdata = os.environ.get("APPDATA", "")
        if appdata:
            terminal_base = os.path.join(appdata, "MetaQuotes", "Terminal")
            if os.path.exists(terminal_base):
                matches = glob.glob(
                    os.path.join(terminal_base, "*", "MQL5", "Files")
                )
                if matches:
                    # Sort by directory mtime descending — most recently used first
                    matches.sort(key=os.path.getmtime, reverse=True)
                    return matches[0]

        # Fallback: standard Program Files installation
        program_files = os.environ.get("PROGRAMFILES", r"C:\Program Files")
        pf_path = os.path.join(program_files, "MetaTrader 5", "MQL5", "Files")
        if os.path.exists(pf_path):
            return pf_path

        return None

    # -------------------------------------------------------------------------
    # Model helpers
    # -------------------------------------------------------------------------
    def get_default_models(self) -> list:
        """
        Return the configured default model types for training.

        The fallback now references DEFAULT_CONFIG instead of a separate
        hardcoded list that previously contradicted it.
        """
        return self.config.get(
            "default_models", self.DEFAULT_CONFIG["default_models"]
        )

    def get_available_models(self) -> list:
        """Return all available model types."""
        return self.config.get(
            "available_models", self.DEFAULT_CONFIG["available_models"]
        )

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------
    def print_config(self) -> None:
        """Print the current configuration to stdout."""
        print("\n" + "=" * 50)
        print("  GGTH Predictor Configuration v2.1")
        print("=" * 50)
        print(f"Config file: {self.config_path}")
        print("\nSettings:")
        for key, value in self.config.items():
            print(f"  {key}: {value}")
        print("=" * 50 + "\n")


# =============================================================================
# Singleton
# =============================================================================
_config_instance: Optional[ConfigManager] = None


def get_config() -> ConfigManager:
    """
    Return the global ConfigManager instance (singleton).

    Call get_config().reload() if you need to pick up on-disk changes made
    after the first call.
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigManager()
    return _config_instance


def get_mt5_files_path() -> str:
    """Convenience wrapper: return the validated MT5 Files path."""
    return get_config().get_mt5_files_path()


# =============================================================================
# CLI utility
# =============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="GGTH Predictor Configuration Utility v2.1"
    )
    parser.add_argument("--set-mt5-path",  help="Set MT5 Files directory path")
    parser.add_argument("--auto-detect",   action="store_true",
                        help="Auto-detect MT5 path (picks most-recently-used terminal)")
    parser.add_argument("--show",          action="store_true",
                        help="Show current configuration")
    parser.add_argument("--list-models",   action="store_true",
                        help="List available model types")
    parser.add_argument("--reload",        action="store_true",
                        help="Force re-read of config.json (useful after manual edits)")

    args = parser.parse_args()

    try:
        config = get_config()
    except ValueError as e:
        print(f"[ERROR] {e}")
        raise SystemExit(1)

    if args.reload:
        config.reload()
        print("Configuration reloaded from disk.")

    if args.set_mt5_path:
        if config.set_mt5_files_path(args.set_mt5_path):
            print(f"✓ MT5 path set to: {args.set_mt5_path}")
        else:
            print("✗ Failed to set MT5 path")

    if args.auto_detect:
        path = config.auto_detect_mt5_path()
        if path:
            print(f"Found MT5 installation at: {path}")
            response = input("Use this path? (y/n): ")
            if response.lower() == "y":
                if config.set_mt5_files_path(path):
                    print("✓ MT5 path configured successfully")
        else:
            print("Could not auto-detect MT5 installation.")

    if args.list_models:
        defaults = config.get_default_models()
        print("\nAvailable model types:")
        for model in config.get_available_models():
            tag = " (default)" if model in defaults else ""
            print(f"  - {model}{tag}")

    if args.show or not any(
        [args.set_mt5_path, args.auto_detect, args.list_models, args.reload]
    ):
        config.print_config()