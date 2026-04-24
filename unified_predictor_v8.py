"""
Multi-Timeframe Predictor v8.2
Author: Jason Rusk jason.w.rusk@gmail.com
Copyright 2026

Fixes/Additions in v8.2:
- FIXED: TCN architecture - added residual connections and wider receptive field
  * Original TCN only had 15-step receptive field for 60-step input
  * Now covers full 60-step lookback with dilations [1,2,4,8,16]
  * Added skip connections to prevent gradient issues
- ADDED: Log return clamping safeguard for all models
  * Prevents extreme predictions from any single model
  * Clamps at 0.5% for 1H, 1% for 4H, 2% for 1D
- FIXED: Macro data handling - gracefully continues if DXY/SPX unavailable

Previous v8.1 fixes:
- Integrated GaussianHMM for Market Regime Detection
- Added Correlation Engine for DXY (USDX) and SPX500
- FIXED: Walk-Forward Backtesting to prevent "Look-Ahead" data leakage
- FIXED: Indentation bug in download_data() validation loop
- ADDED: GRU model option for ensemble
- ADDED: Attention mechanism to LSTM for better performance

"""

import sys
import os
import io
import subprocess
import warnings

# ── Windows cp1252 fix ────────────────────────────────────────────────────────
# The default Windows console encoding (cp1252) cannot encode Unicode arrows,
# em-dashes, or other non-Latin characters used in log output.  Force UTF-8
# before any print() call so we never get a UnicodeEncodeError that silently
# kills regime detection or any other diagnostic path.
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', write_through=True)
if hasattr(sys.stderr, 'buffer'):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', write_through=True)
import time
import json
import pickle
import argparse
import glob
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import pandas as pd

# --- NEW QUANT LIBRARIES ---
try:
    from hmmlearn.hmm import GaussianHMM
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "hmmlearn"])
    from hmmlearn.hmm import GaussianHMM

warnings.filterwarnings('ignore')

# --- CONFIGURATION: MT5 PATH ---
try:
    from config_manager import get_mt5_files_path as get_config_mt5_path, get_config
except ImportError:
    print("=" * 80)
    print("ERROR: config_manager.py not found!")
    print("=" * 80)
    sys.exit(1)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ── Refactored modules ────────────────────────────────────────────────────────
from logger import get_logger, log_startup_banner
from model_builders import build_dl_model, KalmanFilter, TransformerBlock, AttentionLayer


def install_package(package_name: str, pip_name: Optional[str] = None) -> None:
    try:
        __import__(package_name)
    except (ImportError, ModuleNotFoundError):
        install_name = pip_name or package_name
        subprocess.check_call([sys.executable, "-m", "pip", "install", install_name])


required_packages = [
    ('MetaTrader5', 'MetaTrader5'), ('pandas', 'pandas'), ('numpy', 'numpy'),
    ('tensorflow', 'tensorflow'), ('sklearn', 'scikit-learn'),
    ('lightgbm', 'lightgbm'), ('keras_tuner', 'keras-tuner')
]

for package, pip_name in required_packages:
    install_package(package, pip_name)

import MetaTrader5 as mt5
import tensorflow as tf
import keras
from keras import layers
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import RobustScaler
import lightgbm as lgb
import keras_tuner as kt

np.random.seed(42)
tf.random.set_seed(42)
tf.config.run_functions_eagerly(False)


# --- Helper Classes ---

# KalmanFilter, TransformerBlock, AttentionLayer -> see model_builders.py

# --- Main Predictor Class ---

class UnifiedLSTMPredictor:
    def __init__(self, symbol: str = "EURUSD", related_symbols: Optional[List[str]] = None,
                 ensemble_model_types: Optional[List[str]] = None, use_kalman: bool = False,
                 use_multitimeframe: bool = False,
                 train_start: Optional[str] = None,
                 train_end:   Optional[str] = None,
                 predict_start: Optional[str] = None,
                 predict_end:   Optional[str] = None):
        self.log = get_logger("ggth.predictor")
        log_startup_banner("v8")
        self.symbol = symbol.upper()
        # --- NEW MACRO SYMBOLS ---
        self.dxy_symbol = "USDX"
        self.spx_symbol = "SPX500"

        self.related_symbols = related_symbols or ["AUDUSD", "GBPUSD"]
        self.ensemble_model_types = ensemble_model_types if ensemble_model_types is not None else ['lstm', 'transformer', 'lgbm']
        self.num_ensemble_models = len(self.ensemble_model_types)
        self.lookback_periods = 60
        self.base_path = self.get_mt5_files_path()
        self.use_kalman = use_kalman
        self.use_multitimeframe = use_multitimeframe

        # --- DATE RANGE FILTERS ---
        # Parse ISO date strings (YYYY-MM-DD) into datetime objects when provided
        def _parse_date(s: Optional[str]) -> Optional[datetime]:
            if s is None:
                return None
            try:
                return datetime.strptime(s, "%Y-%m-%d")
            except ValueError:
                raise ValueError(f"Date '{s}' must be in YYYY-MM-DD format.")

        self.train_start:   Optional[datetime] = _parse_date(train_start)
        self.train_end:     Optional[datetime] = _parse_date(train_end)
        self.predict_start: Optional[datetime] = _parse_date(predict_start)
        self.predict_end:   Optional[datetime] = _parse_date(predict_end)

        # Validate: predict window must not overlap training window
        if self.train_end and self.predict_start:
            if self.predict_start < self.train_end:
                raise ValueError(
                    f"--predict-start ({predict_start}) must be >= --train-end ({train_end}) "
                    f"to avoid look-ahead bias."
                )

        # File paths
        self.predictions_file = os.path.join(self.base_path, f"{self.symbol}_predictions_multitf.json")
        self.status_file = os.path.join(self.base_path, f"lstm_status_{self.symbol}.json")
        self.feature_scaler_path = os.path.join(self.base_path, f"feature_scaler_{self.symbol}.pkl")
        self.target_scaler_path = os.path.join(self.base_path, f"target_scaler_{self.symbol}.pkl")
        self.selected_features_path = os.path.join(self.base_path, f"selected_features_{self.symbol}.json")
        self.pending_eval_path = os.path.join(self.base_path, f"pending_evaluations_{self.symbol}.json")
        self.tuner_dir = os.path.join(self.base_path, 'tuner_results')
        # Manifest records exact train window so backtest generation can verify no overlap
        self.cutoff_manifest_path = os.path.join(self.base_path, f"training_cutoff_{self.symbol}.json")

        self.target_column = 'fwd_log_return_1h'
        self.feature_cols: Optional[List[str]] = None
        self.models: Dict[str, Any] = {}
        self.feature_scaler = RobustScaler()
        self.target_scaler = RobustScaler()

        self.models_by_timeframe: Dict[str, Dict[str, Any]] = {}
        self.scalers_by_timeframe: Dict[str, Tuple[RobustScaler, RobustScaler]] = {}

        self.kalman_config = {
            "1H": {"Q": 0.00001, "R": 0.01},
            "4H": {"Q": 0.00005, "R": 0.02},
            "1D": {"Q": 0.0001, "R": 0.05}
        }
        self.kalman_filters = {tf: KalmanFilter(c["Q"], c["R"]) for tf, c in self.kalman_config.items()}

        self.previous_predictions = {tf: None for tf in self.kalman_config.keys()}
        self.ema_alpha = 0.3
        # Per-timeframe ensemble weights dict: {"1H": [...], "4H": [...], "1D": [...]}
        # Each list has one weight per model, summing to 1.0.
        # Keeping weights separate per timeframe lets each TF learn which model
        # is strongest on its own horizon (e.g. LGBM on ranging 4H, TCN on trending 1D).
        if self.num_ensemble_models > 0:
            equal_weight = 1.0 / self.num_ensemble_models
            self.ensemble_weights: Dict[str, List[float]] = {
                tf_key: [equal_weight] * self.num_ensemble_models
                for tf_key in self.kalman_config.keys()  # "1H", "4H", "1D"
            }
        else:
            self.ensemble_weights: Dict[str, List[float]] = {}
        self.prediction_history = {tf: [] for tf in self.kalman_config.keys()}
        self.ensemble_lookback = 20
        self.ensemble_learning_rate = 0.1

        self.initialize_mt5()
        self.ensure_symbols_selected()

    def get_mt5_files_path(self) -> str:
        mt5_path = get_config_mt5_path()
        if not os.path.exists(mt5_path):
            sys.exit(1)
        return mt5_path

    def initialize_mt5(self) -> None:
        if not mt5.initialize():
            sys.exit(1)
        print(f"Connected to MT5: {mt5.account_info().login}")

    def ensure_symbols_selected(self):
        """Ensures DXY and SP500 are in Market Watch."""
        for s in [self.symbol, self.dxy_symbol, self.spx_symbol] + self.related_symbols:
            mt5.symbol_select(s, True)

    def _download_macro_data(self, bars: int = 300) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Safely download macro data (DXY and SPX).
        Returns (df_dxy, df_spx) - either can be None if unavailable.
        """
        df_dxy = None
        df_spx = None
        
        # Try to get DXY data - check multiple possible symbol names
        dxy_symbols = [self.dxy_symbol, "USDX", "DXY", "DX", "US Dollar Index"]
        for sym in dxy_symbols:
            try:
                mt5.symbol_select(sym, True)
                data = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_H1, 0, bars)
                if data is not None and len(data) > 0:
                    df_dxy = pd.DataFrame(data)
                    df_dxy['time'] = pd.to_datetime(df_dxy['time'], unit='s')
                    df_dxy.set_index('time', inplace=True)
                    self.dxy_symbol = sym  # Update to working symbol
                    break
            except Exception:
                continue
        
        # Try to get SPX data - check multiple possible symbol names
        spx_symbols = [self.spx_symbol, "SPX500", "SP500", "US500", "SPX", "S&P500", "US500.cash"]
        for sym in spx_symbols:
            try:
                mt5.symbol_select(sym, True)
                data = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_H1, 0, bars)
                if data is not None and len(data) > 0:
                    df_spx = pd.DataFrame(data)
                    df_spx['time'] = pd.to_datetime(df_spx['time'], unit='s')
                    df_spx.set_index('time', inplace=True)
                    self.spx_symbol = sym  # Update to working symbol
                    break
            except Exception:
                continue
        
        return df_dxy, df_spx

    def _detect_regime(self, df_main: pd.DataFrame) -> str:
        """
        Classify the current market regime from recent price action alone.

        Three regimes are recognised:
            "trending"  — price is moving directionally; trend-following models
                          (TCN, Transformer) tend to outperform.
            "volatile"  — short-term volatility has spiked well above its norm;
                          all models are less reliable, but LGBM degrades least.
            "ranging"   — low directional movement and normal vol; LGBM and
                          mean-reversion logic holds up best.

        The classification uses two simple statistics computed on the close
        series so it requires no extra data downloads and runs in microseconds.

        Args:
            df_main: H1 OHLCV DataFrame (index = datetime, 'close' column required).

        Returns:
            One of "trending", "volatile", or "ranging".
        """
        try:
            close = df_main['close']
            if len(close) < 40:
                return 'unknown'

            log_ret = np.log(close / close.shift(1)).dropna()

            # Short-term vs long-term volatility ratio
            vol_short = log_ret.iloc[-20:].std()
            vol_long  = log_ret.iloc[-100:].std() if len(log_ret) >= 100 else vol_short

            # Normalised directional slope over the last 20 bars
            # (mean log-return divided by short-term vol -> a dimensionless Z-score)
            mean_ret_short = log_ret.iloc[-20:].mean()
            trend_z = (mean_ret_short / vol_short) if vol_short > 0 else 0.0

            # Regime thresholds (calibrated for EURUSD H1; adjust if needed)
            VOL_SPIKE_RATIO = 1.6   # vol_short > 1.6 × vol_long -> volatile
            TREND_Z_THRESH  = 1.2   # |trend_z| > 1.2 -> trending

            if vol_short > VOL_SPIKE_RATIO * vol_long:
                regime = 'volatile'
            elif abs(trend_z) > TREND_Z_THRESH:
                regime = 'trending'
            else:
                regime = 'ranging'

            print(f"   Regime detection: vol_ratio={vol_short/max(vol_long,1e-10):.2f}, "
                  f"trend_z={trend_z:.2f} -> {regime.upper()}")
            return regime

        except Exception as e:
            print(f"   Warning: Regime detection failed ({e}), defaulting to 'unknown'")
            return 'unknown'

    def get_market_context(self, df_main, df_dxy, df_spx):
        """
        Refined Intermarket Veto Logic.
        Handles missing macro data gracefully.
        Adds 'regime' key (trending / ranging / volatile) for ensemble biasing.
        """
        # Detect regime from price action before any macro check
        regime = self._detect_regime(df_main)

        # Default return if macro data is unavailable
        default_context = {
            "veto_active": False,
            "reasons": [],
            "z_score": 0.0,
            "dxy_corr": 0.0,
            "macro_data_available": False,
            "regime": regime,
        }
        
        # Check if we have valid macro data
        dxy_valid = (df_dxy is not None and 
                     not df_dxy.empty and 
                     'close' in df_dxy.columns and 
                     len(df_dxy) >= 24)
        spx_valid = (df_spx is not None and 
                     not df_spx.empty and 
                     'close' in df_spx.columns and 
                     len(df_spx) >= 24)
        
        if not dxy_valid or not spx_valid:
            missing = []
            if not dxy_valid:
                missing.append(f"DXY ({self.dxy_symbol})")
            if not spx_valid:
                missing.append(f"SPX ({self.spx_symbol})")
            print(f"   Note: Macro data unavailable for {', '.join(missing)} - skipping intermarket analysis")
            return default_context
        
        try:
            # 1. Calculate Z-Score for Risk Sentiment (SPX)
            spx_returns = df_spx['close'].pct_change(24)
            if spx_returns.std() == 0:
                z_score_risk = 0.0
            else:
                z_score_risk = (spx_returns.iloc[-1] - spx_returns.mean()) / spx_returns.std()

            # 2. Institutional Divergence (SMT)
            # Check if DXY and Main Pair are moving in the SAME direction (Abnormal)
            dxy_slope = df_dxy['close'].iloc[-5:].diff().mean()
            main_slope = df_main['close'].iloc[-5:].diff().mean()

            # If slopes are both positive, DXY and EURUSD are rising together (Divergent)
            is_divergent = (dxy_slope * main_slope) > 0

            # 3. Correlation Strength
            current_corr = df_main['close'].rolling(24).corr(df_dxy['close']).iloc[-1]
            
            # Handle NaN correlation
            if pd.isna(current_corr):
                current_corr = 0.0

            # --- VETO SUMMARY ---
            veto_reasons = []
            if z_score_risk < -2.0:
                veto_reasons.append("Extreme Risk-Off Panic")
            if is_divergent:
                veto_reasons.append("Macro Divergence (SMT)")
            if current_corr > -0.70:
                veto_reasons.append("Weak Inverse Correlation")

            return {
                "veto_active": len(veto_reasons) > 0,
                "reasons": veto_reasons,
                "z_score": round(float(z_score_risk), 2),
                "dxy_corr": round(float(current_corr), 4),
                "macro_data_available": True,
                "regime": regime,
            }
            
        except Exception as e:
            print(f"   Warning: Error in market context analysis: {e}")
            return default_context

    def download_data(self, bars: int = 35000,
                      date_from: Optional[datetime] = None,
                      date_to:   Optional[datetime] = None) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Download multi-timeframe data from MT5.

        When date_from / date_to are provided the method uses copy_rates_range()
        so the returned data is strictly bounded by those dates.  This is the
        mechanism that prevents training data from leaking into the test window.

        Args:
            bars:      Number of bars (used only when no date range is given)
            date_from: Inclusive start datetime (UTC)
            date_to:   Inclusive end datetime (UTC)

        Returns:
            Tuple of (df_h1, df_h4, df_d1) DataFrames
        """
        if date_from or date_to:
            # Resolve defaults so copy_rates_range always gets explicit bounds
            _from = date_from or datetime(2000, 1, 1)
            _to   = date_to   or datetime.utcnow()
            range_str = f"{_from.strftime('%Y-%m-%d')} to {_to.strftime('%Y-%m-%d')}"
            print(f"Downloading multi-timeframe data for {self.symbol} [{range_str}]...")
            try:
                raw_h1 = mt5.copy_rates_range(self.symbol, mt5.TIMEFRAME_H1, _from, _to)
                raw_h4 = mt5.copy_rates_range(self.symbol, mt5.TIMEFRAME_H4, _from, _to)
                raw_d1 = mt5.copy_rates_range(self.symbol, mt5.TIMEFRAME_D1, _from, _to)
            except Exception as e:
                print(f"Error downloading data by range: {e}")
                return None, None, None
        else:
            print(f"Downloading multi-timeframe data for {self.symbol} (last {bars} bars)...")
            try:
                raw_h1 = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_H1, 0, bars)
                raw_h4 = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_H4, 0, bars // 4)
                raw_d1 = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_D1, 0, bars // 20)
            except Exception as e:
                print(f"Error downloading data: {e}")
                return None, None, None

        try:
            df_h1 = pd.DataFrame(raw_h1)
            df_h4 = pd.DataFrame(raw_h4)
            df_d1 = pd.DataFrame(raw_d1)

            min_bars = {"H1": 100, "H4": 50, "D1": 20}
            for df, name in [(df_h1, "H1"), (df_h4, "H4"), (df_d1, "D1")]:
                required = min_bars.get(name, 100)
                if df is None or df.empty or len(df) < required:
                    print(f"Failed to download {name} data for {self.symbol} "
                          f"(got {len(df) if df is not None else 0}, need {required})")
                    return None, None, None
                df['time'] = pd.to_datetime(df['time'], unit='s')
                df.set_index('time', inplace=True)
                df.rename(columns={'tick_volume': 'volume'}, inplace=True)

            print(f"Downloaded {len(df_h1)} H1 bars from {df_h1.index.min()} to {df_h1.index.max()}")
            return df_h1, df_h4, df_d1

        except Exception as e:
            print(f"Error processing downloaded data: {e}")
            return None, None, None

    def create_features(self, df_h1: pd.DataFrame, df_h4: pd.DataFrame, df_d1: pd.DataFrame) -> pd.DataFrame:
        """
        Create a rich multi-timeframe feature set for model training.

        Feature groups
        --------------
        H1 momentum   : log returns (1h / 4h / 1d / 1w) + 5 lagged 1h returns
        H1 volatility : ATR-14, rolling std-14, rolling std-30, vol ratio, BB width/%B
        H1 trend      : SMA-10/20/50, price-to-SMA ratios, SMA crossover ratio,
                        MACD line/signal/hist, ADX-14 with +DI / -DI
        H1 oscillators: RSI-7/14, Stochastic %K/%D
        H4 (reindexed): SMA-20/50, RSI-14, ATR-14, MACD hist,
                        price-to-SMA-20 ratio
        D1 (reindexed): SMA-20/50, RSI-14, price-to-SMA-20 ratio
        Volume/spread : tick volume (H1), spread (when available)
        Time          : hour sin/cos, day-of-week sin/cos,
                        London session flag, NY session flag,
                        London-NY overlap flag
        Targets       : fwd_log_return_1h / 4h / 1d  (excluded from feature cols)

        All indicators are computed with EWM (Wilder smoothing) where applicable
        so they match the definitions used in most professional platforms.
        OHLC raw prices are retained in the DataFrame for target computation but
        are excluded from the feature selection candidate pool in
        perform_feature_selection().

        Args:
            df_h1: H1 OHLCV DataFrame  (index = datetime, renamed tick_volume -> volume)
            df_h4: H4 OHLCV DataFrame
            df_d1: D1 OHLCV DataFrame

        Returns:
            DataFrame aligned to the H1 index with all features + targets.
        """
        print("Creating advanced features...")
        df = df_h1.copy()

        # ── Local indicator helpers ────────────────────────────────────────────
        # All helpers operate on plain pd.Series / DataFrames and return Series.
        # Using EWM with adjust=False approximates Wilder's smoothing (span = period).

        def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
            delta = close.diff()
            gain  = delta.where(delta > 0, 0.0).ewm(span=period, adjust=False).mean()
            loss  = (-delta.where(delta < 0, 0.0)).ewm(span=period, adjust=False).mean()
            return 100 - (100 / (1 + gain / (loss + 1e-8)))

        def _atr(ohlc: pd.DataFrame, period: int = 14) -> pd.Series:
            h, l, c = ohlc['high'], ohlc['low'], ohlc['close']
            tr = pd.concat([h - l,
                            (h - c.shift(1)).abs(),
                            (l - c.shift(1)).abs()], axis=1).max(axis=1)
            return tr.ewm(span=period, adjust=False).mean()

        def _adx(ohlc: pd.DataFrame, period: int = 14):
            """Returns (adx, plus_di, minus_di) as three Series."""
            h, l = ohlc['high'], ohlc['low']
            up   = h.diff()
            down = -l.diff()
            plus_dm  = np.where((up > down) & (up > 0),   up,   0.0)
            minus_dm = np.where((down > up) & (down > 0), down, 0.0)
            atr_s    = _atr(ohlc, period)
            plus_di  = (100 * pd.Series(plus_dm,  index=ohlc.index)
                            .ewm(span=period, adjust=False).mean() / (atr_s + 1e-8))
            minus_di = (100 * pd.Series(minus_dm, index=ohlc.index)
                            .ewm(span=period, adjust=False).mean() / (atr_s + 1e-8))
            dx  = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-8))
            adx = dx.ewm(span=period, adjust=False).mean()
            return adx, plus_di, minus_di

        def _macd(close: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9):
            """Returns (macd_line, signal, histogram) as three Series."""
            line   = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
            signal = line.ewm(span=sig, adjust=False).mean()
            return line, signal, line - signal

        def _bollinger(close: pd.Series, period: int = 20, n_std: float = 2.0):
            """Returns (bb_width, bb_pct) — width normalised by SMA, %B in [0,1]."""
            sma   = close.rolling(period).mean()
            std   = close.rolling(period).std()
            upper = sma + n_std * std
            lower = sma - n_std * std
            width = (upper - lower) / (sma + 1e-8)
            pct   = (close - lower) / (upper - lower + 1e-8)
            return width, pct

        def _stochastic(ohlc: pd.DataFrame, k: int = 14, d: int = 3):
            """Returns (%K, %D) both in [0, 100]."""
            lo  = ohlc['low'].rolling(k).min()
            hi  = ohlc['high'].rolling(k).max()
            pct_k = 100 * (ohlc['close'] - lo) / (hi - lo + 1e-8)
            return pct_k, pct_k.rolling(d).mean()

        def _reindex(series: pd.Series) -> pd.Series:
            """Forward-fill a lower-timeframe series onto the H1 index."""
            return series.reindex(df.index, method='ffill')

        # ── Backward-looking returns (momentum features — valid inputs) ────────
        df['log_return_1h'] = np.log(df['close'] / df['close'].shift(1))
        df['log_return_4h'] = np.log(df['close'] / df['close'].shift(4))
        df['log_return_1d'] = np.log(df['close'] / df['close'].shift(24))
        df['log_return_1w'] = np.log(df['close'] / df['close'].shift(168))

        # Lagged 1h log-returns (give the model recent direction history)
        for lag in [1, 2, 3, 5, 10]:
            df[f'log_return_lag_{lag}'] = df['log_return_1h'].shift(lag)

        # ── Volatility ────────────────────────────────────────────────────────
        df['atr_14_h1']      = _atr(df, 14)
        df['volatility_14']  = df['log_return_1h'].rolling(14).std()
        df['volatility_30']  = df['log_return_1h'].rolling(30).std()
        # Ratio of short-to-long vol: >1 -> vol spike (regime signal)
        df['vol_ratio']      = df['volatility_14'] / (df['volatility_30'] + 1e-8)

        # ── Trend — SMA crossovers and price position ─────────────────────────
        df['sma_10_h1']         = df['close'].rolling(10).mean()
        df['sma_20_h1']         = df['close'].rolling(20).mean()
        df['sma_50_h1']         = df['close'].rolling(50).mean()
        df['close_sma10_ratio'] = df['close'] / (df['sma_10_h1'] + 1e-8) - 1
        df['close_sma20_ratio'] = df['close'] / (df['sma_20_h1'] + 1e-8) - 1
        df['close_sma50_ratio'] = df['close'] / (df['sma_50_h1'] + 1e-8) - 1
        df['sma10_sma20_ratio'] = df['sma_10_h1'] / (df['sma_20_h1'] + 1e-8) - 1  # crossover signal

        # ── MACD ──────────────────────────────────────────────────────────────
        df['macd_line'], df['macd_signal'], df['macd_hist'] = _macd(df['close'])

        # ── Bollinger Bands ───────────────────────────────────────────────────
        df['bb_width'], df['bb_pct'] = _bollinger(df['close'], 20)

        # ── RSI (fast + standard) ─────────────────────────────────────────────
        df['rsi_7_h1']  = _rsi(df['close'], 7)
        df['rsi_14_h1'] = _rsi(df['close'], 14)

        # ── ADX (trend strength + directional components) ─────────────────────
        df['adx_14_h1'], df['plus_di_14_h1'], df['minus_di_14_h1'] = _adx(df, 14)

        # ── Stochastic ────────────────────────────────────────────────────────
        df['stoch_k'], df['stoch_d'] = _stochastic(df, 14, 3)

        # ── Volume ────────────────────────────────────────────────────────────
        if 'volume' in df.columns:
            vol_ma = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / (vol_ma + 1e-8)  # normalised vs. recent average

        # ── Spread (available from MT5 rates) ─────────────────────────────────
        if 'spread' in df.columns:
            spread_ma = df['spread'].rolling(20).mean()
            df['spread_ratio'] = df['spread'] / (spread_ma + 1e-8)

      # ── H4 features (shifted 1 period BEFORE reindexing to prevent look-ahead bias) ──
        df['sma_20_h4']         = _reindex(df_h4['close'].rolling(20).mean().shift(1))
        df['sma_50_h4']         = _reindex(df_h4['close'].rolling(50).mean().shift(1))
        df['rsi_14_h4']         = _reindex(_rsi(df_h4['close'], 14).shift(1))
        df['atr_14_h4']         = _reindex(_atr(df_h4, 14).shift(1))
        _, _, macd_hist_h4      = _macd(df_h4['close'])
        df['macd_hist_h4']      = _reindex(macd_hist_h4.shift(1))
        df['close_sma20_h4_ratio'] = df['close'] / (_reindex(df_h4['close'].rolling(20).mean().shift(1)) + 1e-8) - 1

        # ── D1 features (shifted 1 period BEFORE reindexing to prevent look-ahead bias) ──
        df['sma_20_d1']            = _reindex(df_d1['close'].rolling(20).mean().shift(1))
        df['sma_50_d1']            = _reindex(df_d1['close'].rolling(50).mean().shift(1))
        df['rsi_14_d1']            = _reindex(_rsi(df_d1['close'], 14).shift(1))
        df['close_sma20_d1_ratio'] = df['close'] / (_reindex(df_d1['close'].rolling(20).mean().shift(1)) + 1e-8) - 1
        # ── Time / session features ────────────────────────────────────────────
        # Complete cyclic pairs — a sin alone carries no phase information
        hour = df.index.hour
        dow  = df.index.dayofweek
        df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        df['dow_sin']  = np.sin(2 * np.pi * dow  / 7)
        df['dow_cos']  = np.cos(2 * np.pi * dow  / 7)

        # Session windows (UTC hours) — EURUSD volatility profile is session-driven
        df['session_london']  = ((hour >= 7)  & (hour < 16)).astype(np.int8)
        df['session_ny']      = ((hour >= 13) & (hour < 21)).astype(np.int8)
        df['session_overlap'] = ((hour >= 13) & (hour < 16)).astype(np.int8)

        # ── Forward-looking targets (EXCLUDED from feature cols) ──────────────
        df['fwd_log_return_1h'] = np.log(df['close'].shift(-1)  / df['close'])
        df['fwd_log_return_4h'] = np.log(df['close'].shift(-4)  / df['close'])
        df['fwd_log_return_1d'] = np.log(df['close'].shift(-24) / df['close'])

        # ── Clean ─────────────────────────────────────────────────────────────
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

        if len(df) < self.lookback_periods + 100:
            print(f"WARNING: Only {len(df)} bars after feature creation — may be insufficient for training.")

        # Count candidate features (excludes raw OHLC and targets)
        _exclude = {'open', 'high', 'low', 'close', 'volume', 'spread', 'real_volume',
                    'fwd_log_return_1h', 'fwd_log_return_4h', 'fwd_log_return_1d'}
        n_candidates = sum(1 for c in df.columns if c not in _exclude)
        print(f"   Created {n_candidates} candidate features from {len(df)} bars")
        return df

    def perform_feature_selection(self, df: pd.DataFrame, num_features: int = 30) -> pd.DataFrame:
        """
        Select most important features using LightGBM.

        Args:
            df: DataFrame with all features
            num_features: Number of top features to select (default raised to 30
                          to match the expanded candidate pool from create_features)

        Returns:
            DataFrame with selected features
        """
        print(f"Performing feature selection to find top {num_features} features...")
        target = self.target_column
        # Exclude forward-looking targets and raw OHLC price levels.
        # Backward-looking log returns (log_return_4h, log_return_1d, etc.) are
        # intentionally kept as candidate features — they carry valid momentum signal.
        exclude_cols = {
            target,
            'fwd_log_return_1h', 'fwd_log_return_4h', 'fwd_log_return_1d',
            'close', 'open', 'high', 'low',
        }
        features = [col for col in df.columns if col not in exclude_cols]

        X = df[features]
        y = df[target]

        # Train LightGBM for feature importance
        lgb_train = lgb.Dataset(X, y)
        params = {
            'objective': 'regression_l1',
            'metric': 'mae',
            'n_estimators': 200,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'verbose': -1,
            'n_jobs': -1,
            'seed': 42
        }

        model = lgb.train(params, lgb_train, num_boost_round=100)
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importance()
        }).sort_values('importance', ascending=False)

        self.feature_cols = feature_importance['feature'].head(num_features).tolist()
        print(f"   Selected top {len(self.feature_cols)} features.")

        # Save selected features
        with open(self.selected_features_path, 'w') as f:
            json.dump(self.feature_cols, f)

        return df[self.feature_cols + ['fwd_log_return_1h', 'fwd_log_return_4h', 'fwd_log_return_1d', 'close']]

    def _prepare_sequential_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare sequential data for deep learning models.

        Args:
            df: DataFrame with features

        Returns:
            Tuple of (X_train, y_train, X_val, y_val)
        """
        target_col = self.target_column
        train_size = int(len(df) * 0.70)
        val_size = int(len(df) * 0.15)

        train_df = df[:train_size]
        val_df = df[train_size:train_size + val_size]

        print(f"   Target stats - Mean: {train_df[target_col].mean():.6f}, Std: {train_df[target_col].std():.6f}")

        # Remove extreme outliers (more than 5 std devs)
        mean_return = train_df[target_col].mean()
        std_return = train_df[target_col].std()
        train_df = train_df[abs(train_df[target_col] - mean_return) < (5 * std_return)]
        print(f"   Removed {train_size - len(train_df)} outliers from training data")

        # Initialize scalers
        self.feature_scaler = RobustScaler()
        self.target_scaler = RobustScaler()

        # Scale features and target
        train_scaled_features = self.feature_scaler.fit_transform(train_df[self.feature_cols])
        train_scaled_target = self.target_scaler.fit_transform(train_df[[target_col]])

        val_scaled_features = self.feature_scaler.transform(val_df[self.feature_cols])
        val_scaled_target = self.target_scaler.transform(val_df[[target_col]])

        print(f"   Target scaler - Center: {self.target_scaler.center_[0]:.6f}, Scale: {self.target_scaler.scale_[0]:.6f}")

        def create_sequences(features: np.ndarray, target: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
            """Create sequences for time series prediction."""
            X, y = [], []
            for i in range(lookback, len(features)):
                X.append(features[i - lookback:i])
                y.append(target[i])
            return np.array(X), np.array(y)

        X_train, y_train = create_sequences(train_scaled_features, train_scaled_target, self.lookback_periods)
        X_val, y_val = create_sequences(val_scaled_features, val_scaled_target, self.lookback_periods)

        print(f"   Prepared sequential data: X_train shape {X_train.shape}")
        return X_train, y_train, X_val, y_val

    def _prepare_tabular_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str]]:
        """
        Prepare tabular data for tree-based models.

        Args:
            df: DataFrame with features

        Returns:
            Tuple of (X_train, y_train, X_val, y_val, feature_columns)
        """
        print("Preparing tabular data for tree-based models...")
        df_tabular = df.copy()
        feature_cols_tabular = self.feature_cols[:]

        # Create lag features
        for col in self.feature_cols:
            for lag in [1, 3, 5, 10]:
                new_col = f'{col}_lag_{lag}'
                df_tabular[new_col] = df_tabular[col].shift(lag)
                if new_col not in feature_cols_tabular:
                    feature_cols_tabular.append(new_col)

        df_tabular.dropna(inplace=True)
        final_feature_cols = [c for c in feature_cols_tabular if c in df_tabular.columns]

        # Split data
        train_size = int(len(df_tabular) * 0.85)
        train_df = df_tabular[:train_size]
        val_df = df_tabular[train_size:]

        X_train = train_df[final_feature_cols]
        y_train = train_df[self.target_column]
        X_val = val_df[final_feature_cols]
        y_val = val_df[self.target_column]

        print(f"   Prepared tabular data: X_train shape {X_train.shape}")
        return X_train, y_train, X_val, y_val, final_feature_cols

    def _build_dl_model(self, model_type: str, input_shape: Tuple[int, int], hp: Optional[kt.HyperParameters] = None) -> Model:
        """Delegate to model_builders.build_dl_model — see model_builders.py for architectures."""
        return build_dl_model(model_type, input_shape, hp)

    def tune_hyperparameters(self) -> None:
        """Run hyperparameter tuning for deep learning models."""
        print("\n" + "=" * 60 + "\nStarting Hyperparameter Tuning...\n" + "=" * 60)

        # Download and prepare data
        df_h1, df_h4, df_d1 = self.download_data()
        if df_h1 is None:
            return

        df_features = self.create_features(df_h1, df_h4, df_d1)
        df_selected = self.perform_feature_selection(df_features)
        X_train, y_train, X_val, y_val = self._prepare_sequential_data(df_selected)

        # Save scalers
        with open(self.feature_scaler_path, 'wb') as f:
            pickle.dump(self.feature_scaler, f)
        with open(self.target_scaler_path, 'wb') as f:
            pickle.dump(self.target_scaler, f)

        # Create tuner
        def model_builder(hp):
            return self._build_dl_model('lstm', (X_train.shape[1], X_train.shape[2]), hp=hp)

        tuner = kt.RandomSearch(
            model_builder,
            objective='val_loss',
            max_trials=15,
            executions_per_trial=1,
            directory=self.tuner_dir,
            project_name=f'tuner_{self.symbol}'
        )

        # Run tuning
        tuner.search(
            X_train, y_train,
            epochs=50,
            validation_data=(X_val, y_val),
            callbacks=[EarlyStopping('val_loss', patience=5)]
        )

        # Display results
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        print("\n--- Best Hyperparameters Found ---")
        for param, value in best_hps.values.items():
            print(f"{param}: {value}")
        print("---------------------------------\n")
        print("Tuning complete. Re-run with 'train --force' to use these new settings.")

    # ------------------------------------------------------------------
    # Training cutoff manifest
    # ------------------------------------------------------------------
    def _save_cutoff_manifest(self, actual_train_start: Optional[datetime],
                               actual_train_end: Optional[datetime]) -> None:
        """
        Persist the exact training window to disk so that backtest / predict
        commands can always verify they're not overlapping with training data.
        """
        manifest = {
            "symbol":       self.symbol,
            "train_start":  actual_train_start.strftime("%Y-%m-%d") if actual_train_start else None,
            "train_end":    actual_train_end.strftime("%Y-%m-%d")   if actual_train_end   else None,
            "trained_at":   datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            "models":       self.ensemble_model_types,
        }
        with open(self.cutoff_manifest_path, 'w') as f:
            json.dump(manifest, f, indent=4)
        print(f"\n[MANIFEST] Training cutoff saved: {self.cutoff_manifest_path}")
        print(f"           Train window: {manifest['train_start']} -> {manifest['train_end']}")

    def _load_cutoff_manifest(self) -> Optional[Dict]:
        """Load the training cutoff manifest if it exists."""
        if os.path.exists(self.cutoff_manifest_path):
            with open(self.cutoff_manifest_path, 'r') as f:
                return json.load(f)
        return None

    def _warn_if_predict_overlaps_training(self) -> None:
        """
        Print a warning (but don't abort) if the requested prediction window
        overlaps with the recorded training window.
        """
        manifest = self._load_cutoff_manifest()
        if not manifest or not manifest.get("train_end"):
            return
        train_end_dt = datetime.strptime(manifest["train_end"], "%Y-%m-%d")
        pred_start   = self.predict_start or datetime(2000, 1, 1)
        if pred_start < train_end_dt:
            print("\n" + "!" * 70)
            print("  WARNING: predict-start is INSIDE the training window.")
            print(f"  Training ended:     {manifest['train_end']}")
            print(f"  Prediction starts:  {pred_start.strftime('%Y-%m-%d')}")
            print("  This will produce look-ahead bias in the backtest results.")
            print("  Set --predict-start >= " + manifest['train_end'] + " for a clean test.")
            print("!" * 70 + "\n")

    def train_model(self, force_retrain: bool = False) -> None:
        """
        Train the ensemble of models (OLD METHOD - single timeframe).
        For multi-timeframe training, use train_model_multitimeframe() instead.

        Args:
            force_retrain: Force retraining even if models exist
        """
        print("\n" + "=" * 60 + "\nStarting Hybrid Ensemble Training (Single Timeframe)...\n" + "=" * 60)
        print("WARNING: This trains only 1H models and scales predictions.")
        print("For better results, use train_model_multitimeframe() instead.\n")

        if self.train_start or self.train_end:
            print(f"[DATE FILTER] Training data restricted to: "
                  f"{self.train_start.strftime('%Y-%m-%d') if self.train_start else 'beginning'} "
                  f"-> {self.train_end.strftime('%Y-%m-%d') if self.train_end else 'end'}\n")

        # Check if models already exist
        model_type_counts_check = defaultdict(int)
        all_models_exist = True
        for model_type in self.ensemble_model_types:
            model_index = model_type_counts_check[model_type]
            if not os.path.exists(self._get_model_path(model_type, model_index)):
                all_models_exist = False
                break
            model_type_counts_check[model_type] += 1

        if all_models_exist and not force_retrain:
            print("All models already exist. Loading them. Use --force to retrain.")
            self.load_model_assets()
            return

        # Download and prepare data (date-bounded when args supplied)
        df_h1, df_h4, df_d1 = self.download_data(date_from=self.train_start, date_to=self.train_end)
        if df_h1 is None:
            return

        df_features = self.create_features(df_h1, df_h4, df_d1)
        df_selected = self.perform_feature_selection(df_features)

        # Prepare data for different model types
        X_train_seq, y_train_seq, X_val_seq, y_val_seq = self._prepare_sequential_data(df_selected)
        X_train_tab, y_train_tab, X_val_tab, y_val_tab, _ = self._prepare_tabular_data(df_selected)

        # Save scalers
        with open(self.feature_scaler_path, 'wb') as f:
            pickle.dump(self.feature_scaler, f)
        with open(self.target_scaler_path, 'wb') as f:
            pickle.dump(self.target_scaler, f)

        # Try to load best hyperparameters from tuning
        best_hps = None
        try:
            tuner = kt.RandomSearch(
                lambda hp: self._build_dl_model('lstm', (X_train_seq.shape[1], X_train_seq.shape[2]), hp),
                objective='val_loss',
                directory=self.tuner_dir,
                project_name=f'tuner_{self.symbol}'
            )
            best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
            print("Found best hyperparameters from tuning.")
        except Exception:
            print("No tuning data found. Using default hyperparameters for DL models.")

        # Train each model in the ensemble
        model_type_counts = defaultdict(int)
        for model_type in self.ensemble_model_types:
            model_index = model_type_counts[model_type]
            print(f"\n--- Training Model {model_type.upper()} (Instance {model_index}) ---")
            tf.random.set_seed(42 + model_index)

            if model_type in ['lstm', 'gru', 'transformer', 'tcn']:
                # Train deep learning model
                model = self._build_dl_model(model_type, (X_train_seq.shape[1], X_train_seq.shape[2]), hp=best_hps)
                callbacks = [
                    EarlyStopping('val_loss', patience=15, restore_best_weights=True),
                    ReduceLROnPlateau('val_loss', patience=5, factor=0.5)
                ]
                model.fit(
                    X_train_seq, y_train_seq,
                    validation_data=(X_val_seq, y_val_seq),
                    epochs=150,
                    batch_size=64,
                    callbacks=callbacks,
                    verbose=1
                )
                model.save(self._get_model_path(model_type, model_index))

            elif model_type == 'lgbm':
                # Train LightGBM model
                model = lgb.LGBMRegressor(
                    objective='regression_l1',
                    n_estimators=1000,
                    learning_rate=0.05,
                    random_state=42 + model_index,
                    n_jobs=-1,
                    verbose=-1
                )
                model.fit(
                    X_train_tab, y_train_tab,
                    eval_set=[(X_val_tab, y_val_tab)],
                    eval_metric='mae',
                    callbacks=[lgb.early_stopping(100, verbose=False)]
                )
                with open(self._get_model_path(model_type, model_index), 'wb') as f:
                    pickle.dump(model, f)

            model_type_counts[model_type] += 1

        print("\nEnsemble training complete and all assets saved.")
        self.load_model_assets()
        # Record the exact training window for future look-ahead checks
        actual_end = self.train_end or (df_h1.index.max().to_pydatetime() if df_h1 is not None else None)
        self._save_cutoff_manifest(self.train_start, actual_end)

    def train_model_multitimeframe(self, force_retrain: bool = False) -> None:
        """
        Train separate models for each timeframe (1H, 4H, 1D).
        This is the RECOMMENDED method for accurate multi-timeframe predictions.

        Args:
            force_retrain: Force retraining even if models exist
        """
        print("\n" + "=" * 60)
        print("Starting Multi-Timeframe Ensemble Training...")
        print("=" * 60 + "\n")
        print("This will train 3 separate ensembles (1H, 4H, 1D)")
        print(f"Total models to train: {len(self.ensemble_model_types) * 3}")
        print("Estimated time: 4-6 hours\n")

        if self.train_start or self.train_end:
            print(f"[DATE FILTER] Training data restricted to: "
                  f"{self.train_start.strftime('%Y-%m-%d') if self.train_start else 'beginning'} "
                  f"-> {self.train_end.strftime('%Y-%m-%d') if self.train_end else 'end'}\n")

        # Download and prepare data (date-bounded when args supplied)
        df_h1, df_h4, df_d1 = self.download_data(date_from=self.train_start, date_to=self.train_end)
        if df_h1 is None:
            return

        df_features = self.create_features(df_h1, df_h4, df_d1)
        df_selected = self.perform_feature_selection(df_features)

        # Define target columns for each timeframe
        timeframe_targets = {
            '1H': 'fwd_log_return_1h',
            '4H': 'fwd_log_return_4h',
            '1D': 'fwd_log_return_1d'
        }

        # Try to load best hyperparameters from tuning
        best_hps = None
        try:
            tuner = kt.RandomSearch(
                lambda hp: self._build_dl_model('lstm', (60, 25), hp),
                objective='val_loss',
                directory=self.tuner_dir,
                project_name=f'tuner_{self.symbol}'
            )
            best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
            print("Found best hyperparameters from tuning.")
        except Exception:
            print("No tuning data found. Using default hyperparameters.")

        # Train a separate ensemble for each timeframe
        for tf_name, target_col in timeframe_targets.items():
            print(f"\n{'=' * 60}")
            print(f"Training Ensemble for {tf_name} Predictions (Target: {target_col})")
            print(f"{'=' * 60}\n")

            # Temporarily change the target column
            original_target = self.target_column
            self.target_column = target_col

            # Prepare data with this target
            X_train_seq, y_train_seq, X_val_seq, y_val_seq = self._prepare_sequential_data(df_selected)
            X_train_tab, y_train_tab, X_val_tab, y_val_tab, _ = self._prepare_tabular_data(df_selected)

            # Save scalers for this timeframe
            scaler_suffix = f"_{tf_name}"
            with open(self.feature_scaler_path.replace('.pkl', f'{scaler_suffix}.pkl'), 'wb') as f:
                pickle.dump(self.feature_scaler, f)
            with open(self.target_scaler_path.replace('.pkl', f'{scaler_suffix}.pkl'), 'wb') as f:
                pickle.dump(self.target_scaler, f)

            # Train each model type for this timeframe
            model_type_counts = defaultdict(int)
            for model_type in self.ensemble_model_types:
                model_index = model_type_counts[model_type]
                print(f"\n--- Training {model_type.upper()} for {tf_name} (Instance {model_index}) ---")
                tf.random.set_seed(42 + model_index)

                if model_type in ['lstm', 'gru', 'transformer', 'tcn']:
                    model = self._build_dl_model(model_type, (X_train_seq.shape[1], X_train_seq.shape[2]), hp=best_hps)
                    callbacks = [
                        EarlyStopping('val_loss', patience=15, restore_best_weights=True),
                        ReduceLROnPlateau('val_loss', patience=5, factor=0.5)
                    ]
                    model.fit(
                        X_train_seq, y_train_seq,
                        validation_data=(X_val_seq, y_val_seq),
                        epochs=150,
                        batch_size=64,
                        callbacks=callbacks,
                        verbose=1
                    )
                    # Save with timeframe suffix
                    model_path = self._get_model_path(model_type, model_index).replace('.keras', f'_{tf_name}.keras')
                    model.save(model_path)
                    print(f"Saved: {model_path}")

                elif model_type == 'lgbm':
                    model = lgb.LGBMRegressor(
                        objective='regression_l1',
                        n_estimators=1000,
                        learning_rate=0.05,
                        random_state=42 + model_index,
                        n_jobs=-1,
                        verbose=-1
                    )
                    model.fit(
                        X_train_tab, y_train_tab,
                        eval_set=[(X_val_tab, y_val_tab)],
                        eval_metric='mae',
                        callbacks=[lgb.early_stopping(100, verbose=False)]
                    )
                    # Save with timeframe suffix
                    model_path = self._get_model_path(model_type, model_index).replace('.pkl', f'_{tf_name}.pkl')
                    with open(model_path, 'wb') as f:
                        pickle.dump(model, f)
                    print(f"Saved: {model_path}")

                model_type_counts[model_type] += 1

            # Restore original target
            self.target_column = original_target

        # Copy 1H scalers and models to base names for backward compatibility
        print("\nSaving base scalers and models for backward compatibility...")
        try:
            import shutil

            # Copy scalers
            h1_feature_scaler = self.feature_scaler_path.replace('.pkl', '_1H.pkl')
            h1_target_scaler = self.target_scaler_path.replace('.pkl', '_1H.pkl')

            if os.path.exists(h1_feature_scaler):
                shutil.copy(h1_feature_scaler, self.feature_scaler_path)
                print(f"[OK] Copied {os.path.basename(h1_feature_scaler)} -> {os.path.basename(self.feature_scaler_path)}")

            if os.path.exists(h1_target_scaler):
                shutil.copy(h1_target_scaler, self.target_scaler_path)
                print(f"[OK] Copied {os.path.basename(h1_target_scaler)} -> {os.path.basename(self.target_scaler_path)}")

            # Copy model files
            print("\nCopying 1H models to base names...")
            model_type_counts = defaultdict(int)
            for model_type in self.ensemble_model_types:
                model_index = model_type_counts[model_type]
                base_model_path = self._get_model_path(model_type, model_index)

                # Construct 1H model path
                ext = '.keras' if model_type in ['lstm', 'gru', 'transformer', 'tcn'] else '.pkl'
                h1_model_path = base_model_path.replace(ext, f'_1H{ext}')

                if os.path.exists(h1_model_path):
                    shutil.copy(h1_model_path, base_model_path)
                    print(f"[OK] Copied {os.path.basename(h1_model_path)} -> {os.path.basename(base_model_path)}")

                model_type_counts[model_type] += 1

        except Exception as e:
            print(f"Warning: Could not copy all base files: {e}")
            print("This may cause issues with backtest mode, but multi-TF predictions will work fine.")

        print("\n" + "=" * 60)
        print("Multi-timeframe ensemble training complete!")
        print("=" * 60)
        print(f"\nTrained {len(self.ensemble_model_types) * 3} models total")
        print("Use predict-multitf command to make predictions with these models")
        # Record the exact training window for future look-ahead checks
        actual_end = self.train_end or (df_h1.index.max().to_pydatetime() if df_h1 is not None else None)
        self._save_cutoff_manifest(self.train_start, actual_end)

    def load_model_assets(self) -> bool:
        """
        Load all trained models and scalers (single timeframe method).

        Returns:
            True if successful, False otherwise
        """
        print("Loading all model assets for the ensemble...")

        # Auto-detect models if not specified
        if not self.ensemble_model_types:
            self.ensemble_model_types = self._detect_trained_models()
            if not self.ensemble_model_types:
                print("Error: No trained models found. Please run the 'train' command first.")
                return False
            self.num_ensemble_models = len(self.ensemble_model_types)
            equal_weight = 1.0 / self.num_ensemble_models
            self.ensemble_weights = {
                tf_key: [equal_weight] * self.num_ensemble_models
                for tf_key in self.kalman_config.keys()
            }
            print(f"Detected trained models: {self.ensemble_model_types}")

        try:
            # Load feature list and scalers
            with open(self.selected_features_path, 'r') as f:
                self.feature_cols = json.load(f)

            # Try to load base scaler first, fallback to multi-TF scalers if needed
            feature_scaler_loaded = False
            target_scaler_loaded = False

            # Try base scaler first
            if os.path.exists(self.feature_scaler_path):
                with open(self.feature_scaler_path, 'rb') as f:
                    self.feature_scaler = pickle.load(f)
                feature_scaler_loaded = True
            else:
                # Check for multi-timeframe scalers (try 1H first as base timeframe)
                for tf_suffix in ['_1H', '_4H', '_1D']:
                    mtf_path = self.feature_scaler_path.replace('.pkl', f'{tf_suffix}.pkl')
                    if os.path.exists(mtf_path):
                        print(f"Note: Using multi-timeframe scaler: {os.path.basename(mtf_path)}")
                        with open(mtf_path, 'rb') as f:
                            self.feature_scaler = pickle.load(f)
                        feature_scaler_loaded = True
                        break

            if not feature_scaler_loaded:
                raise FileNotFoundError(f"Feature scaler not found: {self.feature_scaler_path}")

            # Try base target scaler first
            if os.path.exists(self.target_scaler_path):
                with open(self.target_scaler_path, 'rb') as f:
                    self.target_scaler = pickle.load(f)
                target_scaler_loaded = True
            else:
                # Check for multi-timeframe target scalers
                for tf_suffix in ['_1H', '_4H', '_1D']:
                    mtf_path = self.target_scaler_path.replace('.pkl', f'{tf_suffix}.pkl')
                    if os.path.exists(mtf_path):
                        with open(mtf_path, 'rb') as f:
                            self.target_scaler = pickle.load(f)
                        target_scaler_loaded = True
                        break

            if not target_scaler_loaded:
                raise FileNotFoundError(f"Target scaler not found: {self.target_scaler_path}")

            # Load each model
            self.models = {}
            model_type_counts = defaultdict(int)

            for model_type in self.ensemble_model_types:
                model_index = model_type_counts[model_type]
                model_path = self._get_model_path(model_type, model_index)
                model_name = f"{model_type}_{model_index}"

                # Auto-detect multi-timeframe model files if base doesn't exist
                actual_model_path = model_path
                if not os.path.exists(model_path):
                    # Check for multi-timeframe model files
                    for tf_suffix in ['_1H', '_4H', '_1D']:
                        if model_type in ['lstm', 'gru', 'transformer', 'tcn']:
                            # Try .keras first (newer), then .h5 (older)
                            for ext in ['.keras', '.h5']:
                                mtf_path = model_path.replace('.keras', f'{tf_suffix}{ext}').replace('.h5', f'{tf_suffix}{ext}')
                                if os.path.exists(mtf_path):
                                    print(f"Note: Using multi-timeframe model: {os.path.basename(mtf_path)}")
                                    actual_model_path = mtf_path
                                    break
                        else:  # lgbm
                            mtf_path = model_path.replace('.pkl', f'{tf_suffix}.pkl')
                            if os.path.exists(mtf_path):
                                print(f"Note: Using multi-timeframe model: {os.path.basename(mtf_path)}")
                                actual_model_path = mtf_path
                                break

                        if actual_model_path != model_path:
                            break

                if not os.path.exists(actual_model_path):
                    print(f"ERROR: Model file not found: {model_path}")
                    print(f"       Also checked for multi-TF versions with suffixes _1H, _4H, _1D")
                    return False

                if model_type in ['lstm', 'gru', 'transformer', 'tcn']:
                    print(f"  Loading {model_name} from {os.path.basename(actual_model_path)}...")
                    self.models[model_name] = load_model(
                        actual_model_path,
                        custom_objects={
                            'TransformerBlock': TransformerBlock,
                            'AttentionLayer': AttentionLayer
                        },
                        compile=False
                    )
                    print(f"  Loaded {model_name}")
                elif model_type == 'lgbm':
                    with open(actual_model_path, 'rb') as f:
                        self.models[model_name] = pickle.load(f)

                model_type_counts[model_type] += 1

            print(f"Successfully loaded {len(self.models)} models: {list(self.models.keys())}")
            return True

        except FileNotFoundError as e:
            print(f"Error: Model assets not found. Please train the model first. Missing: {e.filename}")
            return False
        except TypeError as e:
            # This usually indicates version mismatch between training and loading
            error_msg = str(e)
            if "Could not deserialize" in error_msg or "keras.src" in error_msg:
                print("\n" + "=" * 80)
                print("ERROR: MODEL VERSION MISMATCH DETECTED!")
                print("=" * 80)
                print("\nYour saved models are incompatible with your current Keras/TensorFlow version.")
                print("\nThis happens when:")
                print("  - Models were trained with Keras 3.x but you have Keras 2.x")
                print("  - Models were trained with Keras 2.x but you have Keras 3.x")
                print("  - TensorFlow version changed after training")
                print("\nSOLUTION:")
                print("  1. Retrain your models with your current environment")
                print(f"     Command: python unified_predictor_v8.py train-multitf --symbol {self.symbol} --force")
                print("\n  2. Or run from GUI: Select 'Train Models' and check 'Force Retrain'")
                print("\nYour current versions:")
                print(f"  TensorFlow: {tf.__version__}")
                print(f"  Keras: {keras.__version__}")
                print("\n" + "=" * 80)
            else:
                print(f"Error loading model assets: {e}")
                import traceback
                traceback.print_exc()
            return False
        except Exception as e:
            print(f"Error loading model assets: {e}")
            import traceback
            traceback.print_exc()
            return False

    def load_model_assets_multitimeframe(self) -> bool:
        """
        Load models for all timeframes (multi-timeframe method).

        Returns:
            True if successful, False otherwise
        """
        print("Loading multi-timeframe model assets...")

        try:
            if not self.ensemble_model_types:
                self.ensemble_model_types = self._detect_trained_models()
                if not self.ensemble_model_types:
                    print("Error: No trained models found.")
                    return False

            # Load feature list
            try:
                with open(self.selected_features_path, 'r') as f:
                    self.feature_cols = json.load(f)
            except FileNotFoundError:
                print("ERROR: Feature list not found. Please train models first.")
                return False

            timeframe_list = ['1H', '4H', '1D']
            self.models_by_timeframe = {}
            self.scalers_by_timeframe = {}

            for tf_name in timeframe_list:
                print(f"\nLoading models for {tf_name}...")

                # Load scalers for this timeframe
                try:
                    scaler_suffix = f"_{tf_name}"
                    with open(self.feature_scaler_path.replace('.pkl', f'{scaler_suffix}.pkl'), 'rb') as f:
                        feature_scaler = pickle.load(f)
                    with open(self.target_scaler_path.replace('.pkl', f'{scaler_suffix}.pkl'), 'rb') as f:
                        target_scaler = pickle.load(f)
                    self.scalers_by_timeframe[tf_name] = (feature_scaler, target_scaler)
                    print(f"  Loaded scalers for {tf_name}")
                except FileNotFoundError:
                    print(f"WARNING: Scalers not found for {tf_name}")
                    return False

                # Load models for this timeframe
                models = {}
                model_type_counts = defaultdict(int)

                for model_type in self.ensemble_model_types:
                    model_index = model_type_counts[model_type]
                    model_name = f"{model_type}_{model_index}"

                    if model_type in ['lstm', 'gru', 'transformer', 'tcn']:
                        model_path = self._get_model_path(model_type, model_index).replace('.keras', f'_{tf_name}.keras')
                        if os.path.exists(model_path):
                            print(f"  Loading {model_name} from {os.path.basename(model_path)}...")
                            # compile=False skips TF graph recompilation at load time.
                            # Without it, Keras traces the full computation graph for every
                            # custom-layer model, which can take minutes or hang entirely on
                            # CPU.  We only need inference here — no gradients required.
                            models[model_name] = load_model(
                                model_path,
                                custom_objects={
                                    'TransformerBlock': TransformerBlock,
                                    'AttentionLayer': AttentionLayer
                                },
                                compile=False
                            )
                            print(f"  Loaded {model_name}")
                        else:
                            print(f"ERROR: Model not found: {model_path}")
                            return False

                    elif model_type == 'lgbm':
                        model_path = self._get_model_path(model_type, model_index).replace('.pkl', f'_{tf_name}.pkl')
                        if os.path.exists(model_path):
                            with open(model_path, 'rb') as f:
                                models[model_name] = pickle.load(f)
                            print(f"  Loaded {model_name}")
                        else:
                            print(f"ERROR: Model not found: {model_path}")
                            return False

                    model_type_counts[model_type] += 1

                self.models_by_timeframe[tf_name] = models
                print(f"  Total models for {tf_name}: {len(models)}")

            print(f"\nSuccessfully loaded models for all {len(self.models_by_timeframe)} timeframes")
            return len(self.models_by_timeframe) > 0

        except TypeError as e:
            # This usually indicates version mismatch between training and loading
            error_msg = str(e)
            if "Could not deserialize" in error_msg or "keras.src" in error_msg:
                print("\n" + "=" * 80)
                print("ERROR: MODEL VERSION MISMATCH DETECTED!")
                print("=" * 80)
                print("\nYour saved models are incompatible with your current Keras/TensorFlow version.")
                print("\nThis happens when:")
                print("  - Models were trained with Keras 3.x but you have Keras 2.x")
                print("  - Models were trained with Keras 2.x but you have Keras 3.x")
                print("  - TensorFlow version changed after training")
                print("\nSOLUTION:")
                print("  1. Retrain your models with your current environment")
                print(f"     Command: python unified_predictor_v8.py train-multitf --symbol {self.symbol} --force")
                print("\n  2. Or run from GUI: Select 'Train Models' and check 'Force Retrain'")
                print("\nYour current versions:")
                print(f"  TensorFlow: {tf.__version__}")
                print(f"  Keras: {keras.__version__}")
                print("\n" + "=" * 80)
            else:
                print(f"Error loading model assets: {e}")
                import traceback
                traceback.print_exc()
            return False
        except Exception as e:
            print(f"Error loading multi-timeframe model assets: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _detect_trained_models(self) -> Optional[List[str]]:
        """
        Detect trained models from saved files.

        Returns:
            List of detected model types or None if none found
        """
        # Find all model files
        all_model_files = glob.glob(os.path.join(self.base_path, f"model_{self.symbol}_*.h5"))
        all_model_files += glob.glob(os.path.join(self.base_path, f"model_{self.symbol}_*.pkl"))
        all_model_files += glob.glob(os.path.join(self.base_path, f"model_{self.symbol}_*.keras"))

        found_models = set()

        # Parse model files
        for f in all_model_files:
            parts = os.path.basename(f).split('_')
            if len(parts) >= 4:
                model_type = parts[2]
                try:
                    # Handle both regular and multitimeframe models
                    index_part = parts[3].split('.')[0]

                    # Also detect multi-timeframe models
                    if index_part in ['1H', '4H', '1D']:
                        # This is a multi-TF model
                        if len(parts) >= 5:
                            # Format: model_SYMBOL_TYPE_INDEX_TIMEFRAME.ext
                            try:
                                model_index = int(parts[3])
                                if model_type in ['lstm', 'gru', 'transformer', 'tcn', 'lgbm']:
                                    found_models.add((model_type, model_index))
                            except ValueError:
                                pass
                        continue

                    model_index = int(index_part)
                    if model_type in ['lstm', 'gru', 'transformer', 'tcn', 'lgbm']:
                        found_models.add((model_type, model_index))
                except (ValueError, IndexError):
                    continue

        # Sort by index then type
        found_models = sorted(list(found_models), key=lambda x: (x[1], x[0]))
        detected = [model_type for model_type, model_index in found_models]

        print(f"   Found model files: {found_models}")
        print(f"   Detected models: {detected}")
        return detected if detected else None

    def _get_model_path(self, model_type: str, index: int) -> str:
        """Get the file path for a model."""
        ext = 'keras' if model_type in ['lstm', 'gru', 'transformer', 'tcn'] else 'pkl'
        return os.path.join(self.base_path, f"model_{self.symbol}_{model_type}_{index}.{ext}")

    def _apply_regime_bias(
        self,
        base_weights: List[float],
        model_names: List[str],
        regime: str,
    ) -> List[float]:
        """
        Multiply base ensemble weights by regime-specific per-model scalars,
        then renormalise so the result still sums to 1.0.

        The intuition behind the bias table:
            trending  — TCN and Transformer capture long-range directional
                        structure well; LGBM treats each bar as i.i.d. and
                        misses momentum, so it gets a mild penalty.
            ranging   — LGBM excels at mean-reversion because decision trees
                        have no assumption of temporal order.  TCN/Transformer
                        can overfit to spurious trends, so they get down-weighted.
            volatile  — All deep models are hurt by distributional shift; LGBM
                        is slightly more robust.  Dampen everything a little and
                        let LGBM lead.
            unknown   — No bias; keep whatever base_weights are.

        The scalar values are intentionally conservative (range 0.8–1.3) so
        the regime can nudge but not override learned accuracy weights.  Tune
        on your own backtest data once you have enough evaluated predictions.

        Args:
            base_weights: Current per-TF weight vector (already accuracy-weighted).
            model_names:  Names of each model in the same order as base_weights
                          (e.g. ["lstm_0", "transformer_1", "lgbm_2"]).
            regime:       Output of _detect_regime().

        Returns:
            Renormalised weight list of the same length.
        """
        REGIME_BIAS: Dict[str, Dict[str, float]] = {
            'trending': {
                'lstm': 1.1, 'gru': 1.0, 'transformer': 1.3, 'tcn': 1.3, 'lgbm': 0.8
            },
            'ranging': {
                'lstm': 1.0, 'gru': 1.0, 'transformer': 0.8, 'tcn': 0.8, 'lgbm': 1.3
            },
            'volatile': {
                'lstm': 0.9, 'gru': 0.9, 'transformer': 0.8, 'tcn': 0.8, 'lgbm': 1.2
            },
        }

        bias_map = REGIME_BIAS.get(regime, {})
        if not bias_map:
            return base_weights  # 'unknown' or unrecognised regime — no change

        biased = []
        for w, name in zip(base_weights, model_names):
            # Extract model type from names like "lstm_0", "transformer_1", "lgbm_2"
            model_type = name.split('_')[0] if '_' in name else name
            scalar = bias_map.get(model_type, 1.0)
            biased.append(w * scalar)

        total = sum(biased)
        if total <= 0:
            return base_weights  # safety: don't zero out all weights
        return [b / total for b in biased]

    def run_prediction_cycle(self):
        """Updated with Macro integration."""
        print(f"\n--- Single-Timeframe Prediction Cycle: {self.symbol} ---")

        # Download main data first
        df_h1, df_h4, df_d1 = self.download_data(500)
        if df_h1 is None:
            return

        # Download macro data safely
        df_dxy, df_spx = self._download_macro_data(300)
        context = self.get_market_context(df_h1, df_dxy, df_spx)

        print("\n" + "=" * 60)
        print(f"Starting Prediction Cycle for {self.symbol} at {datetime.now()}")
        print("=" * 60 + "\n")
        print("WARNING: Using single-timeframe models with scaling.")
        print("For better predictions, use run_prediction_cycle_multitimeframe()\n")

        # Load models if not already loaded
        if not self.models:
            if not self.load_model_assets():
                return

        # Evaluate past predictions and update weights
        self._evaluate_past_predictions()
        self.update_ensemble_weights()

        # Download fresh data.
        # H1 bar count must cover the longest warmup period (log_return_1w = 168 bars)
        # plus the sequence lookback (60) plus a safety buffer -> 500 is sufficient.
        df_h1 = pd.DataFrame(mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_H1, 0, 500))
        df_h4 = pd.DataFrame(mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_H4, 0, 150))
        df_d1 = pd.DataFrame(mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_D1, 0,  75))

        # Validate data — minimums are per-timeframe to match the download counts above
        _min_bars = {"H1": 250, "H4": 60, "D1": 55}
        for df, name in [(df_h1, "H1"), (df_h4, "H4"), (df_d1, "D1")]:
            required = _min_bars[name]
            if df.empty or len(df) < required:
                print(f"ERROR: Insufficient {name} data for prediction "
                      f"(got {len(df) if not df.empty else 0}, need {required})")
                return
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            if 'tick_volume' in df.columns:
                df.rename(columns={'tick_volume': 'volume'}, inplace=True)

        # Create features
        df = self.create_features(df_h1, df_h4, df_d1)
        current_price = df['close'].iloc[-1]

        # ── Feature compatibility check ────────────────────────────────────────
        if self.feature_cols:
            missing_cols = [c for c in self.feature_cols if c not in df.columns]
            if missing_cols:
                print("\n" + "!" * 70)
                print("  FEATURE MISMATCH — saved feature list does not match current DataFrame.")
                print(f"  Missing columns ({len(missing_cols)}): {missing_cols}")
                print("  This happens when create_features() is updated after models were trained.")
                print("  ACTION REQUIRED: retrain all models with:")
                print("      python unified_predictor_v8.py train-multitf --force")
                print("!" * 70 + "\n")
                return

        # Prepare sequential input
        last_sequence_raw = df.iloc[-self.lookback_periods:][self.feature_cols].values
        last_sequence_scaled = self.feature_scaler.transform(last_sequence_raw)
        X_pred_seq = last_sequence_scaled.reshape(1, self.lookback_periods, len(self.feature_cols))

        # Convert to TensorFlow tensor to avoid retracing warnings
        X_pred_seq = tf.convert_to_tensor(X_pred_seq, dtype=tf.float32)

        # Prepare tabular input for LightGBM
        df_tabular = df.copy()
        final_tab_cols = []
        for model_name, model in self.models.items():
            if 'lgbm' in model_name:
                final_tab_cols = model.feature_name_
                break

        X_pred_tab = None
        if final_tab_cols:
            for col in self.feature_cols:
                for lag in [1, 3, 5, 10]:
                    new_col = f'{col}_lag_{lag}'
                    if new_col in final_tab_cols:
                        df_tabular[new_col] = df_tabular[col].shift(lag)

            # Check for NaN values in tabular features
            last_row = df_tabular.iloc[-1][final_tab_cols]
            if last_row.isna().any():
                print("WARNING: NaN values detected in tabular features, filling with forward fill")
                df_tabular.ffill(inplace=True)

            X_pred_tab = df_tabular.iloc[-1][final_tab_cols].values.reshape(1, -1)

        # Make predictions
        predictions = {}
        timeframes = {"1H": 1, "4H": 4, "1D": 24}
        ensemble_predictions_map = {}

        print("\nMaking predictions with hybrid ensemble...")
        for tf_name, steps in timeframes.items():
            ensemble_preds = []
            raw_log_returns = []

            # Get predictions from each model
            for model_name, model in self.models.items():
                pred_log_return = 0.0

                try:
                    if 'lgbm' in model_name and X_pred_tab is not None:
                        pred_log_return = model.predict(X_pred_tab)[0]
                    elif 'lgbm' not in model_name:
                        # Use direct call to avoid retracing
                        pred_log_return_scaled = model(X_pred_seq, training=False).numpy()[0][0]
                        pred_log_return = self.target_scaler.inverse_transform([[pred_log_return_scaled]])[0][0]
                    
                    # SAFEGUARD: Clamp extreme log returns (before scaling)
                    # Base max return is 0.5% for 1H
                    max_base_log_return = 0.005
                    if abs(pred_log_return) > max_base_log_return:
                        original_lr = pred_log_return
                        pred_log_return = np.clip(pred_log_return, -max_base_log_return, max_base_log_return)
                        print(f"  WARNING: {model_name} log return clamped from {original_lr:.6f} to {pred_log_return:.6f}")
                        
                except Exception as e:
                    print(f"WARNING: Error predicting with {model_name}: {e}")
                    continue

                raw_log_returns.append(pred_log_return)

                # Scale log return by time horizon (square root scaling)
                steps_adjusted = np.sqrt(steps) if steps > 1 else steps
                predicted_price = current_price * np.exp(pred_log_return * steps_adjusted)

                # Validate predicted price
                if np.isnan(predicted_price) or np.isinf(predicted_price):
                    print(f"WARNING: Invalid prediction from {model_name}, using current price")
                    predicted_price = current_price

                ensemble_preds.append(predicted_price)

            # Check if we have valid predictions
            if not ensemble_preds:
                print(f"ERROR: No valid predictions for {tf_name}, skipping")
                continue

            print(f"\n{tf_name} Debug:")
            print(f"  Raw log returns: {[f'{lr:.6f}' for lr in raw_log_returns]}")
            print(f"  Steps: {steps} -> Adjusted: {steps_adjusted:.2f}")
            print(f"  Predicted prices: {[f'{p:.5f}' for p in ensemble_preds]}")

            ensemble_predictions_map[tf_name] = ensemble_preds
            tf_weights = self.ensemble_weights.get(tf_name, [])
            if tf_weights and len(tf_weights) >= len(ensemble_preds):
                # Fix 4: apply regime bias before combining
                regime = context.get('regime', 'unknown')
                model_names = list(self.models.keys())
                tf_weights = self._apply_regime_bias(
                    tf_weights[:len(ensemble_preds)], model_names[:len(ensemble_preds)], regime
                )
                raw_prediction = np.average(ensemble_preds, weights=tf_weights)
            else:
                raw_prediction = np.mean(ensemble_preds)

            print(f"  Raw ensemble average: {raw_prediction:.5f}")

            # Apply smoothing to log returns
            raw_log_return = np.log(raw_prediction / current_price)

            if self.use_kalman:
                # Use Kalman filtering
                print(f"  Kalman state before: x={self.kalman_filters[tf_name].x:.6f}, p={self.kalman_filters[tf_name].p:.6f}")
                smoothed_log_return = self.kalman_filters[tf_name].update(raw_log_return)
                print(f"  Kalman smoothed log return: {smoothed_log_return:.6f} (raw: {raw_log_return:.6f})")
                print(f"  Kalman state after: x={self.kalman_filters[tf_name].x:.6f}, p={self.kalman_filters[tf_name].p:.6f}")
                smoothed_prediction = current_price * np.exp(smoothed_log_return)
            else:
                # Use EMA smoothing
                if self.previous_predictions[tf_name] is not None:
                    prev_log_return = np.log(self.previous_predictions[tf_name] / current_price)
                    smoothed_log_return = self.ema_alpha * raw_log_return + (1 - self.ema_alpha) * prev_log_return
                    smoothed_prediction = current_price * np.exp(smoothed_log_return)
                    print(f"  EMA smoothed log return: {smoothed_log_return:.6f} (raw: {raw_log_return:.6f})")
                else:
                    smoothed_prediction = raw_prediction
                    print(f"  Using raw prediction (first prediction)")

            self.previous_predictions[tf_name] = smoothed_prediction

            max_change_pct = {'1H': 0.5, '4H': 1.0, '1D': 2.0}
            max_change = current_price * (max_change_pct.get(tf_name, 1.0) / 100.0)

            if abs(smoothed_prediction - current_price) > max_change:
                original_pred = smoothed_prediction
                if smoothed_prediction > current_price:
                    smoothed_prediction = current_price + max_change
                else:
                    smoothed_prediction = current_price - max_change
                print(f"  Capped from {original_pred:.5f} to {smoothed_prediction:.5f}")

            # change percentage for the EA
            change_pct = ((smoothed_prediction - current_price) / current_price) * 100.0

            predictions[tf_name] = {
                'prediction': round(smoothed_prediction, 5),
                'change_pct': round(change_pct, 3),
                'ensemble_std': round(np.std(ensemble_preds), 5),
            }

        # Log predictions for future evaluation
        self._log_prediction_for_evaluation(timeframes, ensemble_predictions_map, current_price)

        # Uncertainty veto (same logic as multi-TF path — Fix 3)
        UNCERTAINTY_THRESHOLD = 0.003
        uncertainty_veto = any(
            data['ensemble_std'] / current_price > UNCERTAINTY_THRESHOLD
            for data in predictions.values()
        )
        if uncertainty_veto:
            print(f"   VETO (uncertainty): ensemble std exceeds {UNCERTAINTY_THRESHOLD*100:.1f}% threshold")

        # Save predictions and status
        status = {
            'last_update': datetime.now().isoformat(),
            'status': 'online',
            'symbol': self.symbol,
            'current_price': round(current_price, 5),
            'ensemble_weights': {
                tf_key: [round(w, 3) for w in weights]
                for tf_key, weights in self.ensemble_weights.items()
            },
            'price': df_h1['close'].iloc[-1],
            'market_context': context,
            'regime': context.get('regime', 'unknown'),
            'uncertainty_veto': uncertainty_veto,
            'trade_allowed': not context['veto_active'] and not uncertainty_veto
        }

        self.save_to_file(self.predictions_file, predictions)
        self.save_to_file(self.status_file, status)

        # Display results
        print("\n--- Prediction Cycle Complete! ---")
        print(f"Current Price: {current_price:.5f}")
        for timeframe, data in predictions.items():
            direction = "UP" if data['prediction'] > current_price else "DOWN"
            change_pct = ((data['prediction'] - current_price) / current_price) * 100
            print(f"   {direction} {timeframe}: {data['prediction']:.5f} ({change_pct:+.3f}%) (Uncertainty: +/-{data['ensemble_std']:.5f})")

    def run_prediction_cycle_multitimeframe(self):
        """Updated with Macro integration."""
        print(f"\n--- Multi-Timeframe Cycle: {self.symbol} ---")

        # Download main data first
        df_h1, df_h4, df_d1 = self.download_data(500)
        if df_h1 is None:
            return

        # Download macro data safely
        df_dxy, df_spx = self._download_macro_data(300)
        context = self.get_market_context(df_h1, df_dxy, df_spx)

        print("\n" + "=" * 60)
        print(f"Starting Multi-Timeframe Prediction Cycle for {self.symbol}")
        print("=" * 60 + "\n")

        if not hasattr(self, 'models_by_timeframe') or not self.models_by_timeframe:
            if not self.load_model_assets_multitimeframe():
                return

        # Evaluate past predictions logged in the previous cycle and use the
        # results to update per-timeframe ensemble weights.  These calls were
        # only present in run_prediction_cycle (single-TF path) — their absence
        # here meant multi-TF weights were permanently frozen at the equal
        # initialisation value of 0.200 regardless of model accuracy.
        self._evaluate_past_predictions()
        self.update_ensemble_weights()

        # Download fresh data.
        # H1 bar count must cover the longest warmup period (log_return_1w = 168 bars)
        # plus the sequence lookback (60) plus a safety buffer -> 500 is sufficient.
        # H4 and D1 only need enough for SMA-50 warmup.
        df_h1 = pd.DataFrame(mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_H1, 0, 500))
        df_h4 = pd.DataFrame(mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_H4, 0, 150))
        df_d1 = pd.DataFrame(mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_D1, 0,  75))

        # Validate data — minimums are per-timeframe to match the download counts above
        _min_bars = {"H1": 250, "H4": 60, "D1": 55}
        for df, name in [(df_h1, "H1"), (df_h4, "H4"), (df_d1, "D1")]:
            required = _min_bars[name]
            if df.empty or len(df) < required:
                print(f"ERROR: Insufficient {name} data "
                      f"(got {len(df) if not df.empty else 0}, need {required})")
                return
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            if 'tick_volume' in df.columns:
                df.rename(columns={'tick_volume': 'volume'}, inplace=True)

        # Create features
        df = self.create_features(df_h1, df_h4, df_d1)
        current_price = df['close'].iloc[-1]

        # ── Feature compatibility check ────────────────────────────────────────
        # self.feature_cols was saved during training. If create_features() has
        # been updated since then the column names may no longer match, producing
        # a cryptic pandas KeyError. Catch it here with an actionable message.
        if self.feature_cols:
            missing_cols = [c for c in self.feature_cols if c not in df.columns]
            if missing_cols:
                print("\n" + "!" * 70)
                print("  FEATURE MISMATCH — saved feature list does not match current DataFrame.")
                print(f"  Missing columns ({len(missing_cols)}): {missing_cols}")
                print("  This happens when create_features() is updated after models were trained.")
                print("  ACTION REQUIRED: retrain all models with:")
                print("      python unified_predictor_v8.py train-multitf --force")
                print("!" * 70 + "\n")
                return

        predictions = {}
        # Accumulate raw per-model price predictions per timeframe so they can
        # be logged for future evaluation (feeds update_ensemble_weights).
        # Previously this map was never built in the multi-TF path, so
        # _log_prediction_for_evaluation could never be called and weights
        # were permanently stuck at the equal-weight initialisation of 0.200.
        ensemble_predictions_map: Dict[str, List[float]] = {}

        print("\nMaking predictions with timeframe-specific models...")
        for tf_name, models in self.models_by_timeframe.items():
            if not models:
                print(f"WARNING: No models for {tf_name}, skipping")
                continue

            # Get scalers for this timeframe
            feature_scaler, target_scaler = self.scalers_by_timeframe[tf_name]

            # Prepare input data
            last_sequence_raw = df.iloc[-self.lookback_periods:][self.feature_cols].values
            last_sequence_scaled = feature_scaler.transform(last_sequence_raw)
            X_pred_seq = last_sequence_scaled.reshape(1, self.lookback_periods, len(self.feature_cols))

            # Convert to TensorFlow tensor to avoid retracing warnings
            X_pred_seq = tf.convert_to_tensor(X_pred_seq, dtype=tf.float32)

            # Get predictions from each model
            ensemble_preds = []
            for model_name, model in models.items():
                try:
                    if 'lgbm' in model_name:
                        # Prepare tabular data for LightGBM
                        df_tabular = df.copy()
                        for col in self.feature_cols:
                            for lag in [1, 3, 5, 10]:
                                new_col = f'{col}_lag_{lag}'
                                df_tabular[new_col] = df_tabular[col].shift(lag)
                        df_tabular.ffill(inplace=True)

                        X_pred_tab = df_tabular.iloc[-1][model.feature_name_].values.reshape(1, -1)
                        pred_log_return = model.predict(X_pred_tab)[0]
                    else:
                        # Deep learning model - use direct call to avoid retracing
                        pred_log_return_scaled = model(X_pred_seq, training=False).numpy()[0][0]
                        pred_log_return = target_scaler.inverse_transform([[pred_log_return_scaled]])[0][0]

                    # SAFEGUARD: Clamp extreme log returns before converting to price
                    # Max expected returns: 1H=0.5%, 4H=1%, 1D=2%
                    max_log_return = {'1H': 0.005, '4H': 0.01, '1D': 0.02}.get(tf_name, 0.01)
                    
                    if abs(pred_log_return) > max_log_return:
                        original_lr = pred_log_return
                        pred_log_return = np.clip(pred_log_return, -max_log_return, max_log_return)
                        print(f"  WARNING: {model_name} log return clamped from {original_lr:.6f} to {pred_log_return:.6f}")

                    # Convert log return to price (NO SCALING!)
                    predicted_price = current_price * np.exp(pred_log_return)

                    if not np.isnan(predicted_price) and not np.isinf(predicted_price):
                        ensemble_preds.append(predicted_price)

                except Exception as e:
                    print(f"WARNING: Error with {model_name} for {tf_name}: {e}")
                    continue

            if not ensemble_preds:
                print(f"ERROR: No valid predictions for {tf_name}")
                continue

            # Record raw per-model predictions for this timeframe so they can
            # be logged at the end of the cycle and evaluated in the next cycle.
            ensemble_predictions_map[tf_name] = ensemble_preds

            # Weighted ensemble average using per-timeframe weights (Fix 1 & 2).
            # Falls back to a plain mean only if the weight vector length mismatches
            # the number of predictions that actually came back (e.g. a model failed).
            tf_weights = self.ensemble_weights.get(tf_name, [])
            if tf_weights and len(tf_weights) == len(ensemble_preds):
                # Fix 4: bias the accuracy-learned weights toward regime-appropriate models
                regime = context.get('regime', 'unknown')
                model_names = list(models.keys())  # same order as ensemble_preds
                tf_weights = self._apply_regime_bias(tf_weights, model_names, regime)
                raw_prediction = np.average(ensemble_preds, weights=tf_weights)
                weight_info = (f"weights={[f'{w:.3f}' for w in tf_weights]} "
                               f"[regime={regime}]")
            else:
                raw_prediction = np.mean(ensemble_preds)
                weight_info = "weights=equal (fallback — count mismatch)"

            print(f"\n{tf_name}:")
            print(f"  Ensemble predictions: {[f'{p:.5f}' for p in ensemble_preds]}")
            print(f"  Weighted average: {raw_prediction:.5f} ({weight_info})")

            # Apply smoothing
            raw_log_return = np.log(raw_prediction / current_price)

            if self.use_kalman:
                smoothed_log_return = self.kalman_filters[tf_name].update(raw_log_return)
                smoothed_prediction = current_price * np.exp(smoothed_log_return)
                print(f"  Kalman smoothed: {smoothed_prediction:.5f}")
            else:
                if self.previous_predictions[tf_name] is not None:
                    prev_log_return = np.log(self.previous_predictions[tf_name] / current_price)
                    smoothed_log_return = self.ema_alpha * raw_log_return + (1 - self.ema_alpha) * prev_log_return
                    smoothed_prediction = current_price * np.exp(smoothed_log_return)
                    print(f"  EMA smoothed: {smoothed_prediction:.5f}")
                else:
                    smoothed_prediction = raw_prediction
                self.previous_predictions[tf_name] = smoothed_prediction

            # Sanity check
            max_change_pct = {'1H': 0.5, '4H': 1.0, '1D': 2.0}
            max_change = current_price * (max_change_pct.get(tf_name, 1.0) / 100.0)

            if abs(smoothed_prediction - current_price) > max_change:
                original_pred = smoothed_prediction
                if smoothed_prediction > current_price:
                    smoothed_prediction = current_price + max_change
                else:
                    smoothed_prediction = current_price - max_change
                print(f"  Capped from {original_pred:.5f} to {smoothed_prediction:.5f}")

            # Calculate change percentage
            change_pct = ((smoothed_prediction - current_price) / current_price) * 100.0

            predictions[tf_name] = {
                'prediction': round(smoothed_prediction, 5),
                'change_pct': round(change_pct, 3),
                'ensemble_std': round(np.std(ensemble_preds), 5)
            }

        # ── Fix 3: Uncertainty veto ────────────────────────────────────────────
        # When the ensemble models strongly disagree on a given bar their
        # individual predictions scatter widely.  A high relative std is a
        # signal that the bar is ambiguous; we block trading rather than
        # pick a direction arbitrarily.
        UNCERTAINTY_THRESHOLD = 0.003  # 0.3 % of current price
        uncertainty_veto = False
        for tf_key, tf_data in predictions.items():
            rel_std = tf_data['ensemble_std'] / current_price
            if rel_std > UNCERTAINTY_THRESHOLD:
                uncertainty_veto = True
                print(f"   VETO (uncertainty): {tf_key} relative std={rel_std*100:.3f}% "
                      f"exceeds threshold {UNCERTAINTY_THRESHOLD*100:.1f}%")

        # ── Fix 5: Cross-timeframe directional agreement ───────────────────────
        # If the three timeframes are not pointing in the same direction the
        # signal is split and conviction is low.  Require at least 2/3 agreement.
        if len(predictions) >= 2:
            directions = [
                1 if data['prediction'] > current_price else -1
                for data in predictions.values()
            ]
            agreement_score = abs(sum(directions)) / len(directions)
            print(f"   Directional agreement score: {agreement_score:.2f} "
                  f"(1.0=full, 0.33=split)")
            agreement_veto = agreement_score < 0.67
            if agreement_veto:
                print(f"   VETO (agreement): timeframes disagree on direction "
                      f"({agreement_score:.2f} < 0.67)")
        else:
            agreement_score = 1.0
            agreement_veto = False

        trade_allowed = (
            not context['veto_active']
            and not uncertainty_veto
            and not agreement_veto
        )

        # Log this cycle's raw predictions so the next cycle can evaluate them
        # against the actual prices and feed the results into update_ensemble_weights.
        # This call was missing from the multi-TF path entirely — without it
        # pending_evaluations_*.json stays empty and weights can never evolve.
        timeframes_steps = {"1H": 1, "4H": 4, "1D": 24}
        self._log_prediction_for_evaluation(timeframes_steps, ensemble_predictions_map, current_price)

        # Save predictions and status
        status = {
            'last_update': datetime.now().isoformat(),
            'status': 'online',
            'symbol': self.symbol,
            'current_price': round(current_price, 5),
            'method': 'multi-timeframe',
            'ensemble_weights': {
                tf_key: [round(w, 3) for w in weights]
                for tf_key, weights in self.ensemble_weights.items()
            },
            'market_context': context,
            'agreement_score': round(agreement_score, 3),
            'uncertainty_veto': uncertainty_veto,
            'agreement_veto': agreement_veto,
            'trade_allowed': trade_allowed
        }

        self.save_to_file(self.predictions_file, predictions)
        self.save_to_file(self.status_file, status)

        # Display results
        print("\n--- Prediction Cycle Complete! ---")
        print(f"Current Price: {current_price:.5f}")
        for timeframe, data in predictions.items():
            direction = "UP" if data['prediction'] > current_price else "DOWN"
            change_pct = ((data['prediction'] - current_price) / current_price) * 100
            print(f"   {direction} {timeframe}: {data['prediction']:.5f} ({change_pct:+.3f}%) (±{data['ensemble_std']:.5f})")

    def _log_prediction_for_evaluation(self, timeframes_steps: Dict[str, int],
                                       ensemble_predictions_map: Dict[str, List[float]],
                                       current_price: float) -> None:
        """Log predictions for future evaluation."""
        try:
            with open(self.pending_eval_path, 'r') as f:
                pending = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            pending = []

        now = datetime.now()
        for tf_name, steps in timeframes_steps.items():
            if tf_name in ensemble_predictions_map:
                pending.append({
                    "eval_timestamp": (now + timedelta(hours=steps)).isoformat(),
                    "pred_timestamp": now.isoformat(),
                    "timeframe": tf_name,
                    "start_price": current_price,
                    "predictions": ensemble_predictions_map[tf_name]
                })

        with open(self.pending_eval_path, 'w') as f:
            json.dump(pending, f, indent=2)

    def _evaluate_past_predictions(self) -> None:
        """Evaluate past predictions against actual prices."""
        print("Evaluating past predictions for ensemble weighting...")
        try:
            with open(self.pending_eval_path, 'r') as f:
                pending = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            print("   No pending predictions to evaluate.")
            return

        remaining_evals = []
        evaluated_count = 0
        now = datetime.now()

        for entry in pending:
            try:
                eval_time = datetime.fromisoformat(entry['eval_timestamp'])

                # Ensure timezone-naive datetime for comparison
                if eval_time.tzinfo is not None:
                    eval_time = eval_time.replace(tzinfo=None)

                if now >= eval_time:
                    # Fetch actual price at evaluation time
                    rates = mt5.copy_rates_from(self.symbol, mt5.TIMEFRAME_H1, eval_time, 1)
                    if rates is not None and len(rates) > 0:
                        actual_future_price = rates[0]['close']
                        self.prediction_history[entry['timeframe']].append({
                            'predictions': entry['predictions'],
                            'actual': actual_future_price,
                            'timestamp': entry['pred_timestamp']
                        })
                        # Keep only recent history
                        if len(self.prediction_history[entry['timeframe']]) > self.ensemble_lookback:
                            self.prediction_history[entry['timeframe']].pop(0)
                        evaluated_count += 1
                    else:
                        # Keep for retry if data not available yet
                        remaining_evals.append(entry)
                else:
                    # Not time to evaluate yet
                    remaining_evals.append(entry)
            except Exception as e:
                print(f"   Error evaluating entry: {e}")
                continue

        print(f"   Evaluated {evaluated_count} predictions. {len(remaining_evals)} remaining.")

        # Save remaining evaluations
        with open(self.pending_eval_path, 'w') as f:
            json.dump(remaining_evals, f, indent=2)

    def update_ensemble_weights(self) -> None:
        """
        Update per-timeframe ensemble weights based on past prediction accuracy.

        Each timeframe (1H, 4H, 1D) maintains its own weight vector so that
        a model that excels on 1H but is mediocre on 1D gets rewarded
        independently for each horizon.  Previously a single shared list was
        updated by mixing errors across all timeframes, which muddied the
        signal — that bug is fixed here.
        """
        if not self.ensemble_weights:
            return

        print("Updating per-timeframe ensemble weights...")

        for tf_name, tf_history in self.prediction_history.items():
            if not tf_history:
                continue

            model_errors = [0.0] * self.num_ensemble_models
            total_samples = 0

            for entry in tf_history:
                actual = entry['actual']
                for i, pred in enumerate(entry['predictions']):
                    if i < len(model_errors):
                        model_errors[i] += abs(pred - actual)
                total_samples += 1

            if total_samples < 5:
                print(f"   {tf_name}: Not enough evaluated predictions ({total_samples}/5 minimum).")
                continue

            avg_errors = [err / max(total_samples, 1) for err in model_errors]

            # Softmax-style weighting: normalise by the best error then apply
            # temperature scaling to prevent any single model from dominating.
            min_error = min(avg_errors) if avg_errors else 1e-8
            normalized_errors = [err / max(min_error, 1e-8) for err in avg_errors]

            temperature = 2.0  # higher -> more equal weights; lower -> winner-takes-more
            exp_neg_errors = [np.exp(-err / temperature) for err in normalized_errors]
            total_exp = sum(exp_neg_errors)

            if total_exp == 0:
                print(f"   {tf_name}: WARNING: all softmax weights are zero — keeping current weights.")
                continue

            new_weights = [e / total_exp for e in exp_neg_errors]

            # Smooth update (exponential moving average over weight updates)
            current_weights = self.ensemble_weights.get(
                tf_name,
                [1.0 / self.num_ensemble_models] * self.num_ensemble_models
            )
            updated_weights = [
                (1 - self.ensemble_learning_rate) * old_w + self.ensemble_learning_rate * new_w
                for old_w, new_w in zip(current_weights, new_weights)
            ]

            # Re-normalise (float arithmetic can shift the sum slightly off 1.0)
            weight_sum = sum(updated_weights)
            if weight_sum > 0:
                updated_weights = [w / weight_sum for w in updated_weights]

            self.ensemble_weights[tf_name] = updated_weights

            print(f"   {tf_name} - Avg MAE per model: {[f'{e:.6f}' for e in avg_errors]}")
            print(f"   {tf_name} - Updated weights:   {[f'{w:.3f}' for w in updated_weights]}")

    def run_safe_backtest(self):
        """
        Walk-Forward Backtester.
        Fixes the 'Read-Ahead' cheating problem.
        """
        print("\n" + "=" * 80)
        print("Starting Safe Backtest (Walk-Forward Anti-Leakage)")
        print("=" * 80)

        # Try to load multi-timeframe models first, fall back to single-timeframe
        use_multitf = False
        if not self.models_by_timeframe:
            if self.load_model_assets_multitimeframe():
                use_multitf = True
                print("Using multi-timeframe models")
            elif not self.models:
                if not self.load_model_assets():
                    print("ERROR: Cannot run safe backtest without trained models.")
                    return
                print("Using single-timeframe models (less accurate)")
        else:
            use_multitf = True
            print("Using multi-timeframe models")

        # Download data scoped to the prediction window (with lookback padding)
        self._warn_if_predict_overlaps_training()
        if self.predict_start or self.predict_end:
            print(f"[DATE FILTER] Prediction window: "
                  f"{self.predict_start.strftime('%Y-%m-%d') if self.predict_start else 'beginning'} "
                  f"-> {self.predict_end.strftime('%Y-%m-%d') if self.predict_end else 'end'}\n")
        extra = timedelta(hours=2000 + 24)  # pad for walk-forward window
        dl_from = (self.predict_start - extra) if self.predict_start else None
        df_h1, df_h4, df_d1 = self.download_data(bars=15000, date_from=dl_from, date_to=self.predict_end)
        if df_h1 is None:
            return
            
        df_full = self.create_features(df_h1, df_h4, df_d1)
        df_selected = df_full[self.feature_cols + ['fwd_log_return_1h', 'fwd_log_return_4h', 'fwd_log_return_1d', 'close']]

        window = 2000  # Minimum training window
        step = 100     # Step size between predictions
        
        # Only use timeframes the EA supports
        timeframes = {"1H": 1, "4H": 4, "1D": 24}
        results = {tf: {'timestamps': [], 'actual': [], 'predicted': []} for tf in timeframes.keys()}
        
        total_iterations = (len(df_full) - window - 1) // step
        print(f"\nTotal iterations: {total_iterations}")
        print(f"Training window: {window} bars")
        print(f"Step size: {step} bars\n")

        iteration = 0
        for i in range(window, len(df_full) - 1, step):
            iteration += 1
            
            # Training only on past data (NO FUTURE LEAKAGE)
            past = df_full.iloc[:i]
            current_idx = i
            
            # Get current price and timestamp
            current_price = df_selected['close'].iloc[current_idx]
            timestamp = df_selected.index[current_idx]

            # Skip bars outside the requested prediction window
            ts_dt = timestamp.to_pydatetime() if hasattr(timestamp, 'to_pydatetime') else timestamp
            if self.predict_start and ts_dt < self.predict_start:
                continue
            if self.predict_end and ts_dt > self.predict_end:
                break
            
            # Make predictions for each timeframe
            for tf_name, steps in timeframes.items():
                ensemble_preds = []
                
                if use_multitf and tf_name in self.models_by_timeframe:
                    # Use multi-timeframe models (NO SCALING!)
                    models = self.models_by_timeframe[tf_name]
                    feature_scaler, target_scaler = self.scalers_by_timeframe[tf_name]
                    
                    # Refit scalers ONLY on past data (prevents look-ahead)
                    feature_scaler.fit(past[self.feature_cols])
                    
                    # Scale features
                    features_scaled = feature_scaler.transform(df_selected[self.feature_cols].iloc[:i+1].values)
                    
                    if current_idx >= self.lookback_periods:
                        X_pred_seq = features_scaled[current_idx - self.lookback_periods:current_idx].reshape(
                            1, self.lookback_periods, len(self.feature_cols)
                        )
                        X_pred_seq = tf.convert_to_tensor(X_pred_seq, dtype=tf.float32)
                        
                        for model_name, model in models.items():
                            try:
                                if 'lgbm' in model_name:
                                    # Prepare tabular data
                                    df_tabular = df_selected.iloc[:i+1].copy()
                                    for col in self.feature_cols:
                                        for lag in [1, 3, 5, 10]:
                                            new_col = f'{col}_lag_{lag}'
                                            df_tabular[new_col] = df_tabular[col].shift(lag)
                                    df_tabular.ffill(inplace=True)
                                    
                                    if timestamp in df_tabular.index:
                                        X_pred_tab = df_tabular.loc[timestamp][model.feature_name_].values.reshape(1, -1)
                                        pred_log_return = model.predict(X_pred_tab)[0]
                                    else:
                                        continue
                                else:
                                    pred_log_return_scaled = model(X_pred_seq, training=False).numpy()[0][0]
                                    pred_log_return = target_scaler.inverse_transform([[pred_log_return_scaled]])[0][0]
                                
                                # NO SCALING - multi-TF models predict for their specific timeframe
                                predicted_price = current_price * np.exp(pred_log_return)
                                
                                if not np.isnan(predicted_price) and not np.isinf(predicted_price):
                                    ensemble_preds.append(predicted_price)
                            except Exception:
                                continue
                else:
                    # Fallback: single-timeframe models with scaling
                    self.feature_scaler.fit(past[self.feature_cols])
                    self.target_scaler.fit(past[[self.target_column]])
                    
                    features_scaled = self.feature_scaler.transform(df_selected[self.feature_cols].iloc[:i+1].values)
                    
                    if current_idx >= self.lookback_periods:
                        X_pred_seq = features_scaled[current_idx - self.lookback_periods:current_idx].reshape(
                            1, self.lookback_periods, len(self.feature_cols)
                        )
                        X_pred_seq = tf.convert_to_tensor(X_pred_seq, dtype=tf.float32)
                        
                        for model_name, model in self.models.items():
                            try:
                                if 'lgbm' in model_name:
                                    continue  # Skip LGBM for simplicity
                                else:
                                    pred_log_return_scaled = model(X_pred_seq, training=False).numpy()[0][0]
                                    pred_log_return = self.target_scaler.inverse_transform([[pred_log_return_scaled]])[0][0]
                                
                                # Scale by sqrt(steps) for single-TF models
                                steps_adjusted = np.sqrt(steps) if steps > 1 else steps
                                predicted_price = current_price * np.exp(pred_log_return * steps_adjusted)
                                
                                if not np.isnan(predicted_price) and not np.isinf(predicted_price):
                                    ensemble_preds.append(predicted_price)
                            except Exception:
                                continue
                
                if ensemble_preds:
                    tf_weights = self.ensemble_weights.get(tf_name, [])
                    if tf_weights and len(tf_weights) == len(ensemble_preds):
                        weighted_price = np.average(ensemble_preds, weights=tf_weights)
                    else:
                        weighted_price = np.mean(ensemble_preds)

                    # Get actual future price (if available)
                    future_idx = min(current_idx + steps, len(df_selected) - 1)
                    actual_price = df_selected['close'].iloc[future_idx]
                    
                    results[tf_name]['timestamps'].append(timestamp)
                    results[tf_name]['predicted'].append(weighted_price)
                    results[tf_name]['actual'].append(actual_price)
            
            # Progress indicator
            if iteration % 10 == 0 or iteration == 1:
                progress = (iteration / total_iterations) * 100
                print(f"Progress: {progress:.1f}% | Bar: {timestamp} | Price: {current_price:.5f}")
        
        # Calculate and display metrics
        print("\n" + "=" * 80)
        print("SAFE BACKTEST RESULTS (No Look-Ahead Bias)")
        print("=" * 80)
        
        for tf_name in timeframes.keys():
            if len(results[tf_name]['predicted']) > 0:
                predicted = np.array(results[tf_name]['predicted'])
                actual = np.array(results[tf_name]['actual'])
                
                # Calculate metrics
                mae = np.mean(np.abs(predicted - actual))
                mape = np.mean(np.abs((actual - predicted) / actual)) * 100
                rmse = np.sqrt(np.mean((predicted - actual) ** 2))
                
                # Directional accuracy
                pred_direction = np.sign(np.diff(predicted))
                actual_direction = np.sign(np.diff(actual))
                directional_accuracy = np.mean(pred_direction == actual_direction) * 100
                
                print(f"\n{tf_name} Timeframe:")
                print(f"  MAE:  {mae:.5f}")
                print(f"  MAPE: {mape:.2f}%")
                print(f"  RMSE: {rmse:.5f}")
                print(f"  Directional Accuracy: {directional_accuracy:.2f}%")
        
        # Export results to CSV
        self.export_safe_backtest_results(results)
        print("\n" + "=" * 80)
        print("SAFE BACKTEST COMPLETE!")
        print("=" * 80)

    def export_safe_backtest_results(self, results: Dict[str, Dict[str, List]]) -> None:
        """Export safe backtest results to CSV files."""
        print("\nExporting safe backtest results...")
        
        for tf_name, data in results.items():
            if len(data['timestamps']) > 0:
                output_file = os.path.join(self.base_path, f'{self.symbol}_{tf_name}_safe_backtest.csv')
                
                try:
                    df_results = pd.DataFrame({
                        'timestamp': data['timestamps'],
                        'predicted': data['predicted'],
                        'actual': data['actual'],
                        'error': np.array(data['predicted']) - np.array(data['actual']),
                        'abs_error': np.abs(np.array(data['predicted']) - np.array(data['actual']))
                    })
                    
                    df_results.to_csv(output_file, index=False)
                    print(f"   Created: {output_file}")
                except Exception as e:
                    print(f"   Error creating {output_file}: {e}")

    def run_backtest_generation(self) -> None:
        """Generate historical predictions for backtesting."""
        print("\n" + "=" * 60)
        print("Starting Backtest Generation...")
        print("=" * 60)

        # Warn if prediction window overlaps training window
        self._warn_if_predict_overlaps_training()

        if self.predict_start or self.predict_end:
            print(f"[DATE FILTER] Prediction window: "
                  f"{self.predict_start.strftime('%Y-%m-%d') if self.predict_start else 'beginning'} "
                  f"-> {self.predict_end.strftime('%Y-%m-%d') if self.predict_end else 'end'}\n")

        # Try to load multi-timeframe models first, fall back to single-timeframe
        use_multitf = False
        if not self.models_by_timeframe:
            if self.load_model_assets_multitimeframe():
                use_multitf = True
                print("Using multi-timeframe models")
            elif not self.models:
                if not self.load_model_assets():
                    print("ERROR: No trained models found.")
                    return
                print("Using single-timeframe models (less accurate)")
        else:
            use_multitf = True
            print("Using multi-timeframe models")

        # Download historical data scoped to the prediction window
        # We need slightly more data than the window itself so the lookback
        # buffer (60 bars) is populated for the very first prediction.
        extra = timedelta(hours=self.lookback_periods + 24)  # a bit of padding
        dl_from = (self.predict_start - extra) if self.predict_start else None
        df_h1, df_h4, df_d1 = self.download_data(bars=40000, date_from=dl_from, date_to=self.predict_end)
        if df_h1 is None:
            return

        # Create features
        df = self.create_features(df_h1, df_h4, df_d1)
        df_selected = df[self.feature_cols + ['fwd_log_return_1h', 'fwd_log_return_4h', 'fwd_log_return_1d', 'close']]

        # Only generate for timeframes the EA supports
        timeframes = {"1H": 1, "4H": 4, "1D": 24}
        all_predictions = {tf: [] for tf in timeframes.keys()}
        timestamps = []

        print(f"Generating predictions for {len(df_selected) - self.lookback_periods} bars...")

        for i in range(self.lookback_periods, len(df_selected)):
            current_price = df_selected['close'].iloc[i]
            timestamp = df_selected.index[i]

            # --- Skip bars outside the requested prediction window ---
            ts_dt = timestamp.to_pydatetime() if hasattr(timestamp, 'to_pydatetime') else timestamp
            if self.predict_start and ts_dt < self.predict_start:
                continue
            if self.predict_end and ts_dt > self.predict_end:
                break

            # Get predictions for each timeframe
            for tf_name, steps in timeframes.items():
                ensemble_preds = []
                
                if use_multitf and tf_name in self.models_by_timeframe:
                    # Use multi-timeframe models (NO SCALING!)
                    models = self.models_by_timeframe[tf_name]
                    feature_scaler, target_scaler = self.scalers_by_timeframe[tf_name]
                    
                    # Scale features
                    features_scaled = feature_scaler.transform(df_selected[self.feature_cols].values)
                    X_pred_seq = features_scaled[i - self.lookback_periods:i].reshape(
                        1, self.lookback_periods, len(self.feature_cols)
                    )
                    X_pred_seq = tf.convert_to_tensor(X_pred_seq, dtype=tf.float32)
                    
                    for model_name, model in models.items():
                        try:
                            if 'lgbm' in model_name:
                                # Prepare tabular data for LightGBM
                                df_tabular = df_selected.iloc[:i+1].copy()
                                for col in self.feature_cols:
                                    for lag in [1, 3, 5, 10]:
                                        new_col = f'{col}_lag_{lag}'
                                        df_tabular[new_col] = df_tabular[col].shift(lag)
                                df_tabular.ffill(inplace=True)
                                
                                if timestamp in df_tabular.index:
                                    X_pred_tab = df_tabular.loc[timestamp][model.feature_name_].values.reshape(1, -1)
                                    pred_log_return = model.predict(X_pred_tab)[0]
                                else:
                                    continue
                            else:
                                pred_log_return_scaled = model(X_pred_seq, training=False).numpy()[0][0]
                                pred_log_return = target_scaler.inverse_transform([[pred_log_return_scaled]])[0][0]
                            
                            # NO SCALING by steps - multi-TF models already predict for their timeframe
                            predicted_price = current_price * np.exp(pred_log_return)
                            
                            if not np.isnan(predicted_price) and not np.isinf(predicted_price):
                                ensemble_preds.append(predicted_price)
                        except Exception:
                            continue
                else:
                    # Fallback: single-timeframe models with scaling
                    features_scaled = self.feature_scaler.transform(df_selected[self.feature_cols].values)
                    X_pred_seq = features_scaled[i - self.lookback_periods:i].reshape(
                        1, self.lookback_periods, len(self.feature_cols)
                    )
                    X_pred_seq = tf.convert_to_tensor(X_pred_seq, dtype=tf.float32)
                    
                    for model_name, model in self.models.items():
                        try:
                            if 'lgbm' in model_name:
                                continue  # Skip LGBM for simplicity in fallback mode
                            else:
                                pred_log_return_scaled = model(X_pred_seq, training=False).numpy()[0][0]
                                pred_log_return = self.target_scaler.inverse_transform([[pred_log_return_scaled]])[0][0]
                            
                            # Scale by sqrt(steps) for single-TF models
                            steps_adjusted = np.sqrt(steps) if steps > 1 else steps
                            predicted_price = current_price * np.exp(pred_log_return * steps_adjusted)
                            
                            if not np.isnan(predicted_price) and not np.isinf(predicted_price):
                                ensemble_preds.append(predicted_price)
                        except Exception:
                            continue

                if ensemble_preds:
                    tf_weights = self.ensemble_weights.get(tf_name, [])
                    if tf_weights and len(tf_weights) == len(ensemble_preds):
                        weighted_price = np.average(ensemble_preds, weights=tf_weights)
                    else:
                        weighted_price = np.mean(ensemble_preds)
                    all_predictions[tf_name].append(weighted_price)
                else:
                    all_predictions[tf_name].append(current_price)

            timestamps.append(timestamp)

            # Progress indicator
            if (i - self.lookback_periods) % 500 == 0:
                progress = ((i - self.lookback_periods) / (len(df_selected) - self.lookback_periods)) * 100
                print(f"   Progress: {progress:.1f}%")

        # Export backtest files
        self.export_backtest_files(timestamps, all_predictions)
        print("\nBACKTEST GENERATION COMPLETE!")

    def export_backtest_files(self, timestamps: List, predictions: Dict[str, List[float]]) -> None:
        """Export backtest predictions to CSV files."""
        print("\nExporting backtest files...")
        
        # Get the Common Files path for Strategy Tester
        # Common path is: AppData\Roaming\MetaQuotes\Terminal\Common\Files
        common_path = None
        try:
            appdata = os.environ.get('APPDATA', '')
            if appdata:
                common_path = os.path.join(appdata, 'MetaQuotes', 'Terminal', 'Common', 'Files')
                if not os.path.exists(common_path):
                    os.makedirs(common_path, exist_ok=True)
        except Exception as e:
            print(f"   Warning: Could not create Common Files folder: {e}")
        
        for tf_name, pred_values in predictions.items():
            # Save to regular MQL5\Files folder
            lookup_file = os.path.join(self.base_path, f'{self.symbol}_{tf_name}_lookup.csv')
            try:
                with open(lookup_file, 'w') as f:
                    f.write('timestamp,prediction\n')  # Add header
                    for ts, pred in zip(timestamps, pred_values):
                        f.write(f'{ts.strftime("%Y.%m.%d %H:%M")},{pred:.5f}\n')
                print(f"   Created: {lookup_file}")
            except Exception as e:
                print(f"   Error creating {lookup_file}: {e}")
            
            # ALSO save to Common Files folder for Strategy Tester
            if common_path:
                common_lookup_file = os.path.join(common_path, f'{self.symbol}_{tf_name}_lookup.csv')
                try:
                    with open(common_lookup_file, 'w') as f:
                        f.write('timestamp,prediction\n')
                        for ts, pred in zip(timestamps, pred_values):
                            f.write(f'{ts.strftime("%Y.%m.%d %H:%M")},{pred:.5f}\n')
                    print(f"   Created (Common): {common_lookup_file}")
                except Exception as e:
                    print(f"   Error creating Common file: {e}")
        
        print("\n" + "=" * 60)
        print("BACKTEST FILES CREATED")
        print("=" * 60)
        print(f"\nFiles saved to TWO locations:")
        print(f"  1. Regular:  {self.base_path}")
        print(f"  2. Common:   {common_path}")
        print(f"\nFor Strategy Tester, files MUST be in the Common folder.")
        print("=" * 60)

    def save_to_file(self, file_path: str, data: Dict) -> None:
        """Save data to JSON file."""
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"Error saving to {file_path}: {e}")

    def run_continuous(self, interval_minutes: int = 60) -> None:
        """
        Run predictions continuously at specified intervals.

        Args:
            interval_minutes: Minutes between prediction cycles
        """
        prediction_method = self.run_prediction_cycle_multitimeframe if self.use_multitimeframe else self.run_prediction_cycle

        print(f"\nStarting Continuous Mode for {self.symbol} (Interval: {interval_minutes} mins)")
        print(f"Using {'multi-timeframe' if self.use_multitimeframe else 'single-timeframe'} prediction method")

        while True:
            try:
                prediction_method()
                print(f"\nWaiting {interval_minutes} minutes until next cycle...")
                time.sleep(interval_minutes * 60)
            except KeyboardInterrupt:
                print("\nService stopped by user.")
                break
            except Exception as e:
                print(f"\nAn error occurred: {e}")
                import traceback
                traceback.print_exc()
                print("Retrying in 5 minutes...")
                time.sleep(300)


def main():
    """Main entry point for the predictor."""
    print("""
    ================================================================
       Hybrid Ensemble MT5 Predictor v8.2 (With Macro Integration)
       Feat: Transformer, TCN, GRU, LightGBM & Multi-Timeframe Support
       TensorFlow Optimized - 2-3x Faster Predictions
    ================================================================
    """)

    # ------------------------------------------------------------------
    # Shared date-range arguments added to every relevant sub-command via
    # a dedicated parent parser.  All dates are YYYY-MM-DD strings.
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Hybrid Ensemble MT5 Predictor v8.2")
    subparsers = parser.add_subparsers(dest='mode', required=True, help="Operating mode")

    # Parent: symbol (all modes)
    parent_sym = argparse.ArgumentParser(add_help=False)
    parent_sym.add_argument('--symbol', type=str, default="EURUSD", help="Currency symbol (default: EURUSD).")

    # Parent: training date window (train modes)
    parent_train_dates = argparse.ArgumentParser(add_help=False)
    parent_train_dates.add_argument(
        '--train-start', type=str, default=None, metavar='YYYY-MM-DD',
        help="Earliest date of training data (inclusive).  "
             "E.g. --train-start 2020-01-01"
    )
    parent_train_dates.add_argument(
        '--train-end', type=str, default=None, metavar='YYYY-MM-DD',
        help="Latest date of training data (inclusive).  "
             "Set this to your chosen cutoff so the model never sees future data.  "
             "E.g. --train-end 2023-12-31"
    )

    # Parent: prediction / backtest date window (backtest + predict modes)
    parent_pred_dates = argparse.ArgumentParser(add_help=False)
    parent_pred_dates.add_argument(
        '--predict-start', type=str, default=None, metavar='YYYY-MM-DD',
        help="Start of the date range to generate prediction CSV rows for.  "
             "Should be >= --train-end to avoid look-ahead bias.  "
             "E.g. --predict-start 2024-01-01"
    )
    parent_pred_dates.add_argument(
        '--predict-end', type=str, default=None, metavar='YYYY-MM-DD',
        help="End of the date range to generate prediction CSV rows for.  "
             "E.g. --predict-end 2024-12-31"
    )

    # ------------------------------------------------------------------
    # Sub-commands
    # ------------------------------------------------------------------

    # train  (single-timeframe, legacy)
    p_train = subparsers.add_parser(
        'train', parents=[parent_sym, parent_train_dates],
        help="Train the model ensemble (single timeframe, legacy)."
    )
    p_train.add_argument('--force', action='store_true', help="Force retraining even if saved models exist.")
    p_train.add_argument(
        '--models', nargs='+',
        default=['lstm', 'transformer', 'lgbm'],
        choices=['lstm', 'gru', 'transformer', 'tcn', 'lgbm'],
        help="Model types to include in the ensemble."
    )

    # train-multitf  (recommended)
    p_train_mtf = subparsers.add_parser(
        'train-multitf', parents=[parent_sym, parent_train_dates],
        help="Train separate ensembles for 1H/4H/1D (RECOMMENDED)."
    )
    p_train_mtf.add_argument('--force', action='store_true', help="Force retraining even if saved models exist.")
    p_train_mtf.add_argument(
        '--models', nargs='+',
        default=['lstm', 'transformer', 'lgbm'],
        choices=['lstm', 'gru', 'transformer', 'tcn', 'lgbm'],
        help="Model types to include in each timeframe ensemble."
    )

    # tune
    subparsers.add_parser('tune', parents=[parent_sym], help="Run hyperparameter tuning for DL models.")

    # predict  (single-timeframe, live)
    p_predict = subparsers.add_parser(
        'predict', parents=[parent_sym],
        help="Run a live prediction cycle (single timeframe)."
    )
    p_predict.add_argument('--continuous', action='store_true', help="Loop continuously.")
    p_predict.add_argument('--interval', type=int, default=60, help="Minutes between cycles in continuous mode.")
    p_predict.add_argument('--models', nargs='+', choices=['lstm', 'gru', 'transformer', 'tcn', 'lgbm'],
                           help="Override automatic model detection.")
    p_predict.add_argument('--no-kalman', action='store_true', help="Disable Kalman filtering (use EMA).")

    # predict-multitf  (recommended live mode)
    p_predict_mtf = subparsers.add_parser(
        'predict-multitf', parents=[parent_sym],
        help="Run a live prediction cycle using timeframe-specific models (RECOMMENDED)."
    )
    p_predict_mtf.add_argument('--continuous', action='store_true', help="Loop continuously.")
    p_predict_mtf.add_argument('--interval', type=int, default=60, help="Minutes between cycles in continuous mode.")
    p_predict_mtf.add_argument('--models', nargs='+', choices=['lstm', 'gru', 'transformer', 'tcn', 'lgbm'],
                               help="Override automatic model detection.")
    p_predict_mtf.add_argument('--no-kalman', action='store_true', help="Disable Kalman filtering (use EMA).")

    # backtest  (generate lookup CSVs for MT5 Strategy Tester)
    subparsers.add_parser(
        'backtest', parents=[parent_sym, parent_pred_dates],
        help="Generate prediction lookup CSVs for MT5 Strategy Tester.  "
             "Use --predict-start / --predict-end to restrict the date range."
    )

    # safe-backtest  (walk-forward, no look-ahead)
    subparsers.add_parser(
        'safe-backtest', parents=[parent_sym, parent_pred_dates],
        help="Walk-forward backtest that prevents look-ahead bias.  "
             "Use --predict-start / --predict-end to restrict the date range."
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Build predictor keyword arguments from parsed args
    # ------------------------------------------------------------------
    predictor_args: Dict[str, Any] = {'symbol': args.symbol.upper()}

    # Training date window
    if hasattr(args, 'train_start') and args.train_start:
        predictor_args['train_start'] = args.train_start
    if hasattr(args, 'train_end') and args.train_end:
        predictor_args['train_end'] = args.train_end

    # Prediction / backtest date window
    if hasattr(args, 'predict_start') and args.predict_start:
        predictor_args['predict_start'] = args.predict_start
    if hasattr(args, 'predict_end') and args.predict_end:
        predictor_args['predict_end'] = args.predict_end

    # Mode-specific args
    if args.mode in ['train', 'train-multitf']:
        predictor_args['ensemble_model_types'] = args.models
        predictor_args['use_multitimeframe'] = (args.mode == 'train-multitf')
    elif args.mode in ['predict', 'predict-multitf']:
        if hasattr(args, 'models') and args.models:
            predictor_args['ensemble_model_types'] = args.models
        predictor_args['use_kalman'] = not (hasattr(args, 'no_kalman') and args.no_kalman)
        predictor_args['use_multitimeframe'] = (args.mode == 'predict-multitf')

    # Print resolved date windows so user can confirm before training starts
    if any(k in predictor_args for k in ('train_start', 'train_end', 'predict_start', 'predict_end')):
        print("Date windows resolved:")
        if 'train_start' in predictor_args or 'train_end' in predictor_args:
            print(f"  Train  : {predictor_args.get('train_start', 'beginning')} "
                  f"-> {predictor_args.get('train_end', 'latest available')}")
        if 'predict_start' in predictor_args or 'predict_end' in predictor_args:
            print(f"  Predict: {predictor_args.get('predict_start', 'beginning')} "
                  f"-> {predictor_args.get('predict_end', 'latest available')}")
        print()

    # Initialize predictor
    predictor = UnifiedLSTMPredictor(**predictor_args)

    # Execute requested mode
    try:
        if args.mode == 'tune':
            predictor.tune_hyperparameters()
        elif args.mode == 'train':
            predictor.train_model(force_retrain=args.force)
        elif args.mode == 'train-multitf':
            predictor.train_model_multitimeframe(force_retrain=args.force)
        elif args.mode == 'predict':
            if args.continuous:
                predictor.run_continuous(interval_minutes=args.interval)
            else:
                predictor.run_prediction_cycle()
        elif args.mode == 'predict-multitf':
            if args.continuous:
                predictor.run_continuous(interval_minutes=args.interval)
            else:
                predictor.run_prediction_cycle_multitimeframe()
        elif args.mode == 'backtest':
            predictor.run_backtest_generation()
        elif args.mode == 'safe-backtest':
            predictor.run_safe_backtest()
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        mt5.shutdown()
        print("\nShutdown complete. Thank you!")


if __name__ == "__main__":
    main()
