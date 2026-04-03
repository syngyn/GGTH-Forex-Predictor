"""
GGTH Predictor - Model Builders
================================
All custom Keras layers and deep-learning model construction logic,
extracted from unified_predictor_v8.py.

Keeping these here means:
  - unified_predictor_v8.py is the orchestrator (train/predict/backtest)
  - model_builders.py is the only file you touch when changing architectures
  - New model types are added here without touching orchestrator logic

Public API
----------
    build_dl_model(model_type, input_shape, hp=None) → Compiled Keras Model
    KalmanFilter
    TransformerBlock   (Keras serializable)
    AttentionLayer     (Keras serializable)

Supported model_type values
----------------------------
    "lstm"        Bidirectional LSTM + Attention
    "gru"         Bidirectional GRU (2 layers)
    "transformer" Multi-head self-attention (TransformerBlock)
    "tcn"         Temporal Convolutional Network with residual skip
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
import keras
from keras import layers
from keras.models import Model
from keras.optimizers import Adam
import keras_tuner as kt

from logger import get_logger

log = get_logger("ggth.models")


# ═══════════════════════════════════════════════════════════════════════════════
#  Helper / lightweight classes
# ═══════════════════════════════════════════════════════════════════════════════

class KalmanFilter:
    """
    Scalar 1-D Kalman filter for smoothing per-timeframe predictions.

    Args:
        process_variance:     Q — how much the true signal can drift per step.
        measurement_variance: R — how noisy each raw prediction is.

    Typical configs (from unified_predictor_v8.py):
        1H → Q=0.00001, R=0.01
        4H → Q=0.00005, R=0.02
        1D → Q=0.0001,  R=0.05
    """

    def __init__(self, process_variance: float, measurement_variance: float) -> None:
        self.q = process_variance
        self.r = measurement_variance
        self.x = 0.0   # state estimate
        self.p = 1.0   # error covariance
        self.k = 0.0   # Kalman gain

    def update(self, measurement: float) -> float:
        """Incorporate a new measurement and return the updated state estimate."""
        self.p += self.q
        self.k  = self.p / (self.p + self.r)
        self.x += self.k * (measurement - self.x)
        self.p  = (1.0 - self.k) * self.p
        return self.x

    def reset(self) -> None:
        """Reset filter state (useful at the start of each prediction cycle)."""
        self.x = 0.0
        self.p = 1.0
        self.k = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  Custom Keras layers  (must be registered for model.save / load_model)
# ═══════════════════════════════════════════════════════════════════════════════

@keras.saving.register_keras_serializable(package="Custom", name="TransformerBlock")
class TransformerBlock(layers.Layer):
    """
    Standard Transformer encoder block:
        Multi-Head Attention → Add & Norm → FFN → Add & Norm

    Args:
        embed_dim:  Dimensionality of the input embeddings (= number of features).
        num_heads:  Number of attention heads.
        ff_dim:     Hidden units in the feed-forward sub-layer.
        rate:       Dropout rate applied after attention and FFN.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim:    int,
        rate:      float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim    = ff_dim
        self.rate      = rate

        self.att        = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn        = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1   = layers.Dropout(rate)
        self.dropout2   = layers.Dropout(rate)

    def call(self, inputs, training: bool = False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1        = self.layernorm1(inputs + attn_output)
        ffn_output  = self.ffn(out1)
        ffn_output  = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim":    self.ff_dim,
            "rate":      self.rate,
        })
        return cfg

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="Custom", name="AttentionLayer")
class AttentionLayer(layers.Layer):
    """
    Additive (Bahdanau-style) attention over a sequence of LSTM outputs.

    Input:  (batch, timesteps, features)
    Output: (batch, features)  — context vector, ready for Dense layers.

    Learns three weight matrices:
        W  projects hidden states
        b  bias term
        u  context vector (dot-producted with tanh(W·h + b))
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def build(self, input_shape):
        d = input_shape[-1]
        self.W = self.add_weight(
            name="attention_weight",
            shape=(d, d),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="attention_bias",
            shape=(d,),
            initializer="zeros",
            trainable=True,
        )
        self.u = self.add_weight(
            name="attention_context",
            shape=(d,),
            initializer="glorot_uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        # inputs: (batch, T, d)
        score              = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        attention_weights  = tf.nn.softmax(tf.tensordot(score, self.u, axes=1), axis=1)
        context_vector     = tf.reduce_sum(
            inputs * tf.expand_dims(attention_weights, -1), axis=1
        )
        return context_vector

    def get_config(self):
        return super().get_config()

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# ═══════════════════════════════════════════════════════════════════════════════
#  Individual model constructors  (private — called by build_dl_model)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_lstm(
    inputs,
    lstm_units:   int,
    dropout_rate: float,
):
    """
    Bidirectional LSTM with additive attention.

    Architecture:
        BiLSTM(units, return_seq=True) → Dropout → LayerNorm → AttentionLayer
    """
    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True))(inputs)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.LayerNormalization()(x)
    x = AttentionLayer()(x)
    log.debug("LSTM block built  — units=%d, dropout=%.2f", lstm_units, dropout_rate)
    return x


def _build_gru(
    inputs,
    lstm_units:   int,
    dropout_rate: float,
):
    """
    Two-layer Bidirectional GRU.

    Architecture:
        BiGRU(units, return_seq=True) → Dropout → LayerNorm
        → BiGRU(units//2, return_seq=False) → Dropout

    Lighter than LSTM; often comparable performance on FX 1H data.
    """
    x = layers.Bidirectional(layers.GRU(lstm_units, return_sequences=True))(inputs)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Bidirectional(layers.GRU(lstm_units // 2, return_sequences=False))(x)
    x = layers.Dropout(dropout_rate)(x)
    log.debug("GRU block built   — units=%d/%d, dropout=%.2f",
              lstm_units, lstm_units // 2, dropout_rate)
    return x


def _build_transformer(
    inputs,
    input_shape:  Tuple[int, int],
    dropout_rate: float,
):
    """
    Single TransformerBlock followed by GlobalAveragePooling.

    embed_dim is set to the number of input features so no projection is needed.
    num_heads=4 and ff_dim=64 are intentionally conservative for low-data forex.
    """
    x = TransformerBlock(
        embed_dim=input_shape[1],
        num_heads=4,
        ff_dim=64,
        rate=dropout_rate,
    )(inputs)
    x = layers.GlobalAveragePooling1D()(x)
    log.debug("Transformer block built — embed_dim=%d, dropout=%.2f",
              input_shape[1], dropout_rate)
    return x


def _build_tcn(
    inputs,
    input_shape:  Tuple[int, int],
    conv_filters: int,
    dropout_rate: float,
):
    """
    Temporal Convolutional Network with residual skip connection.

    Dilations [1, 2, 4, 8, 16] give a receptive field of ~60 steps,
    matching the 60-bar lookback used across all timeframes.

    Residual path: project input to conv_filters if feature dims differ,
    then add to the final conv output before pooling.
    """
    dilations = [1, 2, 4, 8, 16]
    x = inputs

    for i, d in enumerate(dilations):
        x = layers.Conv1D(
            filters=conv_filters,
            kernel_size=3,
            dilation_rate=d,
            padding="causal",
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        if i < len(dilations) - 1:          # dropout on all but final conv
            x = layers.Dropout(dropout_rate)(x)

    # Residual skip — project input features → conv_filters if necessary
    if input_shape[1] != conv_filters:
        residual = layers.Conv1D(filters=conv_filters, kernel_size=1, padding="same")(inputs)
    else:
        residual = inputs

    x = layers.Add()([x, residual])
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling1D()(x)

    log.debug(
        "TCN block built   — filters=%d, dilations=%s, dropout=%.2f",
        conv_filters, dilations, dropout_rate,
    )
    return x


# ═══════════════════════════════════════════════════════════════════════════════
#  Public factory function
# ═══════════════════════════════════════════════════════════════════════════════

def build_dl_model(
    model_type:  str,
    input_shape: Tuple[int, int],
    hp:          Optional[kt.HyperParameters] = None,
) -> Model:
    """
    Build and compile a deep-learning model for 1-step log-return prediction.

    Args:
        model_type:  One of "lstm", "gru", "transformer", "tcn".
        input_shape: (lookback_steps, n_features) — e.g. (60, 25).
        hp:          keras_tuner HyperParameters object. When provided,
                     units / filters / dropout / lr are tuned; otherwise
                     sensible defaults are used.

    Returns:
        Compiled Keras Model (loss=huber, optimizer=Adam).

    Raises:
        ValueError: If model_type is not recognised.

    Example:
        model = build_dl_model("lstm", (60, 25))
        model.fit(X_train, y_train, ...)
    """
    # ── Hyperparameter defaults (overridden by tuner when hp is not None) ───
    lstm_units   = 64
    conv_filters = 64
    dropout_rate = 0.3
    learning_rate = 0.0005

    if hp is not None:
        lstm_units    = hp.Int("lstm_units",    32, 128, step=32)
        conv_filters  = hp.Int("conv_filters",  32, 128, step=32)
        dropout_rate  = hp.Float("dropout",     0.2, 0.5, step=0.1)
        learning_rate = hp.Choice("learning_rate", [1e-3, 5e-4, 1e-4])

    log.info("Building %s model — input_shape=%s, tuning=%s",
             model_type.upper(), input_shape, hp is not None)

    # ── Shared input ─────────────────────────────────────────────────────────
    inputs = layers.Input(shape=input_shape)

    # ── Architecture-specific backbone ───────────────────────────────────────
    if model_type == "lstm":
        x = _build_lstm(inputs, lstm_units, dropout_rate)

    elif model_type == "gru":
        x = _build_gru(inputs, lstm_units, dropout_rate)

    elif model_type == "transformer":
        x = _build_transformer(inputs, input_shape, dropout_rate)

    elif model_type == "tcn":
        x = _build_tcn(inputs, input_shape, conv_filters, dropout_rate)

    else:
        raise ValueError(
            f"Unknown DL model type: '{model_type}'. "
            f"Valid options: 'lstm', 'gru', 'transformer', 'tcn'."
        )

    # ── Shared output head (identical for all architectures) ─────────────────
    x       = layers.Dense(128, activation="relu")(x)
    x       = layers.Dropout(dropout_rate)(x)
    x       = layers.Dense(64,  activation="relu")(x)
    outputs = layers.Dense(1,   activation="linear")(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="huber",
        metrics=["mae"],
    )

    log.info(
        "%s model compiled — params=%s, lr=%.5f",
        model_type.upper(),
        f"{model.count_params():,}",
        learning_rate,
    )
    return model


# ── Quick smoke-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    _SHAPE = (60, 25)

    for _mtype in ("lstm", "gru", "transformer", "tcn"):
        _m = build_dl_model(_mtype, _SHAPE)
        print(f"  {_mtype.upper():12s}  params={_m.count_params():>10,}  "
              f"output={_m.output_shape}")
