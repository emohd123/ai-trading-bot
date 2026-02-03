"""
LSTM Neural Network for price direction prediction.
Processes sequences of 24 candles to capture temporal patterns.
"""
import numpy as np
import pandas as pd
import os
from typing import Optional, Tuple

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


SEQ_LENGTH = 24  # Number of timesteps (candles) per sequence


def create_lstm_model(n_features: int, seq_length: int = SEQ_LENGTH):
    """
    Create LSTM model for sequence classification.
    
    Architecture:
    - Input: (seq_length, n_features)
    - LSTM 64 units, return sequences
    - Dropout 0.3
    - LSTM 32 units
    - Dropout 0.3
    - Dense 16, ReLU
    - Output: 1, Sigmoid
    """
    if not TENSORFLOW_AVAILABLE:
        return None
    
    model = keras.Sequential([
        layers.Input(shape=(seq_length, n_features)),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.3),
        layers.LSTM(32, return_sequences=False),
        layers.Dropout(0.3),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_sequences(X: np.ndarray, y: np.ndarray, seq_length: int = SEQ_LENGTH) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for LSTM from flat feature matrix.
    Each sample is seq_length consecutive rows of features.
    """
    X_seq = []
    y_seq = []
    
    for i in range(seq_length, len(X)):
        X_seq.append(X[i - seq_length:i])
        y_seq.append(y[i])
    
    return np.array(X_seq), np.array(y_seq)


def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seq_length: int = SEQ_LENGTH,
    epochs: int = 50,
    batch_size: int = 32
) -> Optional[object]:
    """Train LSTM model on sequence data."""
    if not TENSORFLOW_AVAILABLE:
        return None
    
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_length)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, seq_length)
    
    if len(X_train_seq) < 10 or len(X_val_seq) < 5:
        return None
    
    n_features = X_train.shape[1]
    model = create_lstm_model(n_features, seq_length)
    
    if model is None:
        return None
    
    model.fit(
        X_train_seq, y_train_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
        ]
    )
    
    return model


def predict_lstm(model, X: np.ndarray, scaler=None, seq_length: int = SEQ_LENGTH) -> float:
    """
    Predict probability of UP from last seq_length rows of features.
    X should be (n_samples, n_features) - uses last seq_length rows.
    """
    if model is None or not TENSORFLOW_AVAILABLE:
        return 0.5
    
    if len(X) < seq_length:
        return 0.5
    
    X_last = X[-seq_length:]
    if scaler is not None:
        X_last = scaler.transform(X_last)
    
    X_seq = np.expand_dims(X_last, axis=0)
    proba = model.predict(X_seq, verbose=0)
    return float(proba[0][0])
