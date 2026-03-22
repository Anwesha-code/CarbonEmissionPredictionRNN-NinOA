import numpy as np
import time
from typing import Dict, Tuple, List, Callable
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, LSTM, Bidirectional, Dropout,
    BatchNormalization, GlobalMaxPooling1D, Dense
)



def create_lstm_model(params: Dict, seq_len: int, num_feats: int) -> Sequential:
    model = Sequential(name='LSTM_NiOA')
    model.add(Input(shape=(seq_len, num_feats)))

    model.add(Bidirectional(LSTM(params['units'], return_sequences=True)))

    for _ in range(params['lstm_layers'] - 1):
        model.add(LSTM(params['units'], return_sequences=True))
        model.add(Dropout(params['dropout']))

    model.add(BatchNormalization())
    model.add(GlobalMaxPooling1D())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(params['dropout']))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(1))

    if params['optimizer'] == 'adamw':
        optimizer = tf.keras.optimizers.experimental.AdamW(
            learning_rate=params['learning_rate']
        )
    else:
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=params['learning_rate']
        )

    model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
    return model
