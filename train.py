import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
import gc

from models import create_lstm_model  

OPT_EPOCHS = 15
OPT_PATIENCE = 3

def objective_function_lstm(params, X_train, y_train, X_val, y_val):
    model = None 

    try:
        model = create_lstm_model(
            params,
            X_train.shape[1],
            X_train.shape[2]
        )

        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=OPT_PATIENCE,
            restore_best_weights=True,
            verbose=0
        )

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=OPT_EPOCHS,
            batch_size=params['batch_size'],
            callbacks=[early_stop],
            verbose=0
        )

        best_val_loss = min(history.history['val_loss'])

    except Exception as e:
        print("\n❌ Objective function error")
        print("Params:", params)
        print("Error:", e)
        best_val_loss = np.inf


    finally:
        if model is not None:
            del model
        K.clear_session()
        gc.collect()

    return best_val_loss