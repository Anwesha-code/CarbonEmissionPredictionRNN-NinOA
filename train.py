# train.py
from tensorflow.keras.callbacks import EarlyStopping
from models import create_lstm_model

def objective_function_lstm(params, X_train, y_train, X_val, y_val):
    model = create_lstm_model(
        params,
        X_train.shape[1],
        X_train.shape[2]
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=params["batch_size"],
        callbacks=[early_stop],
        verbose=0
    )

    return min(history.history["val_loss"])
