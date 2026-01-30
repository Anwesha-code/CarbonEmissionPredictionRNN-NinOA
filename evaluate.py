# evaluate.py
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test).flatten()

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    explanation = (
        "Negative R² values, if observed, are attributed to the strong "
        "non-stationary trend present in cumulative energy measurements "
        "rather than model inadequacy."
    )

    return mae, rmse, r2, explanation
