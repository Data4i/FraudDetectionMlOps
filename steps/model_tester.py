from zenml.steps import step
from sklearn.base import ClassifierMixin
from sklearn.metrics import f1_score
from typing import Annotated
import numpy as np


@step
def model_tester(
    model: ClassifierMixin,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Annotated[float, "f1_test_score"]:
    """Test the model on the test data

    Keyword arguments:
        model -- trained Model
        X_test -- test features
        y_test -- test labels

    Return: f1_score (float)
    """

    y_pred = model.predict(X_test)
    f1_test_score = f1_score(y_test, y_pred)
    
    return f1_test_score
