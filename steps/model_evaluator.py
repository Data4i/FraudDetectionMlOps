from zenml.steps import step
from sklearn.base import ClassifierMixin
from sklearn.metrics import f1_score
from typing import Annotated
import numpy as np
import mlflow


@step
def model_evaluator(
    model: ClassifierMixin,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Annotated[float, "f1_evaluation_score"]:
    """Evaluate the model on the train data

    Keyword arguments:
        model -- trained Model
        X_train -- train features
        y_train -- train labels

    Return: f1_score (float)
    """
    mlflow.autolog()
    
    y_pred = model.predict(X_train)
    f1_evaluation_score = f1_score(y_train, y_pred)
    
    mlflow.log_metric('evaluation_f1_score', f1_evaluation_score)
    
    return f1_evaluation_score
