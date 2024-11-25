from zenml.steps import step
from sklearn.base import ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from typing import Annotated
import numpy as np

@step
def model_trainer(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Annotated[ClassifierMixin, 'Model']:
    """Evaluate the model on the train data

    Keyword arguments:
        X_train -- train features
        y_train -- train labels

    Return: Model: trained model
    """
    
    model = DecisionTreeClassifier(random_state = 42)
    
    model.fit(X_train, y_train)
    
    return model
