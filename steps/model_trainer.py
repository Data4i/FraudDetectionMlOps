from zenml.steps import step
from zenml.client import Client
import mlflow
from sklearn.base import ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from typing import Annotated
import numpy as np
from materializers.decision_tree_materializer import DecisionTreeClassifierMaterializer

@step(output_materializers=DecisionTreeClassifierMaterializer, experiment_tracker='fraud_detection_mlflow_experiment_tracker')
def model_trainer(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int
) -> Annotated[ClassifierMixin, 'Model']:
    """Evaluate the model on the train data

    Keyword arguments:
        X_train -- train features
        y_train -- train labels

    Return: Model: trained model
    """
    
    mlflow.sklearn.autolog()
    
    model = DecisionTreeClassifier(random_state = random_state)
    
    model.fit(X_train, y_train)
    model_params = model.get_params()
    
    for param_name in model_params.keys():
        mlflow.log_param(param_name, model_params[param_name])
    
    return model
