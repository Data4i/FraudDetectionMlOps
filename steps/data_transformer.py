from zenml.steps import step
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from typing import Annotated, Tuple
import pandas as pd
import numpy as np


@step
def data_transformer(
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: np.ndarray, y_test: np.ndarray
) -> Tuple[
    Annotated[np.ndarray, "Transformed X_train"],
    Annotated[np.ndarray, "y_train"],
    Annotated[np.ndarray, "Transformed X_test"],
    Annotated[np.ndarray, "y_test"],
]:
    """Evaluate the model on the train data

    Keyword arguments:
        X_train -- train features
        X_test -- test labels

    Return:
        Model: trained model
        Transformation Pipeline: pipeline to transform features
    """

    num_cols = list(X_train.select_dtypes(include=np.number))
    cat_cols = list(X_train.select_dtypes(exclude=np.number))

    ct = ColumnTransformer(
        (
            ("num_cols", StandardScaler(), num_cols),
            ("cat_cols", OneHotEncoder(), cat_cols),
        )
    )

    ct.fit(X_train)

    transformed_X_train = ct.transform(X_train)
    transformed_X_test = ct.transform(X_test)

    return transformed_X_train, y_train, transformed_X_test, y_test
