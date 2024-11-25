from zenml.steps import step
from typing import Annotated, Tuple
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split

@step
def data_splitter(data: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, 'X_train'],
    Annotated[pd.DataFrame, 'X_test'],
    Annotated[np.ndarray, 'y_train'],
    Annotated[np.ndarray, 'y_test']
]:
    """Splitting Data into Train and Test sets
    
    Keyword arguments:
    data -- data to be split
    Return:
        X_train -- train features
        X_test -- test features
        y_train -- train labels
        y_test -- test labels
    """
    
    try:
        
        X = data.drop('isFraud', axis=1)
        y = data['isFraud']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.1)
    except Exception as e:
        return logging.info(f'An error was encountered while splitting the data ->> {e}')
    
    return X_train, X_test, y_train.values, y_test.values