from zenml.steps import step
from typing import Annotated
import pandas as pd
import logging

@step
def data_cleaner(data: pd.DataFrame) -> Annotated[pd.DataFrame, 'Cleaned Dataset']:
    """Cleaning The Data
    
    Keyword arguments:
    data (pd.DataFrame) -- Uncleaned Data
    
    Return: Cleaned Dataset (pd.DataFrame)
    """
    
    try:
        data.dropna(axis=0, inplace=True)
        data.drop(['nameDest', 'isFlaggedFraud', 'step', 'nameOrig'], inplace=True, axis=1)
    except IndexError as e:
        logging.info(f'Something went wrong while cleaning data ->>> {e}')
        
    return data