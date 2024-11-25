from zenml.steps import step
from typing import Annotated
import pandas as pd
import logging

@step
def data_loader(data_path: str) -> Annotated[pd.DataFrame, 'Full Dataset']:
    """Loading Data Via Pandas to DataFrame
    
    Keyword arguments:
    data_path (str) -- data_path for the dataset
    
    Return: Full Dataset (pd.DataFrame)
    """
    
    try:
        data = pd.read_csv(data_path)
    except FileNotFoundError:
        logging.info(f'File {data_path} not available')
        
    return data