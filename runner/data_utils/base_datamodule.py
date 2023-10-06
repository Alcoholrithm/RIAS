import os
import pickle
from datetime import datetime
from abc import ABC, abstractmethod
from types import SimpleNamespace
from typing import Tuple, List
import pandas as pd

class DataModule(ABC):
    """A base module to load data
    
    Attributes:
        config: A configuration of the given experiment.
    """
    def __init__(self,
        task: str
        ) -> None:
        """Inits the datamodule

        Args:
            config: A configuration of the given experiment.
        """
        self.task = task
    
    @abstractmethod
    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load features and their labels from data repository or local file

        Returns:
            Tuple[pd.DataFrame, pd.Series]: A tuple of features and labels
        """
        pass
    
    @abstractmethod
    def prepare_data(self) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
        """Return processed data for the experiment.

        If there is cached data that is specified in config, just return them.
        Else, load raw data using 'load_data' method, and process it in appropriate way.
        If there is save_data option in config, save processed data or just return the data.
        
        Returns:
            Tuple[pd.DataFrame, pd.Series, List[str], List[str]]: _description_
        """
        pass

    def save_data(self, 
                data: pd.DataFrame, 
                label: pd.Series,
                continuous_cols: List[str], 
                categorical_cols: List[str]
        ) -> None:
        """Save processed data if there is save_data option in config

        Args:
            data (pd.DataFrame): processed features
            label (pd.Series): corresponding label of data
        """
        now = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        saving_path = f"./data/{now}-{self.task}.pickle"
        if not os.path.exists("./data"):
            os.mkdir("./data")
        with open(saving_path, 'wb') as f:
            pickle.dump({
                    'data' : data,
                    'label' : label,
                    "continuous_cols" : continuous_cols,
                    "categorical_cols" : categorical_cols
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL
            )
