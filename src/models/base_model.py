from abc import ABC, abstractmethod
from types import SimpleNamespace
import pandas as pd
import numpy as np

class BaseModel(ABC):
    def __init__(self, config: SimpleNamespace) -> None:
        self.config = config
    
    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: np.array, X_valid: pd.DataFrame, y_valid: np.array) -> None:
        pass
    
    @abstractmethod
    def predict(self, X_test: pd.DataFrame) -> np.array:
        pass
    
    def predict_proba(self, X_test: pd.DataFrame) -> np.array:
        ### For classifier ###
        pass
    
    def save_model(self) -> None:
        pass
    
    def load_model(self) -> None:
        pass