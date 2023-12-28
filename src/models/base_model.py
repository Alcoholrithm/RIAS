from abc import ABC, abstractmethod
from types import SimpleNamespace
import pandas as pd
import numpy as np

class BaseModel(ABC):
    def __init__(self, **kwargs) -> None:
        """An abstract base class for model of RIAS

        Args:
            **kwargs: A dictionary that has requirements of the model.
        """
        self.config = kwargs["config"]
    
    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: np.array, X_valid: pd.DataFrame, y_valid: np.array) -> None:
        """Fit the model using the given training and validation datasets.

        Args:
            X_train (pd.DataFrame): The features of the training dataset.
            y_train (np.array): The labels of the training dataset.
            X_valid (pd.DataFrame): The features of the validation dataset.
            y_valid (np.array): The labels of the validation dataset.
        """
        pass
    
    @abstractmethod
    def predict(self, X_test: pd.DataFrame) -> np.array:
        """Predict with `X_test`

        Args:
            X_test (pd.DataFrame): A set of features to predict.

        Returns:
            np.array: The prediction for a given X_test.
        """
        pass
    
    def predict_proba(self, X_test: pd.DataFrame) -> np.array:
        """Predict the probability of each `X_test` example being of a given class.

        Args:
            X_test (pd.DataFrame): A set of features to predict.

        Returns:
            np.array: The predicted probability for a given X_test.
        """
        pass
    
    @abstractmethod
    def save_model(self, saving_path: str = None) -> str:
        """Save the model to the specified path

        Args:
            saving_path (str, optional): A path where the model will be saved. Defaults to None.

        Returns:
            str: A path where the model will be saved.
        """
        pass
    
    @abstractmethod
    def load_model(self, model_path: str = None) -> None:
        """Load the model from the specified path

        Args:
            model_path (str, optional): A path where the model will be loaded. Defaults to None.
        """
        pass