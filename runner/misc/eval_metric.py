from typing import Type
import pandas as pd
import numpy as np
from ..models import BaseModel
from abc import ABC, abstractmethod

class EvalMetric(ABC):
    @abstractmethod
    def eval(self, model: Type[BaseModel], X_test: pd.DataFrame, y_test: np.array):
        pass
    
    def __call__(self, model: Type[BaseModel], X_test: pd.DataFrame, y_test: np.array):
        return self.eval(model, X_test, y_test)