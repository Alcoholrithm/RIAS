from typing import Type
import pandas as pd
import numpy as np
from ..models import BaseModel
from abc import ABC, abstractmethod
from typing import Dict

class EvalMetric(ABC):
    @abstractmethod
    def eval(self, model: Type[BaseModel], X_test: pd.DataFrame, y_test: np.array) -> Dict[str, float]:
        pass
    
    def __call__(self, model: Type[BaseModel], X_test: pd.DataFrame, y_test: np.array) -> Dict[str, float]:
        return self.eval(model, X_test, y_test)