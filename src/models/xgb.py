from .base_model import BaseModel
import pandas as pd
import numpy as np
from types import SimpleNamespace
from typing import Dict, Any
from xgboost import XGBClassifier, XGBRegressor
from copy import deepcopy
class XGB(BaseModel):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        config = kwargs["config"]
        hparams = kwargs["hparams"]
        assert (hparams is not None) or (self.config.model.hparams is not None), "No model hyperparameters"
        
        if hparams == None:
            hparams = self.config.model.hparams

        hparams.update(self.config.model.additional_hparams)
        self.xgb_class = XGBRegressor if self.config.experiment.task == "regression" else XGBClassifier
        self.model = self.xgb_class(**hparams)
        
    
    def fit(self, X_train: pd.DataFrame, y_train: np.array, X_valid: pd.DataFrame, y_valid: np.array) -> None:
        fit_params = deepcopy(self.config.model.fit_params)
        self.model.fit(X_train,  y_train, eval_set=[(X_valid, y_valid)], **fit_params)
        
    def predict(self, X_test: pd.DataFrame) -> np.array:
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test: pd.DataFrame) -> np.array:
        return self.model.predict_proba(X_test)
    
    def save_model(self, saving_path: str = None) -> None:
        assert saving_path is not None, "saving_path cannot be None"
        
        if saving_path.split('.')[-1] != 'json':
            saving_path += '.json'
            
        self.model.save_model(saving_path)
        return saving_path

    def load_model(self, model_path: str = None) -> None:
        
        self.model = self.xgb_class()
        if model_path is None:
            self.model.load_model(self.config.model.model_path)
        else:
            self.model.load_model(model_path)
