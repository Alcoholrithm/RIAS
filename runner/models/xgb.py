from .base_model import BaseModel
import pandas as pd
import numpy as np
from types import SimpleNamespace
from typing import Dict, Any
from xgboost import XGBClassifier, XGBRegressor
from copy import deepcopy
class XGB(BaseModel):
    def __init__(self, **kwargs) -> None:
        # config: SimpleNamespace = None, hparams: Dict[str, Any] = None
        config = kwargs["config"]
        hparams = kwargs["hparams"]
        super().__init__(config)
        assert (hparams == None) or (self.config.model.hparams == None), "No model hyperparameters"
        
        if hparams == None:
            hparams = self.config.model.hparams

        hparams["early_stopping_rounds"] = self.config.experiment.early_stopping_patience
        hparams["enable_categorical"] = True
        hparams["n_jobs"] = self.config.experiment.n_jobs
        xgb_class = XGBRegressor if self.config.experiment.task == "regression" else XGBClassifier
        self.model = xgb_class(**hparams)
        
    
    def fit(self, X_train: pd.DataFrame, y_train: np.array, X_valid: pd.DataFrame, y_valid: np.array) -> None:
        fit_params = deepcopy(self.config.model.fit_params)
        self.model.fit(X_train,  y_train, eval_set=[(X_valid, y_valid)], **fit_params)
        
    def predict(self, X_test: pd.DataFrame) -> np.array:
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test: pd.DataFrame) -> np.array:
        return self.model.predict_proba(X_test)
    
    def save_model(self, saving_path: str = None) -> str:
        self.model.save_model(saving_path)

    def load_model(self) -> None:
        self.model = XGBClassifier()
        self.model.load_model(self.config.model.model_path)
        
    def __call__(self, X_test: pd.DataFrame) -> np.array:
        return self.predict(X_test)