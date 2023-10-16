from .base_model import BaseModel
import pandas as pd
import numpy as np
from types import SimpleNamespace
from typing import Dict, Any
from lightgbm import LGBMClassifier, LGBMRegressor
from copy import deepcopy
import pickle

class LGBM(BaseModel):
    def __init__(self, **kwargs) -> None:
        # config: SimpleNamespace = None, hparams: Dict[str, Any] = None
        config = kwargs["config"]
        hparams = kwargs["hparams"]
        super().__init__(config)
        assert (hparams == None) or (self.config.model.hparams == None), "No model hyperparameters"
        
        if hparams == None:
            hparams = self.config.model.hparams

        hparams.update(self.config.model.additional_hparams)
        self.lgbm_class = LGBMRegressor if self.config.experiment.task == "regression" else LGBMClassifier
        
        self.model = self.lgbm_class(**hparams)
        
    
    def fit(self, X_train: pd.DataFrame, y_train: np.array, X_valid: pd.DataFrame, y_valid: np.array) -> None:
        fit_params = deepcopy(self.config.model.fit_params)
        self.model.fit(X_train,  y_train, eval_set=[(X_valid, y_valid)], **fit_params)
        
    def predict(self, X_test: pd.DataFrame) -> np.array:
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test: pd.DataFrame) -> np.array:
        return self.model.predict_proba(X_test)
    
    def save_model(self, saving_path: str = None) -> str:
        if saving_path.split('.')[-1] != 'pickle':
            saving_path += '.pickle'
        pickle.dump(self.model, open(saving_path, 'wb'))
        
        return saving_path

    def load_model(self, model_path: str = None) -> None:
        if model_path is None:
            self.model = pickle.load(open(self.config.model.model_path, 'rb'))
        else:
            self.model = pickle.load(open(model_path, 'rb'))