from .base_model import BaseModel
import pandas as pd
import numpy as np
from types import SimpleNamespace
from typing import List, Dict, Tuple, Union, Any
from numpy.typing import NDArray

from pytorch_tabular.config import (
    DataConfig,
    TrainerConfig,
    ModelConfig,
    OptimizerConfig
)

from pytorch_tabular import TabularModel
from pytorch_tabular.utils import get_balanced_sampler, get_class_weighted_cross_entropy
import os

from abc import ABC, abstractmethod

class PytorchTabularBase(BaseModel, ABC):
    def __init__(self, **kwargs):
        # config: SimpleNamespace, continuous_cols: List[str], categorical_cols: List[str]
        config = kwargs["config"]
        continuous_cols = kwargs["continuous_cols"]
        categorical_cols = kwargs["categorical_cols"]
        super().__init__(config)
        
        os.environ["CUDA_VISIBLE_DEVICES"]=",".join(map(str, self.config.model.gpus))
        
        if self.config.experiment.optuna.direction == 'maximize':
            metric_mode = 'max'
        elif self.config.experiment.optuna.direction == 'minimize':
            metric_mode = 'min'
            
        self.data_config = DataConfig(
                    target=['target'],
                    continuous_cols=continuous_cols,
                    categorical_cols=categorical_cols,
                    continuous_feature_transform=None,
                    normalize_continuous_features=True,
                    num_workers=self.config.experiment.n_jobs,
                )
        
        self.config.experiment.metric = "accuracy" if self.config.experiment.metric == "accuracy_score" else self.config.experiment.metric
        
        self.trainer_config = TrainerConfig(
                accelerator = 'gpu' if self.config.model.gpus[0] != -2 else 'cpu',
                devices = -1 if self.config.model.gpus[0] != -2 else self.config.experiment.n_jobs,
                auto_select_gpus = config.model.auto_select_gpus,
                fast_dev_run=config.experiment.fast_dev_run, 
                max_epochs=config.model.max_epochs, 
                batch_size=config.model.batch_size,
                early_stopping_patience = config.experiment.early_stopping_patience,
                gradient_clip_val = 1,
                checkpoints = None,
                early_stopping='valid_' + self.config.experiment.metric,
                early_stopping_mode = metric_mode,
                deterministic = True,
                seed = self.config.experiment.random_seed,
                )
        
        assert (kwargs["hparams"] == None) or (self.config.model.hparams == None), "No model hyperparameters"
        
        if kwargs["hparams"] == None:
            hparams = self.config.model.hparams
        else:
            hparams = kwargs["hparams"]
            
        self.model = self.get_model(hparams)
        
    @abstractmethod
    def get_model_config(self, 
                        hparams: Dict[str, Any]
        ) -> ModelConfig:
        pass

    def get_optimizer_config(self) -> OptimizerConfig:
        optimizer_config = OptimizerConfig(
            optimizer=self.config.model.optimizer,
            optimizer_params=self.config.model.optimizer_params,
            lr_scheduler=self.config.model.lr_scheduler,
            lr_scheduler_params=self.config.model.lr_scheduler_params,
        )
        return optimizer_config
    
    def get_model(self, 
                params: Dict[str, Any]
        ) -> None:
        model_config = self.get_model_config(params)
        optimizer_config = self.get_optimizer_config()
        model = TabularModel(
            data_config=self.data_config,
            model_config=model_config,
            optimizer_config = optimizer_config, 
            trainer_config=self.trainer_config,
        )
        
        return model
    
    def rename_prob_cols(self, 
                        pred_output: pd.DataFrame
        ) -> pd.DataFrame:
        
        cnt = 0
        for col in pred_output.columns:
            if 'probability' in col:
                cnt += 1
                
        for y in range(cnt):
            if ("%d.0_probability" % y) in pred_output.columns:
                pred_output.rename(columns={('%d.0_probability' % y) : ('%d_probability' % y)}, inplace=True)
        
        return pred_output, cnt
    
    def fit(self, X_train: pd.DataFrame, y_train: np.array, X_valid: pd.DataFrame, y_valid: np.array) -> None:
        train = X_train.copy()
        train["target"] = y_train
        
        valid = X_valid.copy()
        valid["target"] = y_valid
        
        sampler = None
        if self.config.model.use_balanced_sampler:
            sampler = get_balanced_sampler(train['target'].values.ravel())
            
        fit_params = {"train" : train, "validation" : valid, "train_sampler" : sampler, "min_epochs" : 1, "seed" : self.config.experiment.random_seed}
        if self.config.model.use_weighted_loss:
            fit_params["loss"] = get_class_weighted_cross_entropy(train["target"].values.ravel(), mu=self.config.model.mu)

        self.model.fit(**fit_params)
    
    def predict(self, 
                X_test: pd.DataFrame = None,
                return_proba: bool = False
        ) -> Union[Tuple[NDArray[np.int_], NDArray[np.float_]], NDArray[np.int_]]:
        
        X_test['target'] = [0 for _ in range(len(X_test))]
        
        preds = self.model.predict(X_test)
        preds, cnt = self.rename_prob_cols(preds)

        prob_cols = []
        for y in range(cnt):
            prob_cols.append("%d_probability" % y) 
            
        if return_proba:
            return preds[prob_cols].values
        else:
            return preds['prediction']
        
    def predict_proba(self, X_test: pd.DataFrame) -> np.array:
        return self.predict(X_test, True)
    
    def save_model(self, 
                    saving_path: str = None
        ) -> None:
        self.model.save_model(saving_path)
    
    def load_model(self) -> None:
        self.model = TabularModel.load_from_checkpoint(self.config.model.model_path, strict=False)