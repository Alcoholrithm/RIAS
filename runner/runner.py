from types import SimpleNamespace
from typing import Dict, Any, Type, List
from numpy.typing import NDArray

import pandas as pd
import numpy as np
import random

import optuna
from sklearn.model_selection import train_test_split, StratifiedKFold

from sklearn import metrics
import torchmetrics

import pickle
from datetime import datetime
import os

from netcal import scaling, binning
from netcal.metrics import ECE

import dice_ml

import lime.lime_tabular

from .models import BaseModel
from .misc.eval_metric import EvalMetric

class Runner(object):
    def __init__(self, config: SimpleNamespace, model_class: Type[BaseModel], X: pd.DataFrame, y: np.array,
                    continuous_cols: List[str], categorical_cols: List[str]) -> None:
        
        self.start_time = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        
        self.config = config
        self.model_class = model_class
        self.continuous_cols = continuous_cols
        self.categorical_cols = categorical_cols
        self.model = None
        self.calibrator = None
        
        if hasattr(self.config.model, 'gpus'):
            os.environ["CUDA_VISIBLE_DEVICES"]=",".join(map(str, config.model.gpus))
        
        self.X = X
        self.y = y
        
        self.random_seed = self.config.experiment.random_seed
        self.set_random_seed()
        
        self.init_scorer()
        
        if self.config.experiment.fast_dev_run:
            self.config.experiment.early_stop_patience = 1
            self.config.experiment.optuna.n_trials = 1
            self.config.experiment.KFold = 1
            
    
    def check_config(self):
        ### To DO
        assert self.config.experiment.metric != None, "1"
        assert self.config.experiment.metric_params != None, "2"
        assert self.config.experiment.data_config != None, "3"
        assert self.config.experiment.optuna.direction != None, "4"
    
    def set_random_seed(self):
        # Setting seed for Python's random module
        random.seed(self.random_seed )

        # Setting seed for NumPy's random module
        np.random.seed(self.random_seed)

        try:
            # Setting seed for TensorFlow
            import tensorflow as tf
            tf.random.set_seed(self.random_seed)
        except:
            # print("Pass tensorflow when fix the random seed")
            pass

        try:
            # Setting seed for PyTorch
            import torch
            torch.manual_seed(self.random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_seed)
        except:
            # print("Pass pytorch when fix the random seed")
            pass

    def init_scorer(self) -> None:
        """Init evaluation metric"""
        
        self.scorer_dict = {}
        
        for param in self.config.experiment.metric_params:
            self.scorer_dict[param[0]] = param[1]

        if hasattr(metrics, self.config.experiment.metric):
            self.scorer = getattr(metrics, self.config.experiment.metric)
        elif hasattr(torchmetrics.functional, self.config.experiment.metric):
            self.scorer = getattr(torchmetrics.functional, self.config.experiment.metric)
        else:
            raise("Unknown Scorer")
    
    def init_calibrator(self):
        assert hasattr(scaling, self.config.experiment.calibrator) or hasattr(binning, self.config.experiment.calibrator), "Use only calibratior in netcal.scaling and netcal.binning"
        assert self.model is not None, "Must have trained model to calibrate its confidence"
        
        self.set_random_seed()
        X_train, X_valid, y_train, y_valid = train_test_split(self.X, self.y, test_size = self.config.experiment.valid_size, random_state=self.random_seed)
        
        self.ece = ECE(self.config.experiment.ece_bins)

        if hasattr(scaling, self.config.experiment.calibrator):
            self.calibrator = getattr(scaling, self.config.experiment.calibrator)()

            preds_proba = self.model.predict_proba(X_valid)

            self.calibrator.fit(preds_proba, y_valid)
            
            uncalibrated_score = self.ece.measure(preds_proba, y_valid)
            
            calibrated = self.calibrator.transform(preds_proba)
            calibrated_score = self.ece.measure(calibrated, y_valid)
            
            print("Uncalibrated ECE :", uncalibrated_score)
            print("Calibrated ECE :", calibrated_score)
            
        elif hasattr(binning, self.config.experiment.calibrator):
            self.calibrator = getattr(binning, self.config.experiment.calibrator)
            preds_proba = self.model.predict_proba(X_valid)
            
            min_ece = 987654321
                    
            n_bins = self.config.experiment.ece_bins
            ece = ECE(n_bins)
            uncalibrated_score = ece.measure(preds_proba, y_valid)
            
            for b in range(5, 20):
                calibrator = getattr(binning, self.config.experiment.calibrator)(bins = b)
                calibrator.fit(preds_proba, y_valid)
                calibrated = calibrator.transform(preds_proba)
                
                calibrated_score = ece.measure(calibrated, y_valid)
                if calibrated_score < min_ece:
                    min_ece = calibrated_score
                    self.calibrator = calibrator
            
            calibrated = self.calibrator.transform(preds_proba)
            calibrated_score = self.ece.measure(calibrated, y_valid)
            
            print("Uncalibrated ECE :", uncalibrated_score)
            print("Calibrated ECE :", calibrated_score)
    
    def get_score(self, 
                    y_true: NDArray[np.int_], 
                    y_pred: NDArray[np.int_]
        ) -> float:
        """Generates a score of the predicted result.

        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.
        
        Returns:
            A score of the given result using self.scorer.
        """
        self.scorer_dict['y_true'] = y_true
        self.scorer_dict['y_pred'] = y_pred
        return self.scorer(**self.scorer_dict)
    
    def objective(self, 
                    trial: optuna.trial.Trial, 
                    train_idx: NDArray[np.int_]= None, 
                    valid_idx: NDArray[np.int_] = None
        ) -> float:
        """Objective function for optuna

        Args:
            trial: A object which returns hyperparameters of a model of hyperparameter search trial.
            train_idx: Indices of training data in self.X and self.y.
            valid_idx: Indices of valid data in self.X and self.y.
        
        Returns:
            A score of given hyperparameters.
        """
        self.set_random_seed()
        
        hparams = {}
        for k, v in self.config.model.search_range.items():
            hparams[k] = getattr(trial, v[0])(*v[1])

        model_params = {
            "config" : self.config,
            "hparams" : hparams,
            "continuous_cols" : self.continuous_cols,
            "categorical_cols" : self.categorical_cols
        }
        # model = self.model_class(config = self.config, hparams = hparams)
        model = self.model_class(**model_params)

        if train_idx is None or valid_idx is None:
            X_train, X_valid, y_train, y_valid = train_test_split(self.X, self.y, test_size = self.config.experiment.valid_size, random_state=self.random_seed)
        else:
            X_train, y_train = self.X.iloc[train_idx], self.y[train_idx]
            X_valid, y_valid = self.X.iloc[valid_idx], self.y[valid_idx]
        
        model.fit(X_train, y_train, X_valid, y_valid)
        preds = model.predict(X_valid)
        
        score = self.get_score(y_valid, preds)
        
        return score
    
    def get_hparams(self) -> Dict[str, Any]:
        """Search the best hyperparameters for a given model.

        Search the best hyperparameters for a given model using optuna. If the experiment using k-fold cross validation,
        optuna also use k-fold cross validation.
        Returns:
            The best hyperparameters.
        """
        
        assert self.config.model.search_range is not None, "No search range for hyperparameter tunning"
        
        def objective_cv(trial : optuna.trial.Trial) -> float:
            """Objective function of optuna with k-fold cross validation.

            Args:
                trial: A object which returns hyperparameters of a model of hyperparameter search trial.
            
            Returns:
                Average score of hyperparameters over k-fold.
            """
            fold = StratifiedKFold(n_splits=self.config.experiment.KFold, shuffle=True, random_state=self.random_seed)
            scores = []
            
            
            for fold_idx, (train_idx, test_idx) in enumerate(fold.split(self.X, self.y)):
                score = self.objective(trial, train_idx, test_idx)
                scores.append(score)
            
            return np.mean(scores)
        
        opt = optuna.create_study(direction=self.config.experiment.optuna.direction,sampler=optuna.samplers.TPESampler(seed=self.random_seed))

        if self.config.experiment.KFold > 1:
            opt.optimize(objective_cv, n_trials=self.config.experiment.optuna.n_trials)
        else:
            opt.optimize(self.objective, n_trials=self.config.experiment.optuna.n_trials)

        trial = opt.best_trial
        hparams = dict(trial.params.items())

        if self.config.experiment.save_hparams:
            self.save_hparams(hparams)
        
        print("Best Parameters")
        print(hparams)

        return hparams
        
    def save_hparams(self, 
                    hparams: Dict[str, Any]
        ) -> None:
        """Saves the given hyperparameters.

        Args:
            hparams: The hyperparameters to save.
        """
        path = f'hparams/{self.config.experiment.data_config}-{self.model_class.__name__}-{self.start_time}.pickle'
        
        if not os.path.exists('hparams'):
            os.mkdir('hparams')
            
        with open(path, 'wb') as f:
            pickle.dump(hparams, f)
    
    def predict(self, X_test: pd.DataFrame) -> np.array:
        assert self.model is not None, "Must train the model"
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test: pd.DataFrame) -> np.array:
        assert self.model is not None, "Must train the model"
        if type(X_test) == np.ndarray:
            X_test = pd.DataFrame(X_test.reshape((-1, X_test.shape[-1])), columns = self.X.columns)
        X_test = X_test.astype(self.X.dtypes)

        return self.model.predict_proba(X_test)
    
    def train(self) -> None:
        if self.config.model.hparams is None:
            self.config.model.hparams = self.get_hparams()
            
        if isinstance(self.config.model.hparams, str):
            self.config.model.hparams = pickle.load(open(self.config.model.hparams, 'rb'))
            
        model_params = {
            "config" : self.config,
            "continuous_cols" : self.continuous_cols,
            "categorical_cols" : self.categorical_cols,
            "hparams" : None
        }

        self.model = self.model_class(**model_params)
        
        self.set_random_seed()
        X_train, X_valid, y_train, y_valid = train_test_split(self.X, self.y, test_size = self.config.experiment.valid_size, random_state=self.random_seed)
        self.model.fit(X_train, y_train, X_valid, y_valid)
        
        preds = self.model.predict(X_valid)
        
        print("Validation Score: %.4f" % self.get_score(y_valid, preds))
    
    def test(self, X_test: pd.DataFrame, y_test: np.array, eval_metric: EvalMetric = None) -> None:
        if eval_metric is None:
            preds = self.model.predict(X_test)
            print("Test Score: %.4f" % self.get_score(y_test, preds))
        else:
            eval_metric(self.model, X_test, y_test)
    
    def dice(self, X_test: pd.DataFrame) -> None:

        dice_data = self.X.copy()  
        dice_data["target"] = self.y
        
        d = dice_ml.Data(dataframe=dice_data,
                        continuous_features=self.continuous_cols,
                        outcome_name='target')
        # Pre-trained ML model
        m = dice_ml.Model(model = self, backend=self.config.dice.backend, func=self.config.dice.func)
        # DiCE explanation instance
        exp = dice_ml.Dice(d,m)
        

        dice_exp = exp.generate_counterfactuals(X_test, total_CFs=self.config.dice.total_CFs, desired_class=self.config.dice.desired_class, **self.config.dice.additional_kwargs)

        dice_exp.visualize_as_dataframe()
        
        return dice_exp
    
    def lime(self, sample: pd.Series) -> None:
        
        categorical_features = []
        for idx, col in enumerate(self.X.columns):
            if col in self.categorical_cols:
                categorical_features.append(idx)
                
        print("########## The result of lime for the given sample ##########")
        explainer = lime.lime_tabular.LimeTabularExplainer(self.X.values, 
                                                        feature_names=self.X.columns, 
                                                        class_names=self.config.lime.class_names, 
                                                        categorical_features=categorical_features, 
                                                        categorical_names=self.categorical_cols,
                                                        verbose=self.config.lime.verbose , 
                                                        mode="regression" if self.config.experiment.task == "regression" else "classification", 
                                                        discretize_continuous=self.config.lime.discretize_continuous,
                                                        random_state = self.config.experiment.random_seed,
                                                        **self.config.lime.kwargs)

        exp = explainer.explain_instance(
                                            sample, 
                                            self.predict_proba, 
                                            num_features=self.X.shape[-1] if self.config.lime.num_features is None else self.config.lime.num_features)

        exp.save_to_file(self.config.lime.file)
        print()
        return exp

    def save_model(self) -> None:
        pass
    
    def load_model(self) -> None:
        pass
