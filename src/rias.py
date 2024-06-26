from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, Any, Type, List, Union
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
import shap
from BorutaShap import BorutaShap
from tqdm import tqdm

from .models import BaseModel
from .misc.eval_metric import EvalMetric

import importlib

class RIAS(object):
    def __init__(self, config: SimpleNamespace, model_class: Type[BaseModel], X: pd.DataFrame, y: np.array,
                    continuous_cols: List[str], categorical_cols: List[str], calibrate: bool = False) -> None:
        """A reliable and interpretable AI system.

        TODO
        
        Args:
            config (SimpleNamespace): A namespace that has predefined options for RIAS.
            model_class (Type[BaseModel]): The class of the model which uses in RIAS. It should be inherited BaseModel class of RIAS.
            X (pd.DataFrame): A dataset which uses in RIAS.
            y (np.array): A numpy array of the labels corresponding to X.
            continuous_cols (List[str]): A list of continuous_cols columns in X.
            categorical_cols (List[str]): A list of categorical columns in X.
            calibrate (bool, optional): A flag that decide calibrate the prediction or not. Defaults to False.
        """
        
        self.start_time = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        
        self.config = config
        self.model_class = model_class
        self.continuous_cols = continuous_cols
        self.categorical_cols = categorical_cols
        self.model = None
        self.calibrate = calibrate
        self.calibrator = None
        
        self.shap_explainer = None
        self.shap_path = None
        self.feature_selector = None

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
        """Check the given config is valid or not
        """
        assert self.config.experiment.metric != None, "1"
        assert self.config.experiment.metric_params != None, "2"
        assert self.config.experiment.data_config != None, "3"
        assert self.config.experiment.optuna.direction != None, "4"
    
    def set_random_seed(self):
        """Set random seeds as predefined random seed in config
        """
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
        """Initialize evaluation metric
        The scorer must be supported by scikit-learn.metrics or torchmetrics.functional
        """
        
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
        """Initialize a calibrator for an ai model of RIAS
        The calibrator is fitted to validation set based on ECE score.
        It must be supported by netcal.scaling or netcal.binning.
        """
        assert hasattr(scaling, self.config.experiment.calibrator) or hasattr(binning, self.config.experiment.calibrator), "Use only calibratior in netcal.scaling and netcal.binning"
        assert self.model is not None, "Must have trained model to calibrate its confidence"
        
        self.set_random_seed()
        _, X_valid, _, y_valid = train_test_split(self.X, self.y, test_size = self.config.experiment.valid_size, random_state=self.random_seed)
        
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
        """Generates a score of the predicted result using scorer

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
        """Objective function for Optuna

        Args:
            trial (optuna.trial.Trial): A object which returns hyperparameters of a model of hyperparameter search trial.
            train_idx (NDArray[np.int_], optional): Indices of training data in self.X and self.y. Defaults to None.
            valid_idx (NDArray[np.int_], optional): Indices of valid data in self.X and self.y. Defaults to None.
        
        Returns:
            float: A score of given hyperparameters.
        """
        self.set_random_seed()
        
        hparams = {}
        for k, v in self.config.model.search_range.items():
            hparams[k] = getattr(trial, v[0])(*v[1])

        model = self.get_model(hparams)

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
            Dict[str, Any]: The best hyperparameters.
        """
        
        assert self.config.model.search_range is not None, "No search range for hyperparameter tunning"
        
        def objective_cv(trial : optuna.trial.Trial) -> float:
            """Objective function of optuna with k-fold cross validation.

            Args:
                trial (optuna.trial.Trial): A object which returns hyperparameters of a model of hyperparameter search trial.
            
            Returns:
                float: Average score of hyperparameters over k-fold.
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
            hparams (Dict[str, Any]): The hyperparameters to save.
        """
        path = f'hparams/{self.config.experiment.data_config}-{self.model_class.__name__}-{self.start_time}.pickle'
        
        if not os.path.exists('hparams'):
            os.mkdir('hparams')
            
        with open(path, 'wb') as f:
            pickle.dump(hparams, f)
    
    def check_input(self, X_test: Union[pd.DataFrame, np.ndarray, pd.Series]) -> pd.DataFrame:
        """Ensuring the X_test as pandas dataframe

        Args:
            X_test (Union[pd.DataFrame, np.ndarray, pd.Series]): The input data.

        Returns:
            pd.DataFrame: A pandas dataframe of X_test
        """
        if type(X_test) == np.ndarray:
            X_test = pd.DataFrame(X_test.reshape((-1, X_test.shape[-1])), columns = self.X.columns)
        elif isinstance(X_test, pd.Series):
            X_test = pd.DataFrame(X_test.values.reshape((-1, X_test.shape[-1])), columns = self.X.columns)
        X_test = X_test.astype(self.X.dtypes)
        
        return X_test
    
    def predict(self, X_test: pd.DataFrame) -> np.array:
        """Return prediction for given samples

        Args:
            X_test (pd.DataFrame): A set of samples to predict.

        Returns:
            np.array: A set of predictions
        """
        assert self.model is not None, "Must train the model"
        X_test = self.check_input(X_test)
        
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test: pd.DataFrame) -> np.array:
        """Return predicted probabilities for each class for given samples
        If the calibrate flag is True, the probabilities are calibrated.
        
        Args:
            X_test (pd.DataFrame): A set of samples to predict their probabilities for each class.

        Returns:
            np.array: A numpy array of predicted probabilities
        """
        assert self.model is not None, "Must train the model"
        X_test = self.check_input(X_test)

        proba = self.model.predict_proba(X_test)
        
        if self.calibrate:
            proba = self.calibrator.transform(proba)
        
        if len(proba.shape) == 1:
            _proba = np.zeros((len(X_test), 2))
            _proba[:, 0] = 1 - proba
            _proba[:, 1] = proba
            proba = _proba
        proba = np.clip(proba, a_max=1, a_min=0)
        return proba
    
    def get_model(self, hparams: Dict[str, Any] = None) -> Type[BaseModel]:
        """Return a model object according to the model class of RIAS

        Args:
            hparams (Dict[str, Any], optional): The hyperparameter for the model. Defaults to None.

        Returns:
            Type[BaseModel]: A model object.
        """
        
        model_params = {
            "config" : self.config,
            "continuous_cols" : self.continuous_cols,
            "categorical_cols" : self.categorical_cols,
            "hparams" : hparams
        }

        return self.model_class(**model_params)
        
    def train(self) -> None:
        """Train a model and initialize the calibrator if self.calibrate is True
        
        If there are given hyperparamters, train the model using them.
        Else find out the optimized hyperparameters and train with them.
        
        The training dataset and validation datasets for training are derived from self.X and self.y. 
        And they are automatically determined by the config.experiment.valid_size option.
        
        After, training procedure is the end, print the validation score and initialize the calibrator if self.calibrate is True
        """
        if self.config.model.hparams is None:
            self.config.model.hparams = self.get_hparams()
            
        if isinstance(self.config.model.hparams, str):
            self.config.model.hparams = pickle.load(open(self.config.model.hparams, 'rb'))
            
        self.model = self.get_model()
        
        self.set_random_seed()
        X_train, X_valid, y_train, y_valid = train_test_split(self.X, self.y, test_size = self.config.experiment.valid_size, random_state=self.random_seed)
        self.model.fit(X_train, y_train, X_valid, y_valid)
        
        preds = self.model.predict(X_valid)
        
        print("Validation Score: %.4f" % self.get_score(y_valid, preds))
        
        if self.calibrate:
            self.init_calibrator()
    
    def test(self, X_test: pd.DataFrame, y_test: np.array, eval_metric: Type[EvalMetric] = None) -> Dict[str, float]:
        """Test the trained model using given test dataset
        
        Evaluate the model using the given test dataset.
        If the eval_metric is None, use the self.scorer.
        
        Args:
            X_test (pd.DataFrame): A dataset to test the model.
            y_test (np.array): A numpy array of the labels corresponding to X_test.
            eval_metric (EvalMetric, optional): An object that inherits the EvalMetric class. Defaults to None.

        Returns:
            Dict[str, float]: An evaluation results using eval_metric or self.scorer.
        """
        if eval_metric is None:
            preds = self.model.predict(X_test)
            score = self.get_score(y_test, preds)
            print("Test Score: %.4f" % score)
            return {
                "Test Score" : score
            }
        else:
            return eval_metric(self.model, X_test, y_test)
    
    def get_counterfactual_explanations(self, X_test: pd.DataFrame, total_CFs: int = 4, desired_class: int = 0, features_to_vary: Union[str, List[str]] = 'all', **kwargs) -> None:
        """Return counterfactual explanations using DiCE using given samples

        Args:
            X_test (pd.DataFrame): The samples to find out counterfactual explanations.

        Returns:
            _type_: Counterfactual explanations.
        """
        X_test = self.check_input(X_test)
        
        dice_data = self.X.copy()  
        dice_data["target"] = self.y
        
        d = dice_ml.Data(dataframe=dice_data,
                        continuous_features=self.continuous_cols,
                        outcome_name='target')
        # Pre-trained ML model
        m = dice_ml.Model(model = self, backend=self.config.dice.backend, func=self.config.dice.func)
        # DiCE explanation instance
        exp = dice_ml.Dice(d,m)
        

        dice_exp = exp.generate_counterfactuals(X_test, total_CFs=total_CFs, desired_class=desired_class, features_to_vary = features_to_vary, **kwargs)

        dice_exp.visualize_as_dataframe()
        
        return dice_exp

    def init_shap_explainer(self) -> None:
        """Initialize a shap explainer using the trained model of RIAS
        """
        self.shap_explainer = shap.Explainer(self.predict_proba, masker=self.X, algorithm='permutation', seed = self.random_seed)
        
    def calculate_shap_values(self) -> None:
        """Calculate shapely values of self.X and saving them and their base_values
        """
        self.set_random_seed()
        self.shap_values = self.shap_explainer(self.X)
        self.base_values = self.shap_values.base_values.mean(0)
        
    def report_pred(self, sample: pd.Series, target: int = 0, save: bool = False, save_path: str = None) -> shap.plots._force.AdditiveForceVisualizer:
        """Report prediction with local explanations for a given sample.

        Args:
            sample (pd.Series): A sample to report its prediction.
            target (int, optional): A target label. Defaults to 0.
            save (bool, optional): A flag to save the result or not. Defaults to False.
            save_path (str, optional): A save path to save the result. If save is False, save_path is ignored. Defaults to None.

        Returns:
            shap.plots._force.AdditiveForceVisualizer: A force plot for a given sample.
        """
        self.set_random_seed()
        sample = self.check_input(sample)
        shap_value = self.shap_explainer(sample)
        
        plot = shap.force_plot(self.base_values[target], shap_value.values[0, :, target], sample, matplotlib=False, show=False)

        if save:
            if save_path is None:
                save_path = self.start_time + '.html'
            shap.save_html(save_path, plot)
            
        return plot
    
    def calculate_feature_importance(self) -> None:
        """Calculate the global feature importance using BorutaShap algorithm
        """
        self.feature_selector = BorutaShap(importance_measure='shap',
                                        classification=False if self.config.experiment.task == "regression" else True,
                                        )
        
        self.feature_selector.fit(X=self.X, y=self.y, n_trials=self.config.experiment.borutashap.n_trials, sample=False,
            	        train_or_test = 'test', normalize=self.config.experiment.borutashap.normalize,
		                verbose=self.config.experiment.borutashap.verbose, random_state = self.random_seed, stratify=self.y)
        
        self.feature_importances = pd.DataFrame(data={'Features':self.feature_selector.history_x.iloc[1:].columns.values,
        'Average Feature Importance':self.feature_selector.history_x.iloc[1:].mean(axis=0).values,
        'Standard Deviation Importance':self.feature_selector.history_x.iloc[1:].std(axis=0).values})
        
        decision_mapper = self.feature_selector.create_mapping_of_features_to_attribute(maps=['Tentative','Rejected','Accepted', 'Shadow'])
        self.feature_importances['Decision'] = self.feature_importances['Features'].map(decision_mapper)
        self.feature_importances = self.feature_importances.drop(index = [i for i in range(len(self.feature_importances) - 1, len(self.feature_importances) - 5, -1)], axis=0)
        
        self.feature_importances["Decision"] = self.feature_importances["Decision"].apply(lambda x : x if x != "Rejected" else "_Rejected")
        self.feature_importances = self.feature_importances.sort_values(by=["Decision", "Average Feature Importance"], ascending=[True, False])
        self.feature_importances["Decision"] = self.feature_importances["Decision"].apply(lambda x : x if x != "_Rejected" else "Rejected")
        
        # self.feature_importances = self.feature_importances.sort_values(by='Average Feature Importance',ascending=False)
        self.feature_importances.reset_index(drop=True, inplace=True)
        
    def report_feature_importance(self, report_path: str = 'feature_importance') -> None:
        """Report the calculated feature importance as csv format

        Args:
            report_path (str, optional): A saving path for the report. Defaults to 'feature_importance'.
        """
        
        assert self.feature_selector is not None, "Run calculate_feature_importance first"
        
        if not os.path.exists(report_path):
            os.makedirs(report_path, exist_ok=True)
            
        self.feature_selector.results_to_csv(f'{report_path}/{self.config.experiment.data_config}-{self.model_class.__name__}-{self.start_time}.csv')
    
        
        
        
    def save_model(self, save_path: str = 'checkpoints') -> str:
        """Save the model

        Args:
            save_path (str, optional): A saving path for the model. Defaults to 'checkpoints'.

        Returns:
            str: The saving path for the model.
        """
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        save_path = f"{save_path}/{self.config.experiment.data_config}-{self.model_class.__name__}-{self.start_time}"
        save_path = self.model.save_model(save_path)
        return save_path
    
    def load_model(self, model_path: str) -> None:
        """Load the model

        Args:
            model_path (str): A path where the model is saved.
        """

        self.model = self.get_model()
        self.model.load_model(model_path)

    def save_shap(self, shap_path: str = None) -> None:
        """Save the shap explainer

        Args:
            shap_path (str, optional): A saving path for the shap explainer. Defaults to None.
        """
        if shap_path == None:
            self.shap_path = f"./shap-{self.config.experiment.data_config}-{self.model_class.__name__}-{self.start_time}.pickle"
        else:
            self.shap_path = shap_path
            self.shap_explainer.save(open(self.shap_path, 'wb'))
            
        
    def load_shap(self, shap_path: str) -> None:
        """Load the shap explainer

        Args:
            shap_path (str): A path where the shap explainer is saved.
        """
        self.shap_explainer = shap.PermutationExplainer.load(open(shap_path, 'rb'))
        
    def save_rias(self, save_path: str = "./rias_checkpoints") -> None:
        """Save the rias

        Args:
            save_path (str, optional): A saving path for the rias. Defaults to "./rias_checkpoints".
        """
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
            
        if self.shap_explainer is not None:
            self.save_shap(f"{save_path}/shap-{self.config.experiment.data_config}-{self.model_class.__name__}-{self.start_time}.pickle")
            self.shap_explainer = None
        
        self.model_path = self.save_model(save_path)
        self.model = None

        pickle.dump(self, open(f"{save_path}/{self.config.experiment.data_config}-{self.model_class.__name__}-{self.start_time}.pickle", 'wb'))
        
        self.load_model(self.model_path)
        self.load_shap(self.shap_path)
    
    @staticmethod
    def load_rias(rias_path: str) -> RIAS:
        """Load the rias

        Args:
            path (str): A path where the rias is saved.

        Returns:
            RIAS: The loaded rias.
        """
        rias = pickle.load(open(rias_path, 'rb'))
        rias.load_model(rias.model_path)
        del rias.model_path

        if rias.shap_path is not None:
            rias.load_shap(rias.shap_path)
            del rias.shap_path
            
        rias.set_random_seed()
        return rias
    
    @staticmethod
    def prepare_rias(config: SimpleNamespace, model_class: Type[BaseModel], X: pd.DataFrame, y: np.array, continuous_cols: List[str], categorical_cols: List[str], calibrate: bool) -> RIAS:

        rias = RIAS(config = config, model_class=model_class, X=X, y = y, continuous_cols=continuous_cols, categorical_cols=categorical_cols, calibrate=calibrate)
        
        return rias