import argparse
import os
import yaml
import importlib
from types import SimpleNamespace
from typing import Tuple, Dict, Any, Type, List

from src.config import __all__ as config_list

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import random

from src.rias import RIAS
from src.misc.eval_metric import EvalMetric
from src.models import BaseModel

from sklearn.metrics import f1_score, recall_score, accuracy_score, confusion_matrix, accuracy_score
from sklearn.metrics import roc_auc_score, recall_score, average_precision_score

from kamir import KamirDataModule

def main():
    parser = argparse.ArgumentParser(add_help=True)

    parser.add_argument('--config', type=str, choices=config_list)
    parser.add_argument('--data_config', type=str, default='6M_mortality', choices=[file.split('.')[0] for file in os.listdir('./data_config')] )

    parser.add_argument('--test_model', action='store_true')
    parser.add_argument('--test_size', type=str, default = 0.2, help='test set size')
    
    parser.add_argument('--calibrator', type=str, default=None, help="Calibration method for reliable confidence")
    parser.add_argument('--random_seed', type=int, default=0, help="A random seed for the experiment")

    parser.add_argument('--report_feature_importance', action='store_true')
    parser.add_argument('--report_rfe', action='store_true')
    
    parser.add_argument('--save_hparams', action="store_true")
    parser.add_argument('--save_data', action='store_true')
    parser.add_argument('--save_test_data', action='store_true')
    
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--model_path', type=str, default=None)
    
    parser.add_argument('--save_rias', action='store_true')
    parser.add_argument('--load_rias', default=None, type=str)
    parser.add_argument('--gpus', nargs='+', default=None, type = int)
    
    parser.add_argument('--use_dice', action="store_true")
    parser.add_argument('--dice_backend', type=str, choices=["TF2", "PTY", "sklearn"], default="sklearn")
    parser.add_argument('--dice_func', type=str, default=None)
    parser.add_argument('--dice_desired', type=int, default = 0)
    
    parser.add_argument('--use_lime', action="store_true")
    parser.add_argument('--use_shap', action='store_true')
    # parser.add_argument('--lime_class_names', type=List[str], default=None)
    
    parser.add_argument('--hparams', type=str, default = None, help="The location of the cached optimal hyperparameters")
    parser.add_argument('--n_trials', type=int, default = None, help="n_trials of optuna")
    parser.add_argument('--KFold', type=int, default = None)
    parser.add_argument('--fast_dev_run', action="store_true", help="Activate fast dev run")
    
    args = parser.parse_args()

    assert args.config != 'base_config', "Cannot use base config"

    data, label, X_test, y_test, continuous_cols, categorical_cols, data_config = prepare_data(args)
    
    if args.save_test_data:
        pickle.dump((X_test, y_test),open(f'{args.data_config}_test_data.pickle', 'wb'))
        
    config = prepare_config(args, data_config)

    if args.load_rias is None:
        rias = prepare_rias(config, data, label, continuous_cols, categorical_cols, True if config.experiment.calibrator is not None else False)
        if args.model_path is None:
            rias.train()
        else:
            rias.config.model.model_path = args.model_path
            rias.load_model()
    else:
        rias = RIAS.load_rias(args.load_rias)
    
        
    if args.save_model:
        rias.save_model()
        
    if config.experiment.calibrator is not None and args.model_path is not None:
        rias.init_calibrator()
        
    rias.test(X_test, y_test, KamirEvalMetric())
    
    if args.use_shap:
        if args.load_rias is None:
            rias.init_shap_explainer()
            rias.init_shap_base_values()
        # rias.report_pred(X_test.iloc[0], 1, save=True)
        # rias.report_pred(X_test.iloc[random.randint(0, len(X_test))], 1, save=True)
        rias.report_pred(X_test[(y_test == 1)].iloc[15], 1, save=True)
    
    if args.use_dice:
        rias.dice(X_test[(y_test == 1)].iloc[10])

    if args.use_lime:
        rias.lime(X_test.iloc[random.randint(0, len(X_test))].values)
    
    if args.report_feature_importance:
        rias.calculate_feature_importance()
        rias.report_feature_importance()
    
    if args.report_rfe:
        rias.report_recursive_feature_elimination(X_test, y_test, KamirEvalMetric())
        
    if args.save_rias:
        rias.save_rias(f'./rias_checkpoints/{args.random_seed}')


def prepare_data(args: argparse.ArgumentParser) -> Tuple[pd.DataFrame, np.array, pd.DataFrame, np.array]:
    with open("data_config/" + args.data_config + ".yaml", encoding='UTF-8') as f:
        data_config = yaml.load(f, Loader=yaml.FullLoader)
        data_config = SimpleNamespace(**data_config)

    # datalib = importlib.import_module('src.data_utils')
    datamodule = KamirDataModule(args.data_config, data_config)
    
    data, label, continuous_cols, categorical_cols = datamodule.prepare_data(args.save_data)

    train_idx, test_idx, _, _ = train_test_split(np.arange(len(label)).reshape((-1, 1)), label, test_size=args.test_size, random_state=args.random_seed, stratify=label)
    train_idx, test_idx = train_idx.ravel(), test_idx.ravel()

    X_test, y_test = data.iloc[test_idx], label[test_idx]
    data, label = data.iloc[train_idx], label[train_idx]

    return data, label, X_test, y_test, continuous_cols, categorical_cols, data_config

def prepare_config(args: argparse.ArgumentParser, data_config: Dict[str, Any]) -> SimpleNamespace:
    configlib = importlib.import_module('src.config')
    config = getattr(configlib, args.config)

    if args.hparams is not None:
        with open(args.hparams, 'rb') as f:
            hparams = pickle.load(f)
        config.model.hparams = hparams
    
    config.experiment.save_hparams = args.save_hparams
    
    config.experiment.fast_dev_run = args.fast_dev_run
    config.experiment.metric = data_config.metric
    config.experiment.metric_params = data_config.metric_params
    config.experiment.data_config = args.data_config
    config.experiment.optuna.direction = 'maximize'
    config.experiment.random_seed = args.random_seed
    config.experiment.task = "binary"
    config.experiment.KFold = args.KFold if args.KFold is not None else config.experiment.KFold
    
    if hasattr(config.model, 'gpus'):
        config.model.gpus = args.gpus if args.gpus is not None else config.model.gpus
    config.model.hparams = args.hparams
    
    config.experiment.optuna.n_trials = args.n_trials if args.n_trials is not None else config.experiment.optuna.n_trials
    
    config.experiment.calibrator = args.calibrator
    
    config.dice.backend = args.dice_backend
    config.dice.desired_class = args.dice_desired
    
    config.lime.class_names = ["alive", "dead"]
    config.lime.file = "temp.html"
    return config

def prepare_rias(config: SimpleNamespace, X: pd.DataFrame, y: np.array, continuous_cols: List[str], categorical_cols: List[str], calibrate: bool) -> RIAS:
    modellib = importlib.import_module('src.models')
    model_class = getattr(modellib, config.model.model_class)

    rias = RIAS(config = config, model_class=model_class, X=X, y = y, continuous_cols=continuous_cols, categorical_cols=categorical_cols, calibrate=calibrate)
    
    return rias

class KamirEvalMetric(EvalMetric):
    def eval(self, model: Type[BaseModel], X_test: pd.DataFrame, y_test: np.array) -> Dict[str, float]:
        preds_proba = model.predict_proba(X_test)
        preds = preds_proba.argmax(1)
        
        f1 = f1_score(y_test, preds)
        roc = roc_auc_score(y_test, preds_proba[:, 1])
        specificity = recall_score(np.logical_not(y_test) , np.logical_not(preds))
        sensitivity = recall_score(y_test, preds)
        accuracy = accuracy_score(y_test, preds)
        pr_auc = average_precision_score(y_test, preds_proba[:, 1])
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        ppv = tp / (tp + fp)
        npv = tn / (tn + fn)
        
        print("########## Evaluation Results for given test data ##########\n")
        print("F1 Score: %.4f" % f1)
        print("ROC AUC Score: %.4f" % roc)
        print("Specificity Score: %.4f" % specificity)
        print("Sensitivity Score: %.4f" % sensitivity)
        print("Accuracy Score: %.4f" % accuracy)
        print("Precision Recall AUC Score: %.4f" % pr_auc)
        print("PPV Score: %.4f" % ppv)
        print("NPV Score: %.4f" % npv)
        print()
        
        return {
            "F1 Score" : f1,
            "ROC AUC Score" : roc,
            "Specificity Score" : specificity,
            "Sensitivity Score" : sensitivity,
            "Accuracy Score" : accuracy,
            "Precision Recall AUC Score" : pr_auc,
            "PPV Score" : ppv,
            "NPV Score" : npv
        }
        
if __name__ == "__main__":
    main()
