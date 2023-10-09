from .base_config import base_config
from copy import deepcopy

lgbm_config = deepcopy(base_config)

lgbm_config.model.model_class = "LGBM"

lgbm_config.model.additional_hparams["verbose"] = -1
lgbm_config.model.additional_hparams["n_jobs"] = lgbm_config.experiment.n_jobs
lgbm_config.model.additional_hparams["early_stopping_rounds"] = lgbm_config.experiment.early_stopping_patience

lgbm_config.model.search_range = {
                                'num_leaves' : ['suggest_int', ['num_leaves', 300, 4000]],
                                'n_estimators' : ['suggest_int', ['n_estimators', 10, 3000]],
                                'learning_rate' : ['suggest_float', ['learning_rate',0, 1]],
                                'max_depth' : ['suggest_int', ['max_depth', 3, 50]],
                                'scale_pos_weight' : ['suggest_int', ['scale_pos_weight', 1, 100]],
                            }