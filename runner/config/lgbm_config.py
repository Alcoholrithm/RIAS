from .base_config import base_config
from copy import deepcopy

lgbm_config = deepcopy(base_config)

lgbm_config.model.model_class = "LGBMClf"

lgbm_config.model.fit_params["verbose"] = -1

lgbm_config.model.search_range = {
                                'num_leaves' : ['suggest_int', ['num_leaves', 300, 4000]],
                                'n_estimators' : ['suggest_int', ['n_estimators', 10, 3000]],
                                'learning_rate' : ['suggest_float', ['learning_rate',0, 1]],
                                'num_iterations' : ['suggest_int', ['num_iterations',100,2000]],
                                'max_depth' : ['suggest_int', ['max_depth', 3, 50]],
                                'scale_pos_weight' : ['suggest_int', ['scale_pos_weight', 1, 100]],
                            }