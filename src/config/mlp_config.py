from .pytorch_tabular_config import pytorch_tabular_config
from copy import deepcopy

mlp_config = deepcopy(pytorch_tabular_config)

mlp_config.model.model_class = "MLP"

mlp_config.model.search_range = {
    'embedding_dropout' : ['suggest_float', ['embedding_dropout', 0.0, 0.2]],
    'layers' : ['suggest_categorical', ['layers', ['128-64-32', '256-128-64', '128-64-32-16', '256-128-64-32']]],
    'activation' : ['suggest_categorical', ['activation', ['ReLU', 'LeakyReLU']]],
    'learning_rate' : ['suggest_float', ['learning_rate',0.0001, 0.05]], 
}