from .pytorch_tabular_config import pytorch_tabular_config
from copy import deepcopy

ft_config = deepcopy(pytorch_tabular_config)

ft_config.model.model_class = "FTTransformer"

ft_config.model.search_range = {
    'input_embed_dim' : ['suggest_categorical', ['input_embed_dim', [16,24,32,48]]],
    'embedding_dropout' : ['suggest_float', ['embedding_dropout', 0.05, 0.3]],
    'share_embedding' : ['suggest_categorical', ['share_embedding', [True, False]]],
    'num_heads' : ['suggest_categorical', ['num_heads', [1, 2, 4, 8]]],
    'num_attn_blocks' : ['suggest_int', ['num_attn_blocks', 2, 10]],
    'transformer_activation' : ['suggest_categorical', ['transformer_activation', ['GEGLU', 'ReGLU', 'SwiGLU']]],
    'batch_norm_continuous_input' : ['suggest_categorical', ['batch_norm_continuous_input', [True, False]]],
    'learning_rate' : ['suggest_float', ['learning_rate',0.0001, 0.05]],
}