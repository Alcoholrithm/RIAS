from pytorch_tabular.models import FTTransformerConfig

from .pytorch_tabular_base import PytorchTabularBase
from types import SimpleNamespace
from typing import List, Dict, Any

class FTTransformer(PytorchTabularBase):
    
    def __init__(self, 
                **kwargs
        ) -> None:
        super().__init__(**kwargs)
        

    def get_model_config(self, 
                        hparams: Dict[str, Any]
        ) -> FTTransformerConfig:
        model_config = FTTransformerConfig(
                task="regression" if self.config.experiment.task == "regression" else "classification",
                learning_rate=hparams['learning_rate'], 
                seed = self.config.experiment.random_seed,
                input_embed_dim = hparams['input_embed_dim'],
                embedding_dropout=hparams['embedding_dropout'], 
                share_embedding=hparams['share_embedding'],
                num_heads=hparams['num_heads'],
                num_attn_blocks = hparams['num_attn_blocks'],
                transformer_activation = hparams['transformer_activation'],
                batch_norm_continuous_input = hparams['batch_norm_continuous_input'],
                metrics=[self.config.experiment.metric],
                metrics_params=[{'task' : self.config.experiment.task}]
            )
        return model_config