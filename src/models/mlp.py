from pytorch_tabular.models import CategoryEmbeddingModelConfig

from .pytorch_tabular_base import PytorchTabularBase
from types import SimpleNamespace
from typing import List, Dict, Any

class MLP(PytorchTabularBase):
    
    def __init__(self, 
                **kwargs
        ) -> None:
        super().__init__(**kwargs)
        

    def get_model_config(self, 
                        hparams: Dict[str, Any]
        ) -> CategoryEmbeddingModelConfig:

        model_config = CategoryEmbeddingModelConfig(
                task="regression" if self.config.experiment.task == "regression" else "classification",
                learning_rate=hparams['learning_rate'], 
                seed = self.config.experiment.random_seed,
                embedding_dropout = hparams['embedding_dropout'],
                layers = hparams['layers'],
                activation = hparams['activation'],
                metrics=[self.config.experiment.metric],
                metrics_params=[{'task' : self.config.experiment.task}]
            )
        return model_config