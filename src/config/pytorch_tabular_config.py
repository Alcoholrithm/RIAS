from .base_config import base_config
from copy import deepcopy

pytorch_tabular_config = deepcopy(base_config)

pytorch_tabular_config.model.gpus = [1]
pytorch_tabular_config.model.auto_select_gpus = False
pytorch_tabular_config.model.use_balanced_sampler = True
pytorch_tabular_config.model.use_weighted_loss = True
pytorch_tabular_config.model.mu = 0.1

pytorch_tabular_config.model.out_dim = None

pytorch_tabular_config.model.max_epochs = 1000
pytorch_tabular_config.model.batch_size = 256

pytorch_tabular_config.model.optimizer = "AdamW"
pytorch_tabular_config.model.optimizer_params={
                'weight_decay' : 0.1, 
            }

pytorch_tabular_config.model.lr_scheduler='ReduceLROnPlateau'
pytorch_tabular_config.model.lr_scheduler_params={}