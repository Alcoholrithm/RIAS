from types import SimpleNamespace
from typing import Dict, List, Any

base_config = SimpleNamespace()

base_config.model = SimpleNamespace()
base_config.model.hparams: Dict[str, Any] = None
base_config.model.search_range = None
base_config.model.model_path = None
base_config.model.fit_params = {}
base_config.model.additional_hparams = {}

base_config.experiment = SimpleNamespace()
base_config.experiment.random_seed = 42

base_config.experiment.task = None

base_config.experiment.n_jobs = 32

base_config.experiment.KFold = 5
base_config.experiment.valid_size = 0.2
base_config.experiment.metric = None
base_config.experiment.metric_params = None
base_config.experiment.report_metric = None

base_config.experiment.save_hparams = False

base_config.experiment.optuna = SimpleNamespace()
base_config.experiment.optuna.n_trials = 50
base_config.experiment.optuna.direction = None

base_config.experiment.ece_bins = 10
base_config.experiment.calibrator = None

base_config.experiment.early_stopping_patience = 30
base_config.experiment.fast_dev_run = False


base_config.dice = SimpleNamespace()
base_config.dice.backend = None
base_config.dice.func = None


base_config.experiment.borutashap = SimpleNamespace()
base_config.experiment.borutashap.n_trials = 100
base_config.experiment.borutashap.normalize = True
base_config.experiment.borutashap.verbose = True
base_config.experiment.borutashap.stratify = True
base_config.experiment.borutashap.kwargs = {}