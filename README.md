# How to install

```sh
pip install -r requirements.txt
```

# How to use RIAS for a specific dataset and user-defined model

### 1. Define a DataModule Class that inherit DataModule class.
```python
from src.data_utils import DataModule

class MyDataModule(DataModule):
    pass
```

### 2. Define a ModelModule Class that inherit BaseModel or use a predefined ModelModule.

```python
from src.models import BaseModel

class MyModel(BaseModel):
    pass
```

or 

```python
from src.models import XGB
```

### 3. Complete the configuration settings for the experiment. Load the base_config and define the missing options. If you use predefined ModelModule, load corresponding config in the src.config.

```python
from src.config import base_config as config

config.experiment.random_seed = 0
...
```

or

```python
from src.config import xgb_config as config

config.experiment.random_seed = 0
...
```
### 4. Define the evaluation metric for the given dataset.

```python
from src.misc.eval_metric import EvalMetric

class MyEvalMetric(EvalMetric):
    pass
```

### 5. Run RIAS

```python
# Assume we define MyDataModule, MyModel and config
from src.rias import RIAS
from src.models import XGB

### Prepare the data
datamodule = MyDataModule()
data, label, continuous_cols, categorical_cols = datamodule.prepare_data()

test_size = 0.2

train_idx, test_idx, _, _ = train_test_split(np.arange(len(label)).reshape((-1, 1)), label, test_size=test_size, random_state=config.experiment.random_seed, stratify=label)
train_idx, test_idx = train_idx.ravel(), test_idx.ravel()

X_test, y_test = data.iloc[test_idx], label[test_idx]
data, label = data.iloc[train_idx], label[train_idx]


### Run RIAS
rias = RIAS.prepare_rias(config, MyModel, data, label, continuous_cols, categorical_cols, True)

rias.train()

rias.init_calibrator()
rias.test(X_test, y_test, DiabetesEvalMetric())
```

## See the example.ipynb for detail example