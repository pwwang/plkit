# plkit

Deep learning boilerplate based on [pytorch-lightning][1]

## Installation
```
pip install plkit
```

## API documentation

https://pwwang.github.io/plkit/index.html

## Usage

### From pytorch-lightning's minimal example

```python
import os
import torch
from torchvision import transforms
from torchvision.datasets import MNIST
from plkit import Module, Data as PkData, run

class Data(PkData):

    def data_reader(self):
        minst = MNIST(self.sources, train=True,
                      download=True, transform=transforms.ToTensor())
        return {'train': (minst.data, minst.targets)}

class LitClassifier(Module):

    def __init__(self, config):
        super().__init__(config)
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1).float()))

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = self.loss_function(self(x), y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

if __name__ == '__main__':
    config = {
        'gpus': 1,
        'batch_size': 32,
        'max_epochs': 10,
        'data_sources': os.path.join(os.path.dirname(__file__), 'data'),
    }
    run(config, Data, LitClassifier)
```

## Features
- Compatible with `pytorch-lightning`
- Exposed terminal logger
- Even more abstracted boilerplace (What you only need to care is your data and model)
- Trainer from a dictionary (not only from an `ArgumentParser` or a `Namespace`)
- Abstraction of data manager
- Easy hyperparameter logging
- Running `test` automatically when `test_dataloader` is given
- Auto loss function and optimizer
- Builtin measurements (using `pytorch-lightning.metrics`)
- Optuna integration

## Exposed terminal logger
```python
from plkit import logger

# You logic

if __name__ == "__main__":
    logger.info('Fantastic starts ...')
    # ...
    logger.info('Pipeline done.')
```

## More abstracted boilerplate
```python
from plkit import Module, Data, run

# Data manager
class MyData(Data):

    def data_reader(self, sources):
        """Read your data from sources
        Return a list of data
        """
        # your logic

# Model
class MyModel(Module):
    # Your model definition

if __name__ == "__main__":
    config = {
        batch_size=32,
        # other configs
    }
    run(config, MyModel, MyData)
```

## Trainer from a configuration dictionary

Apart from `Trainer.from_argparse_args` to create a trainer, we also added a way to create a trainer by `Trainer.from_dict`, which enables possibilities to use other argument parser packages. For example:
```python
from pyparam import params
params.gpus = 1
params.gpus.desc = 'Number of GPUs to use.'

config = params._parse()

trainer = Trainer.from_dict(config)
# ...
```

Why not using `**config` to create a trainer?
```python
trainer = Trainer(**config)
```

This is because sometimes `config` will have config items other than the arguments that `Trainer` constructor needs, which will raise an error. `Trainer.from_dict` filters the items only the `Trainer` needs.

## Data manager

As it showed in the above examples, you don't need to concern about `Dataset` and `Dataloader` stuff. What you only need to care about is how to read the data from the sources. `plkit` will take care of the `Dataset` and `Dataloader` for you.

What you can fetch from `a, b, c, ... = batch` in `training`, `validation` and `test` steps depends on what you return from `data_reader`. For example, if you return `(data, labels)` from `data_reader`, then you are able to fetch them by `data, labels = batch` in the steps.

`plkit` can also split your data into `training`, `validation` and `test` parts, just by passing a ratio to `Data`: `data = Data(..., ratio=(.7, .2, .1))` (training: 70%, validation: 20%, test: 10%). Or in config: `config = {train_val_test_ratio: (.7, .2, .1)}`

If you don't specify a ratio, you will need to return dictionaries from `data_reader` with keys `train`, `val` and `test` for the data assigned to each part.

## hyperparameter logging
```python
from plkit import Module

# ...
class MyModel(Module):
    def __init__(self, config):
        super().__init__(config)
        # initialization
        # ...
        self.hparams = {
            # hyperparameters you want to log
        }

```

Keep in mind that, to enable this, you have to keep the default logger. Since we switched default logger from tensorboard logger to `HyperparamsSummaryTensorBoardLogger` implemented in `trainer.py` of `plkit`, whose idea was borrowed from [here][3].

To custom a logger, you have to subclass `HyperparamsSummaryTensorBoardLogger`, or just use it:
```python
from plkit.trainer import HyperparamsSummaryTensorBoardLogger

trainer = Trainer(logger=HyperparamsSummaryTensorBoardLogger(...), ...)
```

## Auto loss function and optimizer

A loss function will be initialized according to the `optim` and `loss` configurations.

`optim` supports `adam` and `sgd`, corresponding to `torch.optim.Adam` and `torch.optim.SGD`, respectively. You can specify `learning_rate` in the configuration

If `loss` is `auto`, `MSELoss` will be used for `num_classes==1` and `CrossEntropyLoss` otherwise. You can also specify the loss function by `loss=L1Loss()`.

## Builtin measurements

You can get some of the measurements directly now by `self.measure(logits, labels, method, **kwargs)`, which calls metrics implemented in `pytorch_lightning.metrics`.

- For `num_classes == 1` (regression), `mse`, `rmse`, `mae` and `rmsle` are available.
- Otherwise, `accuracy`, `precision`, `recall`, `f1_score` and `iou`.

For extra `kwargs`, check the [source code][4] (Haven't found them documented yet).

## Optuna integration

```python
config = {
    # default value: 512, using suggest_catigorical,
    # choosing one of 128, 256 and 512
    'hidden_size': OptunaSuggest(512, 'cat', [128, 256, 512]),
    # default value: 1, using suggest_int
    # choosing between 1 and 10
    'seed': OptunaSuggest(1, 'int', 1, 10),
    # other configurations
}
```

The default values don't master if you are running optuna:
```python
optuna = Optuna(on='val_loss', n_trials=100)
# just like plkit.run
optuna.run(config, Data, Model)
```

However, those default values will be used if you want opt optuna out, and we don't need to change anything from the `config`:
```python
plkit.run(config, Data, Model)
```


[1]: https://github.com/PyTorchLightning/pytorch-lightning
[2]: https://github.com/PyTorchLightning/pytorch-lightning/issues/1228
[3]: https://github.com/mRcSchwering/pytorch_lightning_test/blob/master/src/loggers.py
[4]: https://github.com/PyTorchLightning/pytorch-lightning/tree/master/pytorch_lightning/metrics


