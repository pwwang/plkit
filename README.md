# plkit

Deep learning boilerplate based on [pytorch-lightning][1]

## Installation
```
pip install plkit
```

## Usage

### From pytorch-lightning's minimal example

```python
from plkit import Module, Data, Trainner

class MINISTData(Data):

    def data_reader(self, sources):
        minst = MNIST(sources, train=True, download=True,
                      transform=transforms.ToTensor())
        return {'train': (minst.data, minst.targets)}

class LitClassifier(Module):

    def __init__(self, config, num_classes, optim='adam', loss='auto'):
        super().__init__(config, num_classes, optim, loss)
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = self.loss_function(self(x), y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    # we don't need configure_optimizers anymore

config = {
    'batch_size': 32,
    'gpus': 8,
    'precision': 16,
    'num_classes': 2,
    # other options to initial a pytorch-lightning Trainer
}

def main():
    data = MINISTData(os.getcwd(), config['batch_size'])
    model = LitClassifier(config)
    trainer = Trainer.from_dict(config, data=data)
    trainer.fit(model)

if __name__ == '__main__':
    main()
```

Or even simpler for `main`:
```python
from plkit import run

# Data class and model class definition
# ...

def main():
    run(config, LitClassifier, MINISTData)

```

## Features
- Exposed terminal logger
- Even more abstracted boilerplace (What you only need to care is your data and model)
- Trainer from a dictionary (not only from an `ArgumentParser` or a `Namespace`)
- Abstraction of data manager
- Easy hyperparameter logging
- Running `test` automatically when `test_dataloader` is given
- Auto loss function and optimizer
- Builtin measurements (using `pytorch-lightning.metrics`)

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

Usually, you need to return a list of `data` (features) as well as the corresponding `targets` (labels) from your `data_reader` function. In such a case, you will not be able to fetch the names of the samples by `data, targets = batch` in the `training_step`, `validation_step` and `test_setp`.

To do that, you will need to provide `with_name=True` argument when instantialize `Data`: `data = Data(..., with_name=True)`, or in your config:
```python
config = {data_with_name=True, ...}
data = Data(sources, config)
```
and then, return dictionaries of data and targets with keys as the names. Finally, you are able to fetch the names in any of the steps by `data, targets, names = batch`.

`plkit` can also split your data into `training`, `validation` and `test` parts, just by passing a ratio to `Data`: `data = Data(..., ratio=(.7, .2, .1))` (training: 70%, validation: 20%, test: 10%). Or in config: `config = {train_val_test_ratio: (.7, .2, .1)}`

If you don't specify a ratio, you will need to return dictionaries from `data_reader` with keys `train`, `val` and `test` for the data assigned to each part.

## hyperparameter logging

To log hyperparameters using raw `pytorch-lightning` will be painful. See [#1228][2].

Here instead of all those tweakings, we only need to use a decorator:
```python
from plkit import log_hparams, Module

# ...
class MyModel(Module):

    @log_hparams
    def validation_step(self, batch, batch_idx):
        # do you logic
        # return the logs as what you did with pytorch-lightning
```

Keep in mind that, to enable this, you have to keep the default logger. Since we switched default logger from tensorboard logger to `HyperparamsSummaryTensorBoardLogger` implemented in `trainer.py` of `plkit`, whose idea was borrowed from [here][3].

To custom a logger, you have to subclass `HyperparamsSummaryTensorBoardLogger`, or just use it:
```python
from plkit.trainer import HyperparamsSummaryTensorBoardLogger

trainer = Trainer(logger=HyperparamsSummaryTensorBoardLogger(...), ...)
```

The only side effect of this is to have a `__hparams_placeholder__` metric to be logged, which is used to be inserted to identify the metrics logs together with the hyperparameters.

## Auto loss function and optimizer

A loss function will be initialized according to the `optim` and `loss` configurations.

`optim` supports `adam` and `sgd`, corresponding to `torch.optim.Adam` and `torch.optim.SGD`, respectively. You can specify `learning_rate` in the configuration

If `loss` is `auto`, `MSELoss` will be used for `num_classes==1` and `CrossEntropyLoss` otherwise. You can also specify the loss function by `loss=L1Loss()`.

## Builtin measurements

You can get some of the measurements directly now by `self.measure(logits, labels, method, **kwargs)`, which calls metrics implemented in `pytorch_lightning.metrics`.

- For `num_classes == 1` (regression), `mse`, `rmse`, `mae` and `rmsle` are available.
- Otherwise, `accuracy`, `precision`, `recall`, `f1_score` and `iou`.

For extra `kwargs`, check the [source code][4] (Haven't found them documented yet).




[1]: https://github.com/PyTorchLightning/pytorch-lightning
[2]: https://github.com/PyTorchLightning/pytorch-lightning/issues/1228
[3]: https://github.com/mRcSchwering/pytorch_lightning_test/blob/master/src/loggers.py
[4]: https://github.com/PyTorchLightning/pytorch-lightning/tree/master/pytorch_lightning/metrics


