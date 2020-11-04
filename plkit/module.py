"""The core base module class based on pytorch_lightning.LightningModule"""
import torch
from torch import nn
from pytorch_lightning import LightningModule
from .utils import collapse_suggest_config

class Module(LightningModule): # pylint: disable=too-many-ancestors
    """The Module class

    `on_epoch_end` is added to print a newline to keep the progress bar and the
    stats on it for each epoch. If you don't want this, just overwrite it with:
        >>> def on_epoch_end(self):
        >>>     pass

    If you have other stuff to do in `on_epoch_end`, make sure to you call:
        >>> super().on_epoch_end()

    You may or may not need to write `loss_function`, as it will be inferred
    from config item `loss` and `num_classes`. Basically, `MSELoss` will be
    used for regression and `CrossEntropyLoss` for classification.

    `measure` added for convinience to get some metrics between logits
    and targets.

    Args:
        config: The configuration dictionary

    Attributes:
        Apart from attributes of `LightningModule`, following attributes added:
        config: The configs
        optim: The optimizer name. currently only `adam` and `sgd`
            are supported. With this, of course you can, but you don't need to
            write `configure_optimizers`.
        num_classes: Number of classes to predict. 1 for regression
        _loss_func: The loss function
    """

    def __init__(self, config):
        super().__init__()
        self.config = collapse_suggest_config(config)
        self.optim = self.config.get('optim', 'adam')
        self.num_classes = self.config.get('num_classes')
        # We may run test only without measurement.
        # if not self.num_classes:
        #     raise ValueError('We need `num_classes` from config or passed in '
        #                      'explictly to check final logits size.')

        loss = self.config.get('loss', 'auto')
        if loss == 'auto':
            if self.num_classes == 1: # regression
                self._loss_func = nn.MSELoss()
            else:
                self._loss_func = nn.CrossEntropyLoss()
        else:
            self._loss_func = loss

        # the hyperparameters to be logged to tensorboard
        self.hparams = {}

    def on_epoch_end(self):
        """Keep the epoch progress bar
        This is not documented but working."""
        print()

    def loss_function(self, logits, labels):
        """Calculate the loss"""
        return self._loss_func(logits, labels)

    # pylint: disable=inconsistent-return-statements
    def configure_optimizers(self):
        """Configure the optimizers"""
        if self.optim == 'adam':
            return torch.optim.Adam(self.parameters(),
                                    lr=self.config.get('learning_rate', 1e-3))
        if self.optim == 'sgd':
            return torch.optim.SGD(self.parameters(),
                                   lr=self.config.get('learning_rate', 1e-3),
                                   momentum=self.config.get('momentum', .9))
        # more to support
