"""The core base module class based on pytorch_lightning.LightningModule"""
import torch
from torch import nn
from pytorch_lightning import LightningModule
from pytorch_lightning.metrics.functional import regression, classification
from .exceptions import PlkitMeasurementException
from .utils import collapse_suggest_config, warning_to_logging

def _check_logits_shape(logits, dim, dim_to_check=1):
    """Check if the logits are in the right shape

    Args:
        logits (Tensor): The logits to check
        dim (int): The expected size
        dim_to_check (int): The dimension to check

    Raises:
        PlkitMeasurementException: when the size at dimension
            `dim_to_check` != `dim`
    """
    if logits.shape[dim_to_check] != dim:
        # checking is done in measurements
        raise PlkitMeasurementException(f"Logits require size of {dim} at "
                                        f"dimension {dim_to_check}")

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

        self.optim = config.get('optim', 'adam')
        self.num_classes = config.get('num_classes')
        # We may run test only without measurement.
        # if not self.num_classes:
        #     raise ValueError('We need `num_classes` from config or passed in '
        #                      'explictly to check final logits size.')

        loss = config.get('loss', 'auto')
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

    def measure(self, logits, labels, method, **kwargs):
        """Do some measurements with logits and labels

        Args:
            logits (Tensor): The logits from the model
                It's usually in the shape of [batch_size x num_classes]
            labels (Tensor): The labels of the batch
            method (str): The method for the metric
                For regression, supported methods are:
                `'mse', 'rmse', 'mae', 'rmsle'`
                For classification, supported methods are:
                `'accuracy', 'precision', 'recall', 'f1_score', 'iou'`
            **kwargs: Other arguments for the method.
                See pytorhc-lightning's doc for the metrics.
        """
        # See: https://github.com/PyTorchLightning/pytorch-lightning/issues/2768
        with warning_to_logging():
            # regression
            if self.num_classes == 1:
                if method not in ('mse', 'rmse', 'mae', 'rmsle'):
                    raise PlkitMeasurementException(
                        f"Method not supported for regression: {method}"
                    )

                _check_logits_shape(logits, 1, 1)
                return getattr(regression, method)(
                    logits.view(-1),
                    labels.view(-1),
                    **kwargs
                )

            # classification
            else:
                _check_logits_shape(logits, self.num_classes, 1)

                if method in ('accuracy', 'precision',
                              'recall', 'f1_score', 'iou'):
                    return getattr(classification, method)(
                        logits,
                        labels.view(-1),
                        num_classes=self.num_classes,
                        **kwargs
                    )
                if method == 'fbeta_score':
                    if 'beta' not in kwargs: # pragma: no cover
                        raise PlkitMeasurementException(
                            'fbeta_score requires a beta keyword argument.'
                        )
                    return classification.fbeta_score(logits, labels.view(-1),
                                                      **kwargs)

                if method in ('auroc', 'average_precision', 'dice_score'):
                    return getattr(classification, method)(
                        logits, labels.view(-1), **kwargs
                    )

                raise PlkitMeasurementException(
                    f"Method not supported for classification: {method}"
                )
