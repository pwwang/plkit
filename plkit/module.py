"""The core base module class based on pytorch_lightning.LightningModule"""
import torch
from torch import nn
from pytorch_lightning import LightningModule
from pytorch_lightning.metrics.functional import regression, classification
from .exceptions import PlkitMeasurementException
from .utils import _collapse_suggest_config

def _check_logits_shape(logits, dim, dim_to_check=1):
    if logits.shape[dim_to_check] != dim:
        # checking is done in measurements
        raise PlkitMeasurementException(f"Logits require size of {dim} at "
                                        f"dimension {dim_to_check}")

class Module(LightningModule):
    """Base Module"""

    def __init__(self, config):
        super().__init__()

        config = _collapse_suggest_config(config)

        self.config = config
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

    def on_train_start(self):
        """Log hyperparameters to tensorboard"""
        if self.hparams:
            self.logger.log_hyperparams_metrics(self.hparams, {})

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
        """Do some measurements"""
        # regression
        if self.num_classes == 1:
            if method not in ('mse', 'rmse', 'mae', 'rmsle'):
                raise PlkitMeasurementException(
                    f"Method not supported for regression: {method}"
                )

            _check_logits_shape(logits, 1, 1)
            return getattr(regression, method)(logits.view(-1), labels.view(-1),
                                               **kwargs)

        # classification
        else:
            _check_logits_shape(logits, self.num_classes, 1)

            if method in ('accuracy', 'precision', 'recall', 'f1_score', 'iou'):
                return getattr(classification, method)(
                    logits,
                    labels.view(-1),
                    num_classes=self.num_classes,
                    **kwargs
                )
            if method == 'fbeta_score':
                if 'beta' not in kwargs:
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

    @property
    def device(self):
        """Get the device of the model."""
        return next(self.parameters()).device
