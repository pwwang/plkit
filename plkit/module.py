"""The core base module class based on pytorch_lightning.LightningModule"""
from functools import wraps
import torch
from torch import nn
from pytorch_lightning import LightningModule
from pytorch_lightning.metrics.functional import regression, classification
#from scipy.stats import spearmanr, pearsonr, kendalltau
from .exceptions import PlkitMeasurementException

def log_hparams(func):
    """A decorator for training_step, validation_epoch_end, etc to
    log hyperparameters"""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        ret = func(self, *args, **kwargs)
        if 'log' in ret and self.hparams:
            ret['log'][self.__class__.HPARAMS_PLACEHOLDER] = 0
        return ret

    return wrapper

def _check_logits_shape(logits, dim, dim_to_check=1):
    if logits.shape[dim_to_check] != dim:
        # checking is done in measurements
        raise PlkitMeasurementException(f"Logits require size of {dim} at "
                                        f"dimension {dim_to_check}")

class Module(LightningModule):
    """Base Module"""

    HPARAMS_PLACEHOLDER = '__hparams_placeholder__'

    def __init__(self, config, num_classes=None, optim=None, loss=None):
        super().__init__()
        self.config = config
        self.optim = optim or config.get('optim', 'adam')
        self.num_classes = num_classes or config.get('num_classes')
        if not self.num_classes:
            raise ValueError('We need `num_classes` from config or passed in '
                             'explictly to check final logits size.')

        loss = loss or config.get('loss', 'auto')
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
        if self.hparams:
            self.logger.log_hyperparams_metrics(
                self.hparams,
                {Module.HPARAMS_PLACEHOLDER: 0}
            )

    def loss_function(self, logits, labels):
        """Calculate the loss"""
        return self._loss_func(logits, labels)

    def configure_optimizers(self):
        """Configure the optimizers"""
        if self.optim == 'adam':
            return torch.optim.Adam(self.parameters(),
                                    lr=self.config.learning_rate)

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
