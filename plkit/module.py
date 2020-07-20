"""The core base module class based on pytorch_lightning.LightningModule"""
from functools import wraps
import torch
from torch import nn
from pytorch_lightning import LightningModule
from sklearn.metrics import roc_auc_score#, mean_squared_error
#from scipy.stats import spearmanr, pearsonr, kendalltau
from .exceptions import PlkitDataSizeException, PlkitMeasurementException

# ------ Measurements --------
# pytorch-lightning is implementing this
# this will be finally replaced with pytorch-lightning's implementation
def measure_accuracy(logits, labels):
    """Measuring accuracy for n-class classification"""
    predictions = torch.argmax(logits, dim=1).view(-1)
    truth = labels.view(-1)
    return torch.sum(predictions == truth).item() / (len(truth) * 1.0)

def measure_auc(logits, labels):
    """Measuring auc for 2-class classification"""
    # probabilities predict to 1
    probs = logits[:, 1]
    return roc_auc_score(labels.view(-1), probs)

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

class Module(LightningModule):
    HPARAMS_PLACEHOLDER = '__hparams_placeholder__'

    """Base Module"""
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

    def _check_logits(self, logits):
        """Check whether the logits is in right shape
        [batch_size x num_classes]"""
        shape = logits.shape
        if len(shape) != 2 or shape[1] != self.num_classes:
            raise PlkitDataSizeException('Excepting logits to be of shape '
                                         '[batch_size x num_classes]')

    def loss_function(self, logits, labels):
        """Calculate the loss"""
        # expect logits.shape == [batch_size x num_classes]
        self._check_logits(logits)
        return self._loss_func(logits, labels)

    def configure_optimizers(self):
        """Configure the optimizers"""
        if self.optim == 'adam':
            return torch.optim.Adam(self.parameters(),
                                    lr=self.config.learning_rate)

        # more to support

    def measure(self, logits, labels, method):
        """Do some measurements"""
        if method == 'accuracy':
            if self.num_classes < 2:
                raise PlkitMeasurementException(
                    'Cannot measure accuracy for regression.'
                )
            return measure_accuracy(logits, labels)

        if method == 'auc':
            if self.num_classes != 2:
                raise PlkitMeasurementException(
                    'AUC can only be calculated for binary classification.'
                )
            return measure_auc(logits, labels)
