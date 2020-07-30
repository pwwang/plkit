"""Wrapup the Trainer class"""
import os
import inspect
from pytorch_lightning import Trainer as PlTrainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers.base import rank_zero_only
from torch.utils.tensorboard.summary import hparams
from .utils import _collapse_suggest_config

# in order to solve logging hyperparamters
# See: https://github.com/PyTorchLightning/pytorch-lightning/issues/1228
# pylint: disable=line-too-long
# And: https://github.com/mRcSchwering/pytorch_lightning_test/blob/master/src/loggers.py
# pylint: enable=line-too-long
class HyperparamsSummaryTensorBoardLogger(TensorBoardLogger):
    # pylint: disable=line-too-long
    """
    This logger follows this idea:
    https://github.com/PyTorchLightning/pytorch-lightning/issues/1228#issuecomment-620558981
    For having metrics attched to hparams I am writing a summary
    with an initial metric.
    This metric will later on be updated by `add_scalar` calls,
    which work out-of-the-box in pytorch lightning using the `log` key.
    To make sure, that the hparams summary has its metric, I need to
    silence the usual `log_hyperparams` call again.
    Otherwise this would be called without metrics at the start of the training.
    To use this logger you need to log a metric with it at
    the beginning of the training.
    Then update this metric/key during the training.

    Examples:
        def on_train_start(self):
            self.logger.log_hyperparams_metrics(
                self.hparams, {'best/val-loss': self.best_val_loss}
            )

        def validation_epoch_end(self, val_steps: List[dict]) -> dict:
            ...
            log = {'best/val-loss': best_loss}
            return {'val_loss': current_loss, 'log': log}
    """
    # pylint: enable=line-too-long
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tags = {}

    def log_hyperparams(self, params, metrics=None):
        """Bypass the function"""

    @rank_zero_only
    def log_hyperparams_metrics(self, params, metrics=None):
        """Log the hyperparameters together with some metrics

        The metrics will be used as key to match where the hyperparameters
        should be logged
        """
        params = self._convert_params(params)
        params = self._flatten_dict(params)
        sanitized_params = self._sanitize_params(params)
        if metrics is None:
            metrics = {}
        exp, ssi, sei = hparams(sanitized_params, metrics)
        writer = self.experiment._get_file_writer()
        writer.add_summary(exp)
        writer.add_summary(ssi)
        writer.add_summary(sei)

        # some alternative should be added
        self.tags.update(sanitized_params)

class Trainer(PlTrainer):
    """The Trainner class"""

    @classmethod
    def from_config(cls, config, **kwargs):
        """
        Create an instance from CLI arguments.
        Args:
            config: The parser or namespace to take arguments from.
                Only known arguments will be
                parsed and passed to the :class:`Trainer`.
            **kwargs: Additional keyword arguments that may override ones in
                the parser or namespace.
                These must be valid Trainer arguments.
        Example:
            >>> config = {'my_custom_arg': 'something'}
            >>> trainer = Trainer.from_dict(config, logger=False)
        """
        # we only want to pass in valid Trainer args,
        # the rest may be user specific
        valid_kwargs = inspect.signature(PlTrainer.__init__).parameters
        trainer_kwargs = dict((name, config[name])
                              for name in valid_kwargs if name in config)
        trainer_kwargs = _collapse_suggest_config(trainer_kwargs)
        trainer_kwargs.update(**kwargs)

        return cls(**trainer_kwargs)

    from_dict = from_config

    def __init__(self, *args, **kwargs):
        self.data = kwargs.pop('data', None)
        # let's see if logger was specified, otherwise we default it to
        # HyperparamsSummaryTensorBoardLogger
        kwargs.setdefault(
            'logger', HyperparamsSummaryTensorBoardLogger(
                save_dir=kwargs.get('default_root_dir', os.getcwd()),
                name='plkit_logs'
            )
        )
        super().__init__(*args, **kwargs)

    def fit(self, model, train_dataloader=None, val_dataloaders=None):
        if not train_dataloader and self.data is not None:
            train_dataloader = self.data.train_dataloader
        if not val_dataloaders and self.data is not None:
            val_dataloaders = self.data.val_dataloader
        super().fit(model, train_dataloader, val_dataloaders)

    def test(self, model=None, test_dataloaders=None,
             ckpt_path='best'):
        if not test_dataloaders and self.data is not None:
            test_dataloaders = self.data.test_dataloader

        if test_dataloaders is not None:
            super().test(model, test_dataloaders, ckpt_path)
