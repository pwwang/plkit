"""Wrapper of the Trainer class"""
import inspect
from pytorch_lightning import Trainer as PlTrainer
from pytorch_lightning.callbacks.progress import (
    ProgressBar as PlProgressBar,
    ProgressBarBase
)
from .utils import collapse_suggest_config, warning_to_logging

class ProgressBar(PlProgressBar):
    """Align the Epoch in progress bar"""
    def on_epoch_start(self, trainer, pl_module):
        """Try to align the epoch number"""
        super().on_epoch_start(trainer, pl_module)

        if self.max_epochs:
            nchar = len(str(self.max_epochs))
            self.main_progress_bar.set_description(
                f'Epoch {str(trainer.current_epoch).rjust(nchar)}'
            )

class Trainer(PlTrainer): # pylint: disable=too-many-ancestors
    """The Trainner class

    `from_config` (aka `from_dict`) added as classmethod to instantiate trainer
    from configuration dictionaries.
    """
    # pylint: disable=signature-differs

    @classmethod
    def from_config(cls, config, **kwargs):
        """Create an instance from CLI arguments.

        Examples:
        >>> config = {'my_custom_arg': 'something'}
        >>> trainer = Trainer.from_dict(config, logger=False)

        Args:
            config: The parser or namespace to take arguments from.
                Only known arguments will be
                parsed and passed to the :class:`Trainer`.
            **kwargs: Additional keyword arguments that may override ones in
                the parser or namespace.
                These must be valid Trainer arguments.
        """

        # we only want to pass in valid Trainer args,
        # the rest may be user specific
        valid_kwargs = inspect.signature(PlTrainer.__init__).parameters
        trainer_kwargs = dict((name, config[name])
                              for name in valid_kwargs if name in config)
        trainer_kwargs.update(**kwargs)
        trainer_kwargs = collapse_suggest_config(trainer_kwargs)

        return cls(**trainer_kwargs)

    from_dict = from_config

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('callbacks', [])
        if not any(isinstance(callback, ProgressBarBase)
                   for callback in kwargs['callbacks']):
            pbar_kwargs = {
                ('refresh_rate' if key == 'progress_bar_refresh_rate'
                 else key) : val
                for key, val in kwargs.items()
                if key in ('process_position', 'progress_bar_refresh_rate')
            }

            pbar = ProgressBar(**pbar_kwargs)
            pbar.max_epochs = kwargs.get('max_epochs')

            kwargs['callbacks'].append(pbar)

        with warning_to_logging():
            super().__init__(*args, **kwargs)

    @property
    def progress_bar_dict(self) -> dict:
        """Format progress bar metrics. """
        metrics = super().progress_bar_dict
        metrics = {
            key: '%.3f' % val if isinstance(val, float) else val
            for key, val in metrics.items()
        }
        return metrics


    def fit(self, *args, **kwargs):
        """Train and validate the model"""
        with warning_to_logging():
            super().fit(*args, **kwargs)

    def test(self, *args, **kwargs):
        """Test the model"""
        with warning_to_logging():
            super().test(*args, **kwargs)
