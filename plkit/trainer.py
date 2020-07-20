"""Wrapup the Trainer class"""
import inspect
from pytorch_lightning import Trainer as PlTrainer

class Trainer(PlTrainer):
    """The Trainner class"""

    @classmethod
    def from_dict(cls, config, **kwargs):
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
        trainer_kwargs.update(**kwargs)

        return cls(**trainer_kwargs)

    def __init__(self, *args, **kwargs):
        self.data = kwargs.pop('data', None)
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
