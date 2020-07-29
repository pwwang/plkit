"""Even higher level wrapper based on pytorch-lightning"""
from functools import wraps
from io import StringIO
# expose terminal logger
# pylint: disable=unused-import
from pytorch_lightning import _logger as logger
from rich.table import Table
from rich.console import Console
from .data import Data
from .module import Module
from .trainer import Trainer

__version__ = "0.0.4"

def log_config(config, items_per_row=2):
    """Log the configurations"""
    table = Table(title="Configurations")
    items = list(config.items())
    for i in range(items_per_row):
        table.add_column("Item")
        table.add_column("Value")

    for i in range(0, len(items), items_per_row):
        row_items = []
        for x in range(items_per_row):
            try:
                row_items.append(items[i + x][0])
                row_items.append(repr(items[i + x][1]))
            except IndexError:
                row_items.append('')
                row_items.append('')

        table.add_row(*row_items)

    console = Console(file=StringIO(), markup=False)
    console.print(table)
    for line in console.file.getvalue().splitlines():
        logger.info(line)

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

def run(config: dict, model_class: callable, data_class: callable):
    """Run the pipeline by give configuration, model_class and data_class

    Args:
        config (dict): A dictionary of configuration, must have following items:
            - sources: The sources to read data from
            - batch_size: The batch size
            - num_classes: The number of classes for classification
                1 means regression
        model_class (class): The model class subclassed from `Module`
        data_class (class): The data class subclassed from `Data`
    """
    # check config
    if 'sources' not in config:
        raise ValueError('We need `sources` from `config` to read data from.')
    if 'batch_size' not in config:
        raise ValueError('We need `batch_size` from `config`.')
    if 'num_classes' not in config:
        raise ValueError('We need `num_classes` from `config`.')

    data = data_class(config.sources, config)
    model = model_class(config)
    trainer = Trainer.from_dict(config, data=data)
    trainer.fit(model)
    trainer.test()
