"""Utility functions for plkit"""
from functools import wraps
from io import StringIO

from rich.table import Table
from rich.console import Console
from pytorch_lightning import _logger as logger

from .exceptions import PlkitConfigException

def _check_config(config,
                  item,
                  how=lambda conf, key: key in conf,
                  msg="Configuration item {key} is required."):
    """Check configuration items"""
    checked = how(config, item)
    if not checked:
        raise PlkitConfigException(msg.format(key=item))

def _collapse_suggest_config(config):
    """Use the default value of OptunaSuggest for config items."""
    from .optuna import OptunaSuggest
    config = config.copy()
    collapsed = {key: val.default
                 for key, val in config.items()
                 if isinstance(val, OptunaSuggest)}
    config.update(collapsed)
    return config

def log_config(config, title='Configurations', items_per_row=2):
    """Log the configurations"""
    table = Table(title=title)
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
