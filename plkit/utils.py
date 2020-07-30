"""Utility functions for plkit"""
import warnings
import logging
from io import StringIO
from contextlib import contextmanager

from rich.table import Table
from rich.console import Console
from pytorch_lightning import _logger as logger

from .exceptions import PlkitConfigException

logger.handlers[0].setFormatter(logging.Formatter(
    '[%(levelname).1s %(asctime)s] %(message)s'
))

def _check_config(config,
                  item,
                  how=lambda conf, key: key in conf,
                  msg="Configuration item {key} is required."):
    """Check configuration items

    Args:
        config (dict): The configuration dictionary
        item (str): The configuration key to check
        how (callable): How to check. Return False to fail the check.
        msg (str): The message to show in the exception.
            `{key}` is available to refer to the key checked.

    Raises:
        PlkitConfigException: When the check fails
    """
    checked = how(config, item)
    if not checked:
        raise PlkitConfigException(msg.format(key=item))

def _collapse_suggest_config(config):
    """Use the default value of OptunaSuggest for config items.
    So that the configs can be used in the case that optuna is opted out.

    Args:
        config (dict): The configuration dictionary

    Returns:
        dict: The collapsed configuration
    """
    from .optuna import OptunaSuggest
    config = config.copy()
    collapsed = {key: val.default
                 for key, val in config.items()
                 if isinstance(val, OptunaSuggest)}
    config.update(collapsed)
    return config

@contextmanager
def _suppress_warnings(warning_type):
    """Suppress warning of a certain type

    Args:
        warning_type: The class of the warning to suppress

    Yield:
        The context
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=warning_type)
        yield

def log_config(config, title='Configurations', items_per_row=2):
    """Log the configurations in a table in terminal

    Args:
        config (dict): The configuration dictionary
        title (str): The title of the table
        items_per_row (int): The number of items to print per row
    """
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
    logger.info('')
    for line in console.file.getvalue().splitlines():
        logger.info(line)
