"""Utility functions for plkit"""
from typing import Iterable, List, Optional, Tuple, Union
import sys
import logging
import warnings
from io import StringIO
from contextlib import contextmanager

from rich.table import Table
from rich.console import Console
from rich.logging import RichHandler
from diot import FrozenDiot
from pytorch_lightning import seed_everything, _logger as logger

from .exceptions import PlkitConfigException

RatioType = Union[int, float]

del logger.handlers[:]
logger.addHandler(RichHandler(show_path=False))

logging.getLogger('py.warnings').addHandler(RichHandler(show_path=False))


def check_config(config,
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

def collapse_suggest_config(config: dict) -> dict:
    """Use the default value of OptunaSuggest for config items.
    So that the configs can be used in the case that optuna is opted out.

    Args:
        config: The configuration dictionary

    Returns:
        The collapsed configuration
    """
    from .optuna import OptunaSuggest
    config = config.copy()
    collapsed = {key: val.default
                 for key, val in config.items()
                 if isinstance(val, OptunaSuggest)}
    if isinstance(config, FrozenDiot):
        with config.thaw():
            config.update(collapsed)
        return config
    config.update(collapsed)
    return FrozenDiot(config)

def normalize_tvt_ratio(
        tvt_ratio: Optional[Union[RatioType, Iterable[RatioType]]]
) -> Optional[Tuple[RatioType, List[RatioType], List[RatioType]]]:
    """Normalize the train-val-test data ratio into a format of
    (.7, [.1, .1], [.05, .05]).

    For `config.data_tvt`, the first element is required. If val or test ratios
    are not provided, it will be filled with `None`

    All numbers could be absolute numbers (>1) or ratios (<=1)

    Args:
        tvt_ratio: The train-val-test ratio

    Returns:
        The normalized ratios

    Raises:
        PlkitConfigException: When the passed-in tvt_ratio is in malformat
    """
    if not tvt_ratio:
        return None

    is_iter = lambda container: isinstance(container, (tuple, list))

    if not is_iter(tvt_ratio):
        tvt_ratio = [tvt_ratio]

    tvt_ratio = list(tvt_ratio)

    if len(tvt_ratio) < 3:
        tvt_ratio += [None] * (3 - len(tvt_ratio))

    if tvt_ratio[1] and not is_iter(tvt_ratio[1]):
        tvt_ratio[1] = [tvt_ratio[1]]
    if tvt_ratio[2] and not is_iter(tvt_ratio[2]):
        tvt_ratio[2] = [tvt_ratio[2]]

    return tuple(tvt_ratio)

@contextmanager
def warning_to_logging():
    """Patch the warning message formatting to only show the message"""
    orig_format = warnings.formatwarning
    logging.captureWarnings(True)
    warnings.formatwarning = (
        lambda msg, category, *args, **kwargs: f'{category.__name__!r}: {msg}'
    )
    yield
    warnings.formatwarning = orig_format
    logging.captureWarnings(False)

@contextmanager
def capture_stdout():
    """Capture the stdout"""
    _stdout = sys.stdout
    sys.stdout = stringio = StringIO()
    yield stringio
    del stringio
    sys.stdout = _stdout

@contextmanager
def capture_stderr():
    """Capture the stderr"""
    _stderr = sys.stderr
    sys.stderr = stringio = StringIO()
    yield stringio
    del stringio
    sys.stderr = _stderr

@contextmanager
def output_to_logging(stdout_level: str = 'info', stderr_level: str = 'error'):
    """Capture the stdout or stderr to logging"""
    with capture_stderr() as err, capture_stdout() as out:
        yield

    getattr(logger, stdout_level)(out.getvalue())
    getattr(logger, stderr_level)(err.getvalue())


def log_config(config, title='Configurations', items_per_row=1):
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

def plkit_seed_everything(config: FrozenDiot):
    """Try to seed everything and set deterministic to True
    if seed in config has been set

    Args:
        config: The configurations
    """
    if config.get('seed') is None:
        return

    seed_everything(config.seed)
    with config.thaw():
        config.setdefault('deterministic', True)
