"""Superset of pytorch-lightning"""
from typing import Any, Dict, Optional, Type
from .data import DataModule
from .module import Module
from .trainer import Trainer
from .optuna import Optuna, OptunaSuggest
from .runner import Runner, LocalRunner, SGERunner
from .utils import logger

__version__ = "0.0.7"

def run(config: Dict[str, Any],
        data_class: Type[DataModule],
        model_class: Type[Module],
        optuna: Optional[Optuna] = None,
        runner: Runner = LocalRunner()) -> Trainer:
    """Run the pipeline by give configuration, model_class, data_class, optuna
    and runner

    Args:
        config: A dictionary of configuration, must have following items:
            - batch_size: The batch size
            - num_classes: The number of classes for classification
                1 means regression
        data_class: The data class subclassed from `Data`
        model_class: The model class subclassed from `Module`
        optuna: The optuna object
        runner: The runner object

    Returns:
        The trainer object
    """

    return runner.run(config, data_class, model_class, optuna)
