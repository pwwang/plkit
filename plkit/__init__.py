"""Even higher level wrapper based on pytorch-lightning"""
from .data import Data, IterData
from .module import Module
from .trainer import Trainer
from .optuna import Optuna, OptunaSuggest
from .utils import log_config, _check_config, logger

__version__ = "0.0.6"

def run(config: dict,
        data_class: callable,
        model_class: callable):
    """Run the pipeline by give configuration, model_class and data_class

    Args:
        config (dict): A dictionary of configuration, must have following items:
            - sources: The sources to read data from
            - batch_size: The batch size
            - num_classes: The number of classes for classification
                1 means regression
        data_class (class): The data class subclassed from `Data`
        model_class (class): The model class subclassed from `Module`
    """
    _check_config(config, 'batch_size')

    data = data_class(config)
    model = model_class(config)
    trainer = Trainer.from_config(config, data=data)
    trainer.fit(model)
    trainer.test()
    return trainer
