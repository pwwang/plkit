"""Even higher level wrapper based on pytorch-lightning"""
from .data import Data
from .module import Module, log_hparams
from .trainer import Trainer

__version__ = "0.0.3"

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
