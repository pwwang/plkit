import pytest

from diot import FrozenDiot
from plkit.trainer import *

from .test_data import Model, DataModuleTest4

def test_instance():
    trainer = Trainer()
    assert isinstance(trainer, PlTrainer)

def test_case():
    config = FrozenDiot(batch_size=3, data_tvt=1, max_epochs=1)
    trainer = Trainer.from_config(config)
    model = Model()
    data = DataModuleTest4(config=config)
    trainer.fit(model, data)
    trainer.test()
