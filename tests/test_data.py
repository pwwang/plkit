import pytest

import numpy
from diot import FrozenDiot
from torch.nn import CrossEntropyLoss, Linear
from torch.optim import Adam
from torch.utils.data._utils.collate import default_convert
from pytorch_lightning import seed_everything, Trainer, LightningModule
from plkit.data import *
from random import choice

seed_everything(8525)

class DataModuleTest1(DataModule):

    @property
    def length(self):
        return 10

    def data_reader(self):
        for i in range(10):
            yield (f'a{i}', f'b{i}')

class DataModuleTest1_np(DataModule):

    @property
    def length(self):
        return 10

    def data_reader(self):
        for i in range(10):
            yield (i, numpy.ndarray([3,3]), choice([0, 1]))

class DataModuleTest2(DataModule):

    def data_reader(self):
        return [
            (f'a{i}', f'b{i}') for i in range(10)
        ]

class DataModuleTest3(DataModule):

    @property
    def length(self):
        return 10

    def data_reader(self):
        for _ in range(10):
            yield (numpy.ndarray([10, 10], dtype=numpy.float32), choice([0, 1]))

class DataModuleTest4(DataModule):

    def data_reader(self):
        return [
            (numpy.ndarray([8, 10], dtype=numpy.float32), choice([0, 1]), f'i{i}')
            for i in range(10)
        ]

class Model(LightningModule):

    def __init__(self) -> None:
        super().__init__()
        self.linear = Linear(8 * 10, 2)
        self.loss_func = CrossEntropyLoss()

    def forward(self, features, targets):
        out = self.linear(features.view([features.shape[0], 8*10]))
        return self.loss_func(out, targets)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch, _):
        features, targets, i = batch
        loss = self(features, targets)

        return {'loss': loss}

def test_dataset():
    ds = Dataset([(f'a{i}', f'b{i}') for i in range(10)], ids=[1,2,3])
    assert len(ds) == 3
    assert ds[0] == ('a1', 'b1')

def test_dataset_np():
    ds = Dataset([(numpy.zeros([2,2]), i) for i in range(10)], ids=[1,2,3])
    assert len(ds) == 3
    assert ds[0][0].sum() == 0.0
    assert ds[0][0].shape == (2, 2)
    assert ds[0][1] == 1

def test_dataset_iter():
    def d():
        for i in range(10):
            yield (f'a{i}', f'b{i}')
    ds = IterDataset(d(), length=10)
    assert len(ds) == 10
    assert next(iter(ds)) == ('a0', 'b0')

def test_dataloader():
    ds = Dataset([(i, i) for i in range(10)], ids=[1,2,3])
    dl = DataLoader(ds, batch_size=2, collate_fn=default_convert)
    assert list(dl) == [[[1,1], [2,2]], [[3,3]]]

def test_dataloader_np():
    config = FrozenDiot(batch_size=3, data_tvt=1)
    data = DataModuleTest4(config=config)
    data.prepare_data()
    data.setup('fit')
    train_data = list(data.train_dataloader())
    assert len(train_data) == 4
    batch1 = train_data[0]
    assert batch1[0].shape == (3, 8, 10)
    assert batch1[1].shape == (3, )
    assert len(batch1[2]) == 3

def test_data_module():
    config = FrozenDiot(batch_size=3, data_tvt=(.1, [.7], [.2]), gpus=0)

    data = DataModuleTest2(config=config)
    data.prepare_data()
    data.setup('fit')

    assert isinstance(data, DataModuleTest2)
    assert isinstance(data.val_dataloader(), DataLoader)

    val_data = list(data.val_dataloader())
    assert len(val_data) == 3
    assert len(val_data[0]) == 2 # ('a0', 'a1', 'a2'), ('b0', 'b1', 'b2')
    assert len(val_data[1]) == 2
    assert len(val_data[2]) == 2

    test_data = list(data.test_dataloader())
    assert len(test_data) == 1
    assert len(test_data[0]) == 2 # ('a0', 'a1', 'a2'), ('b0', 'b1', 'b2')

def test_data_module_iter():
    config = FrozenDiot(batch_size=3, data_tvt=(.3,.7))

    data = DataModuleTest1(config=config)
    data.prepare_data()
    data.setup('fit')

    assert isinstance(data, DataModuleTest1)
    assert isinstance(data.val_dataloader(), DataLoader)

    val_data = list(data.val_dataloader())
    assert list(val_data[0]) == [('a3', 'a4', 'a5'), ('b3', 'b4', 'b5')]
    assert list(val_data[1]) == [('a6', 'a7', 'a8'), ('b6', 'b7', 'b8')]
    assert list(val_data[2]) == [('a9',), ('b9',)]

def test_data_module_iter_np():
    config = FrozenDiot(batch_size=3, data_tvt=(.15,.15,.7))

    data = DataModuleTest1_np(config=config)
    data.prepare_data()
    data.setup('fit')

    assert isinstance(data, DataModuleTest1_np)
    assert isinstance(data.test_dataloader(), DataLoader)

    train_data = list(data.test_dataloader())
    batch1 = train_data[0]
    idx, feat, label = batch1
    assert idx.shape == (3, )
    assert feat.size() == (3, 3, 3)
    assert len(label) == 3

def test_case():
    config = FrozenDiot(batch_size=3, data_tvt=1)
    data = DataModuleTest4(config=config)
    model = Model()
    trainer = Trainer(max_epochs=1)
    trainer.fit(model, data)

def test_iterdata_no_length():
    config = FrozenDiot(batch_size=3, data_tvt=.1)
    class DM(DataModule):
        def data_reader(self):
            yield 1
    dm = DM(config=config)
    dm.prepare_data()
    with pytest.raises(PlkitDataException):
        dm.setup('fit')

def test_none_data():
    dm = DataModule(config={'batch_size':3})
    assert dm.data_splits() is None
    with pytest.raises(PlkitDataException):
        dm.setup('fit')
    dm.splits = {}
    assert dm.train_dataloader() is None
    assert dm.val_dataloader() is None
    assert dm.test_dataloader() is None
