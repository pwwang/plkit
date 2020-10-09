import pytest

from diot import Diot
from torch.nn import Linear, MSELoss
from plkit.trainer import *
from plkit.module import *

from .test_data import DataModuleTest4

class Model(Module):
    def __init__(self, config, metric='accuracy'):
        super().__init__(config)
        self.metric = metric
        self.linear = Linear(8 * 10, 2)

    def training_step(self, batch, _):
        features, targets, i = batch
        out = self.linear(features.view([features.shape[0], 8*10]))
        loss = self.loss_function(out, targets)
        if self.metric == 'fbeta_score':
            acc = self.measure(out, targets, self.metric, beta=1.0)
        else:
            acc = self.measure(out, targets, self.metric)
        return {'loss': loss, 'acc': acc}

class ModelMeasureExc(Model):
    def training_step(self, batch, _):
        features, targets, i = batch
        out = self.linear(features.view([features.shape[0], 8*10]))
        loss = self.loss_function(out, targets)
        acc = self.measure(out, targets, 'nosuchmeasure')
        return {'loss': loss, 'acc': acc}

class ModelRegr(Module):

    def __init__(self, config):
        super().__init__(config)
        self.linear = Linear(8 * 10, 1)

    def training_step(self, batch, _):
        features, targets, i = batch
        out = self.linear(features.view([features.shape[0], 8*10]))
        targets = targets.float()
        loss = self.loss_function(out, targets)
        rmse = self.measure(out, targets, 'rmse')
        return {'loss': loss, 'rmse': rmse}

class ModelRegrMeasureExc(ModelRegr):
    def training_step(self, batch, _):
        features, targets, i = batch
        out = self.linear(features.view([features.shape[0], 8*10]))
        targets = targets.float()
        loss = self.loss_function(out, targets)
        rmse = self.measure(out, targets, 'nosuchmeasure')
        return {'loss': loss, 'rmse': rmse}

@pytest.mark.parametrize('metric', [
    'accuracy', 'fbeta_score', 'auroc'
])
def test_module(metric):
    config = Diot(batch_size=10, data_tvt=1, max_epochs=2, num_classes=2)
    trainer = Trainer.from_config(config)
    model = Model(config, metric)
    data = DataModuleTest4(config=config)
    trainer.fit(model, data)
    trainer.test()

def test_module_regression():
    config = Diot(batch_size=3, data_tvt=1, max_epochs=2, num_classes=1)
    trainer = Trainer.from_config(config)
    model = ModelRegr(config)
    data = DataModuleTest4(config=config)
    trainer.fit(model, data)
    trainer.test()

def test_module_regression_custom_loss():
    config = Diot(batch_size=3, data_tvt=1,
                  max_epochs=2, num_classes=1, loss=MSELoss())
    trainer = Trainer.from_config(config)
    model = ModelRegr(config)
    data = DataModuleTest4(config=config)
    trainer.fit(model, data)
    trainer.test()

def test_measure_exc():
    config = Diot(batch_size=3, data_tvt=1,
                  max_epochs=2, num_classes=2)
    trainer = Trainer.from_config(config)
    model = ModelMeasureExc(config)
    data = DataModuleTest4(config=config)
    with pytest.raises(PlkitMeasurementException):
        trainer.fit(model, data)

def test_regr_measure_exc():
    config = Diot(batch_size=3, data_tvt=1, optim='sgd',
                  max_epochs=2, num_classes=1)
    trainer = Trainer.from_config(config)
    model = ModelRegrMeasureExc(config)
    data = DataModuleTest4(config=config)
    with pytest.raises(PlkitMeasurementException):
        trainer.fit(model, data)

def test_logit_shape_exc():
    config = Diot(batch_size=3, data_tvt=1, max_epochs=2, num_classes=10)
    trainer = Trainer.from_config(config)
    model = Model(config)
    data = DataModuleTest4(config=config)
    with pytest.raises(PlkitMeasurementException):
        trainer.fit(model, data)

