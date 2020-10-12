import pytest
from diot import Diot, FrozenDiot, DiotFrozenError

from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms

from plkit import DataModule, Module, Optuna, Trainer, OptunaSuggest, run
from plkit.utils import output_to_logging

class Data(DataModule):

    def data_reader(self):
        with output_to_logging():
            return MNIST(self.config.datadir, download=True,
                         transform=transforms.ToTensor())

class Model(Module):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = nn.Sequential(nn.Linear(28 * 28, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, 3))
        self.decoder = nn.Sequential(nn.Linear(3, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, 28 * 28))

    def training_step(self, batch, _):
        # training_step defined the train loop. It is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        return {'val_loss': loss}

    def test_step(self, batch, _):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        return {'test_loss': loss}

    def validation_epoch_end(self, outputs):
        mean_loss = sum(output['val_loss']
                        for output in outputs) / float(len(outputs))
        self.log('val_avg_loss', mean_loss, on_epoch=True, prog_bar=True)

@pytest.mark.parametrize('config', [
    FrozenDiot(
        batch_size=32,
        max_epochs=5,
        seed=8525,
        # take a small set
        data_tvt=(300, 100, 100),
        learning_rate=OptunaSuggest(1e-4, 'float', 1e-5, 1e-3)
    ),
    dict(
        batch_size=32,
        max_epochs=5,
        seed=8525,
        # take a small set
        data_tvt=(300, 100, 100),
        learning_rate=OptunaSuggest(1e-4, 'float', 1e-5, 1e-3)
    )
])
def test_run(config, tmp_path):
    try:
        config['datadir'] = str(tmp_path)
    except DiotFrozenError:
        with config.thaw():
            config.datadir = str(tmp_path)
    optuna = Optuna(on='val_avg_loss', n_trials=3)
    if isinstance(config, FrozenDiot):
        trainer = run(config, Data, Model, optuna)
    else:
        trainer = optuna.run(config, Data, Model)
    assert isinstance(trainer, Trainer)

    assert len(optuna.best_params) == 1
    assert 'learning_rate' in optuna.best_params

    assert isinstance(optuna.trials, list)
    assert optuna.trials[0].number == 0
