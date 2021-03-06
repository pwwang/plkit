"""Integration with optuna"""
from pathlib import Path
import torch
from torchvision import transforms
from torchvision.datasets import MNIST
from pytorch_lightning import metrics
from plkit import Module, DataModule, OptunaSuggest, Optuna, logger

class Data(DataModule):

    def data_reader(self):
        return MNIST(Path(__file__).parent / 'data',
                     download=True, transform=transforms.ToTensor())

class LitClassifier(Module):

    def __init__(self, config):
        super().__init__(config)
        self.fc1 = torch.nn.Linear(config.input_size, config.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(config.hidden_size, config.num_classes)
        self.val_acc = metrics.Accuracy()
        self.test_acc = metrics.Accuracy(compute_on_step=False)

    def forward(self, x):
        out = self.fc1(x.view(x.size(0), -1).float())
        out = self.relu(out)
        return self.fc2(out)

    def training_step(self, batch, _):
        x, y = batch
        loss = self.loss_function(self(x), y)
        return {'loss': loss}

    def validation_step(self, batch, _):
        x, y = batch
        y_hat = self(x)
        self.val_acc(y_hat, y)
        ret = {'val_loss': self.loss_function(y_hat, y),
               'val_acc': self.val_acc}
        self.log_dict(ret, prog_bar=True)
        return ret

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = self.val_acc.compute()
        ret = {'val_loss': avg_loss, 'val_acc': avg_acc}
        self.log_dict(ret, prog_bar=True)

    def test_step(self, batch, _):
        x, y = batch
        y_hat = self(x)
        self.test_acc(y_hat, y)

    def test_epoch_end(self, _):
        test_acc = self.test_acc.compute()
        self.log('test_acc', test_acc, logger=True)
        logger.info('test_acc: %s', test_acc)

if __name__ == '__main__':
    configuration = {
        'gpus': 1,
        'batch_size': 32,
        'max_epochs': 20,
        'num_classes': 10,
        'data_tvt': (300, 100, 100), # use a small proportion for example.
        'input_size': 28*28,
        # Let's say we are tuning hidden_size and seed
        'hidden_size': OptunaSuggest(512, 'cat', [128, 256, 512]),
        'seed': OptunaSuggest(1, 'int', 1, 10),
    }

    optuna = Optuna(on='val_acc', n_trials=10, direction='maximize')
    optuna.run(configuration, Data, LitClassifier)

    # if you want to run without optuna, you don't have change
    # any configurations, which OptunaSuggest objects will be collpased
    # into the default values
    #
    # from plkit import run
    # run(configuration, Data, LitClassifier)
