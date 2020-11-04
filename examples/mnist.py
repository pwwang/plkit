"""Plkit Example for MNIST"""
from pathlib import Path
import torch
from diot import Diot
from torchvision import transforms
from torchvision.datasets import MNIST
from pytorch_lightning import metrics
from plkit import Module, DataModule, run, logger

HERE = Path(__file__).parent

class Data(DataModule):

    def data_reader(self):
        return MNIST(HERE / 'data',
                     download=True,
                     transform=transforms.ToTensor())

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
        return {'val_loss': self.loss_function(y_hat, y),
                'val_acc': self.val_acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = self.val_acc.compute()
        ret = {'val_loss': avg_loss, 'val_acc': avg_acc}
        self.log_dict(ret, prog_bar=True)

    def test_step(self, batch, _):
        x, y = batch
        y_hat = self(x)
        self.test_acc(y_hat, y)

    def test_epoch_end(self, outputs):
        test_acc = self.test_acc.compute()
        # also log it to terminal
        logger.info('TEST AVERAGE ACCURACY: %s', test_acc)
        self.log('test_acc', test_acc, on_epoch=True)

if __name__ == '__main__':
    configuration = Diot(
        gpus=1,
        data_tvt=(.7, .15, .15),
        batch_size=32,
        max_epochs=20,
        num_classes=10,
        hidden_size=512,
        input_size=28*28,
    ) # so that we can do config.input_size
    run(configuration, Data, LitClassifier)
