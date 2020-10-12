"""Plkit Example for MNIST"""
from pathlib import Path
import torch
from diot import Diot
from torchvision import transforms
from torchvision.datasets import MNIST
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
        acc = self.measure(y_hat, y, 'accuracy')
        return {'val_loss': torch.nn.functional.cross_entropy(y_hat, y),
                'val_acc': acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        self.log_dict({'val_loss': avg_loss.item(),
                       'val_acc': avg_acc.item()}, prog_bar=True, logger=True)
        # return {'val_loss': avg_loss, 'val_acc': avg_acc}

    def test_step(self, batch, _):
        x, y = batch
        y_hat = self(x)
        acc = self.measure(y_hat, y, 'accuracy')
        return {'test_acc': acc}

    def test_epoch_end(self, outputs):
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        # also log it to terminal
        logger.info('TEST AVERAGE ACCURACY: %s', avg_acc.item())
        self.log('test_acc', avg_acc.item(), on_epoch=True)

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
