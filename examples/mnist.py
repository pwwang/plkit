import os
import torch
# pip install diot
from diot import Diot
from torchvision import transforms
from torchvision.datasets import MNIST
from plkit import Module, Data as PkData, run

class Data(PkData):

    def data_reader(self):
        minst = MNIST(self.sources, train=True,
                      download=True, transform=transforms.ToTensor())
        minst_test = MNIST(self.sources, train=False,
                           download=True, transform=transforms.ToTensor())
        return {'train': (minst.data, minst.targets),
                'val': (minst.data, minst.targets),
                'test': (minst_test.data, minst_test.targets)}

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

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = self.loss_function(self(x), y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        acc = self.measure(y_hat, y, 'accuracy')
        return {'val_loss': torch.nn.functional.cross_entropy(y_hat, y),
                'val_acc': acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': avg_acc}
        return {'val_loss': avg_loss, 'val_acc': avg_acc,
                'log': tensorboard_logs}

    def test_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        acc = self.measure(y_hat, y, 'accuracy')
        return {'test_acc': acc}

    def test_epoch_end(self, outputs):
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        tensorboard_logs = {'test_acc': avg_acc}
        return {'test_acc': avg_acc, 'log': tensorboard_logs}

if __name__ == '__main__':
    config = Diot({
        'gpus': 1,
        'batch_size': 32,
        'max_epochs': 20,
        'num_classes': 10,
        'hidden_size': 512,
        'input_size': 28*28,
        'data_sources': os.path.join(os.path.dirname(__file__), 'data'),
    }) # so that we can do config.input_size
    run(config, Data, LitClassifier)
