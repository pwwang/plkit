"""A minimal example for plkit"""

from pathlib import Path
import torch
from torchvision import transforms
from torchvision.datasets import MNIST
from plkit import Module, DataModule, run

class Data(DataModule):

    def data_reader(self):
        return MNIST(Path(__file__).parent / 'data', train=True,
                     download=True, transform=transforms.ToTensor())

class LitClassifier(Module):

    def __init__(self, config):
        super().__init__(config)
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1).float()))

    def training_step(self, batch, _):
        x, y = batch
        loss = self.loss_function(self(x), y)
        return {'loss': loss}

if __name__ == '__main__':
    configuration = {
        'gpus': 1,
        'data_tvt': .05, # use a small proportion for training
        'batch_size': 32,
        'max_epochs': 11
    }
    run(configuration, Data, LitClassifier)
