import os
import torch
from torchvision import transforms
from torchvision.datasets import MNIST
from plkit import Module, Data as PkData, run

class Data(PkData):

    def data_reader(self):
        minst = MNIST(self.sources, train=True,
                      download=True, transform=transforms.ToTensor())
        return {'train': (minst.data, minst.targets)}

class LitClassifier(Module):

    def __init__(self, config):
        super().__init__(config)
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1).float()))

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = self.loss_function(self(x), y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

if __name__ == '__main__':
    config = {
        'gpus': 1,
        'batch_size': 32,
        'max_epochs': 10,
        'data_sources': os.path.join(os.path.dirname(__file__), 'data'),
    }
    run(config, Data, LitClassifier)
