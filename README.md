# plkit

A wrapper based on pytorch-lightning

## Installation
```
pip install plkit
```

## Usage

### From pytorch-lightning's minimal example

```python
from plkit import Module, Data, Trainner

class MINISTData(Data):

    def data_reader(self, sources):
        minst = MNIST(sources, train=True, download=True,
                      transform=transforms.ToTensor())
        return {'train': (minst.data, minst.targets)}

class LitClassifier(Module):

    def __init__(self, config, num_classes, optim='adam', loss='auto'):
        super().__init__(config, num_classes, optim, loss)
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = self.loss_function(self(x), y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    # we don't need configure_optimizers anymore

config = {
    'batch_size': 32,
    'gpus': 8,
    'precision': 16,
    'num_classes': 2,
    # other options to initial a pytorch-lightning Trainer
}

def main():
    data = MINISTData(os.getcwd(), config['batch_size'])
    model = LitClassifier(config)
    trainer = Trainer.from_dict(config, data=data)
    trainer.fit(model)

if __name__ == '__main__':
    main()
```

Or even simpler for `main`:
```python
from plkit import run

# Data class and model class definition
# ...

def main():
    run(config, LitClassifier, MINISTData)

```

## Features

- Even more abstracted wrapper (What you only need to care is your data and model)
- Abstraction of data manager
- Trainer from a dictionary (not only from an `ArgumentParser` or a `Namespace`)
- Running `test` automatically when `test_dataloader` is given
- Auto loss function and optimizer
- Builtin measurements (accuracy, auc, etc, more to add)
- One function calling to do all stuff (`run`)
