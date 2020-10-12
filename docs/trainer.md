The `Trainer` class is also a subclass of `Trainer` from `pytorch-lightning`, with an addition class method `from_config` (aka `from_dict`), which initializes a trainer object.

So that you can do:

```python
configuration = {
    'gpus': 1,
    'data_tvt': .05, # use a small proportion for training
    'batch_size': 32,
    'max_epochs': 11
}

trainer = Trainer.from_config(configuration)
# or if you have some extra arguments
trainer = Trainer.from_config(configuration, **kwargs)
```
