Ones of the principles of `plkit` is to try to put configuration items in just one dictionary for data and module construction. Any items that work as arguments for `pytorch-lightning`'s `Trainer` initialization could be valid configuration items ([See Trainer API from `pytorch-lightning`'s documentation][1]).

We do have some different or additional configuration items, in terms of their values or behaviors.

## seed

For full reproducibility, one should call `seed_everything` and set `deterministic` to `True` for trainer initialization using `pytorch-lightning` (see [Reproducibility][2])

However, with `plkit`, you only need to set a seed in the configuration (`seed_everything` will be set automatically), and `deterministic` will be automatically set to `True` for trainer initialization.

If you don't want `deterministic` to be `True` when a `seed` is specified, you can set `deterministric` to `False` in configuration.

## num_classes

Specification of `num_classes` in configuration ensures the builtin measurement calling the right loss function and metric for the output and labels (see [configuration loss](#loss) and [Builtin measurement][3] for more details).

## data_num_workers

`num_workers` argument of `DataLoader` for `DataModule`

## data_tvt

Train-val-test ratio for splitting the data read by `DataModule.data_reader`.

It could be a tuple with no more than 3 elements or just a single scalar element. The elements could be ratios (<=1) or absolute numbers.

The first element is for train set and later two are for validation and test sets, which can be a list respectively as multiple sets for validation and test.

If the ratio or number is not specified for the corresponding dataset, such dataset will not be generated. For example:

| data_tvt | Meaning |
|----------|---------|
| .7       | Use 70% of data for training (no val or test data).|
| (.7, .1) | Use 70% for training, 10% for val (no test data)|
| (.7, .15, .15) | Use 70% for training, 15% for val and 15% for test|
| 300 | Use 300 samples for training|
| (300, [100, 100], [100, 100]| Use 300 samples for training, 100 samples for validation (x2) and 100 for testing (x2)|

## optim

The name of optimizer (Currently only `adam` and `sgd` are supported).

## learning_rate

Learning rate for optimizers

## momentum

Momentum for SGD optimizer.

## loss

The loss function. It's `auto` by default, meaning `nn.MSELoss()` for regression (`num_classes=1`) and `nn.CrossEntropyLoss()` for classification. You can specifiy your own loss function: `loss=nn.L1Loss()` for example.

[1]: https://pytorch-lightning.readthedocs.io/en/stable/trainer.html#trainer-class-api
[2]: https://pytorch-lightning.readthedocs.io/en/stable/trainer.html?highlight=reproducibility#reproducibility
[3]: ../module#measure
