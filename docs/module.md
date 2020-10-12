Class `Module` is subclassed from `LightningModule` of `pytorch-lightning`.

The differences are list below.

## Initialization

Every instance of `Module` needs a `config` argument to initialize. It collapses the `OptunaSuggest` objects in it and becomes an attributes of the instance. So the configuration item can be accessed by `self.config.xxx`.

!!! Danger

    `self.config` is a `FrozenDiot` object to prevent the changes to be passed down to other modules (See [diot][1]).

## on_epoch_end

An empty line is printed on every epoch end to keep the progress bar of the previous epoch run so that we can easily keep track of the metrics of all epochs. If you want to disable this just overwrite `on_epoch_end` with a `pass` statement. Or if you want to keep this and do something else in `on_epoch_end`, you should call `super().on_epoch_end()` inside your `on_epoch_end`.

## loss_function

Calculate loss using the loss function according to the `loss` configuration item.

## configure_optimizers

You don't have to implement this function if you have `optim` specified in configuration.

!!! note

    Currently only `adam` and `sgd` are supported. Please also specify `learning_rate` and/or `momentum` for corresponding optimizers (see [configurations][2]).

## measure

This method calculates a metric according to `num_classes`. For `num_classes=1` (regression), available metrics are `mse`, `rmse`, `mae` and `rmsle`. And for classifications, avaiable metrics are `accuracy`, `precision`, `recall`, `f1_score`, `iou`, `fbeta_score`, `auroc`, `average_precision` and `dice_score`.

To use it in your `training_step`, `validation_step` or `test_step`, you can do:
```python
self.measure(output, labels, 'accuracy')
```
If the metric need extra arguments, you can pass them in as well:
```python
self.measure(output, labels, 'fbeta_score', beta=1.0)
```

For more details of the metrics, see `pytorch-lightning`'s [documentation][3]

[1]: https://github.com/pwwang/diot#frozendiot
[2]: ../configurations
[3]: https://pytorch-lightning.readthedocs.io/en/stable/metrics.html
