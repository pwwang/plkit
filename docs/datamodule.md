`DatModule` from `plkit` is a subclass of `LightningDataModule` from `pytorch-lightning`. So you can follow everything that is documented by [`pytorch-lightning`][1]

Other than that, we have two additional methods defined: `data_reader` to read the data into a collection and `data_splits` to split the data collection for training, validation and testing.

The `data_reader` method is required to be defined if you want to use the features of `plkit`'s `DataModule` (auto-splitting data for example). Once you have it defined, you don't need to care about `data_prepare` and `setup` methods that `LightningDataModule` requires.

## data_reader

You can read data into a list of samples by `data_reader` or yield the samples to save some memory.

!!! note

    If `data_reader` is yielding (it is a generateor), `plkit.data.IterDataset` will be used, and samples can not be suffled.

!!! tip

    You can yield multiple features, as well as the labels at the same time as a tuple. For example:

    ```python
    class MyData(plkit.DataModule):
        ...
        def data_reader(
            yield (sample_name, feature_a, feature_b, label)
    ```

    Then in `training_step`, `validation_step` or `test_step`, you can easily decouple them like this:

    ```python
    class MyModule(plkit.Module):
        ...

        def training_step(self, batch, _):
            # Each variable has this batch of features
            sample_name, feature_a, feature_b, label = batch
    ```

    This is also the case when a list of samples returned from `data_reader`.

## data_splits

This method is supposed to split the data collection returned (yielded) from `data_reader`. If you have `data_tvt` (see [data_tvt in configuration][2]) specified in configuration, the collection will be specified automatically based on `data_tvt`. Otherwise, you can return a dictionary like this from `data_splits` to split the data by yourself:

```python
from plkit.data import DataModule, Dataset, IterDataset

class MyData(DataModule):
    ...

    def data_splits(self, data, stage):
        return {
            'train': Dataset(...),
            'val': Dataset(...), # or a list of datasets,
            'test': Dataset(...), # or a list of datasets
        }

```

!!! note

    `setup` is only calling at stage `fit`. If you want to do it at test stage, you will need to override `setup` method.

[1]: https://pytorch-lightning.readthedocs.io/en/stable/datamodules.html
[2]: ../configurations/data_tvt
