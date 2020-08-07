"""Data manager for plkit"""
import os
from itertools import islice
import torch
from sklearn.model_selection import train_test_split
from pytorch_lightning import seed_everything
from .exceptions import PlkitDataException
from .utils import (
    _check_config,
    _collapse_suggest_config,
    logger,
)

def _is_iterable(obj):
    """Make sure obj is an iterable but not a list with len known"""
    try:
        iter(obj)
    except TypeError:
        return None
    else:
        try:
            len(obj)
        except TypeError:
            return True
        else:
            return False

def _split_into(iterable, sizes):
    """Yield a list of iterables with given sizes

    Unlike more_itertools' split_into, this keeps inner generators.
    """
    ite = iter(iterable)

    for size in sizes:
        if size is None:
            yield ite
            return
        else:
            yield islice(ite, size)

class Dataset(torch.utils.data.Dataset):
    """The dataset that used internally by Data class
    """
    def __init__(self, data, ids):
        """Construct

        Args:
            data (tuple): A tuple of data, including targets that are read from
                data_reader
            ids (list): A list of ids corresponding to each of the data
        """
        self.data = data
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        gid = self.ids[idx]
        return tuple(dat[gid] for dat in self.data)

class IterDataset(torch.utils.data.IterableDataset):
    """Iterable dataset"""
    def __init__(self, data, length):
        self.data = data
        self.length = length

    def __len__(self):
        return self.length

    def __iter__(self):
        return iter(self.data)

class IterData:
    """Iterable data"""

    def __init__(self, config, dataset_class=IterDataset):
        seed_everything(config.get('seed'))
        config = _collapse_suggest_config(config)

        _check_config(config, 'data_sources')
        self.sources = config['data_sources']

        self.dataset_class = dataset_class

        self.config = config
        self.batch_size = config['batch_size']

        self.ratio = config.get('tvt_ratio')
        if self.ratio and not (isinstance(self.ratio, (list, tuple)) and
                               1 <= len(self.ratio) <= 3 and
                               all(0 <= rat <= 1.0 for rat in self.ratio)):
            raise PlkitDataException("Ratio should be a 2- or 3-element "
                                     "tuple with floats <= 1.0")

        self.num_workers = config.get('data_workers', 1)
        avai_workers = len(os.sched_getaffinity(0))
        if self.num_workers != 0 and self.num_workers < avai_workers:
            logger.warning('Consider increasing `data_workers`. Available: %d',
                           avai_workers)

        self._train_dataloader = None
        self._val_dataloader = None
        self._test_dataloader = None

        alldata = self.data_reader()

        (self._train_data, self._train_ids, # length
         self._val_data, self._val_ids,
         self._test_data, self._test_ids) = (
             self._parse_data_read(alldata)
         )

    def data_reader(self):
        """Read the data and labels, and return a tuple of data items
        These data items will be then fetched in the steps in the way:
            >>> x, y, z = batch
        """
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def _parse_data_read(self, data):
        """Check the returned value from data_reader

        Args:
            data (dict|generator): The data returned from `data_reader`

        Returns:
            tuple: The train/val/test data and ids
        """

        length = len(self)
        if not self.ratio:
            if (
                    not isinstance(data, dict) or
                    set(data.keys()) - {'train', 'val', 'test'}
                ):
                raise PlkitDataException(
                    "No ratio specified, expecting data_reader to return "
                    "a dictionary with `train`, `val`, and/or `test` as keys."
                )
            # data is dict
            train_data = data.get('train')
            val_data = data.get('val')
            test_data = data.get('test')

            return (train_data, length.get('train'),
                    val_data, length.get('val'),
                    test_data, length.get('test'))

        # ratio specified
        # data should be a generator

        train_size = int(self.ratio[0] * length)

        if len(self.ratio) == 1:
            return (list(_split_into(data, [train_size]))[0],
                    train_size) + (None, ) * 4

        if len(self.ratio) == 2:
            val_size = (int(self.ratio[1] * length)
                        if sum(self.ratio) < 1.0
                        else length - train_size)
            # split data to train/val
            data = list(_split_into(data, [train_size, val_size]))

            return (data[0], train_size, data[1], val_size, None, None)

        if self.ratio[1] == 0.0:
            test_size = (int(self.ratio[1] * length)
                         if sum(self.ratio) < 1.0
                         else length - train_size)
            # split each data item to train/test
            data = list(_split_into(data, [train_size, test_size]))

            return (data[0], train_size, None, None, data[1], test_size)

        val_size = int(self.ratio[1] * length)
        test_size = (int(self.ratio[2] * length)
                     if sum(self.ratio) < 1.0
                     else length - train_size - val_size)

        data = list(_split_into(data, [train_size, val_size, test_size]))

        return (data[0], train_size, data[1], val_size, data[2], test_size)


    @property
    def train_dataloader(self):
        """Get the train_dataloader

        Returns:
            DataLoader: The train data loader
        """
        if not self._train_ids:
            return None

        return torch.utils.data.DataLoader(
            self.dataset_class(self._train_data, self._train_ids),
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    @property
    def val_dataloader(self):
        """Get the val_dataloader

        Returns:
            DataLoader: The validation data loader if provided
        """
        if not self._val_ids:
            return None

        return torch.utils.data.DataLoader(
            self.dataset_class(self._val_data, self._val_ids),
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    @property
    def test_dataloader(self):
        """Get the test_dataloader

        Returns:
            DataLoader: The test data loader if test data provided
        """
        if not self._test_ids:
            return None

        return torch.utils.data.DataLoader(
            self.dataset_class(self._test_data, self._test_ids),
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

class Data(IterData):

    def __init__(self, config, dataset_class=Dataset):
        super().__init__(config, dataset_class)

    def _parse_data_read(self, data):
        """Check the returned value from data_reader

        Args:
            data (tuple): The data returned from `data_reader`

        Returns:
            tuple: The train/val/test data and ids
        """

        if not self.ratio:
            if (
                    not isinstance(data, dict) or
                    set(data.keys()) - {'train', 'val', 'test'}
                ):
                raise PlkitDataException(
                    "No ratio specified, expecting data_reader to return "
                    "a dictionary with `train`, `val`, and/or `test` as keys."
                )
            # data is dict
            train_data = data.get('train')
            val_data = data.get('val')
            test_data = data.get('test')

            if train_data and not isinstance(train_data, tuple):
                train_data = (train_data, )
            if val_data and not isinstance(val_data, tuple):
                val_data = (val_data, )
            if test_data and not isinstance(test_data, tuple):
                test_data = (test_data, )

            train_ids = list(range(len(train_data[0]))) if train_data else None
            val_ids = list(range(len(val_data[0]))) if val_data else None
            test_ids = list(range(len(test_data[0]))) if test_data else None

            return (
                train_data, train_ids,
                val_data, val_ids,
                test_data, test_ids,
            )

        # ratio specified
        if not isinstance(data, tuple):
            data = (data, )

        all_ids = list(range(len(data[0])))

        train_ids, val_test_ids = train_test_split(
            all_ids, train_size=self.ratio[0]
        )
        # only training data
        if len(self.ratio) == 1:
            return (data, train_ids, None, None, None, None)

        if len(self.ratio) == 2:
            return (data, train_ids, data, val_test_ids, None, None)

        # allow empty validaation set.
        if self.ratio[1] == 0.0:
            return (data, train_ids, None, None, data, val_test_ids)

        val_ids, test_ids = train_test_split(
            val_test_ids,
            train_size=(self.ratio[1]/(self.ratio[1]+self.ratio[2]))
        )
        return (data, train_ids,
                data, val_ids,
                data, test_ids)
