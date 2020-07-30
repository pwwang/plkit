"""Data manager for plkit"""
import torch
from sklearn.model_selection import train_test_split
from pytorch_lightning import seed_everything
from .exceptions import PlkitDataException
from .utils import _check_config, _collapse_suggest_config

class Dataset(torch.utils.data.Dataset):
    """The dataset"""
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

class Data:
    """Data manager

    You have to define data_reader, to read the data and labels from the sources
    Then you will be able to get train_dataloader, val_dataloader and
    test_dataloader from this.
    """
    def __init__(self, config):
        """Construct

        Args:
            sources (any|list): The data sources
            batch_size (int): The batch size
            ratio (tuple): The ratio to split the data and targets
                They should be summed up to 1.0
                Expect 2 or 3 numbers.
                If two numbers are given, test_dataloader will not be available

                If not provided, the data_reader should return a dictionary
                with keys `train`, `val` or `test`, and values the data and
                labels
        """
        config = _collapse_suggest_config(config)

        seed_everything(config.get('seed'))

        _check_config(config, 'data_sources')
        self.sources = config['data_sources']

        self.config = config
        self.batch_size = config['batch_size']

        self.ratio = config.get('tvt_ratio')
        self.num_workers = config.get('data_workers', 1)

        self._train_dataloader = None
        self._val_dataloader = None
        self._test_dataloader = None

        alldata = self.data_reader()

        (self._train_data, self._train_ids,
         self._val_data, self._val_ids,
         self._test_data, self._test_ids) = (
             self._parse_data_read(alldata)
         )

    def data_reader(self):
        """Read the data and labels"""
        raise NotImplementedError()

    def _parse_data_read(self, data):
        """Check the returned value from data_reader"""

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

        else: # ratio specified
            if not isinstance(data, tuple):
                data = (data, )

            if not (isinstance(self.ratio, (list, tuple)) and
                    1 <= len(self.ratio) <= 3 and
                    all(0 <= rat <= 1.0 for rat in self.ratio)):
                raise PlkitDataException("Ratio should be a 2- or 3-element "
                                         "tuple with floats <= 1.0")

            all_ids = list(range(len(data[0])))

            train_ids, val_test_ids = train_test_split(
                all_ids, train_size=self.ratio[0]
            )
            # only training data
            if len(self.ratio) == 1:
                return (data, train_ids, None, None, None, None)

            if len(self.ratio) == 2:
                return (data, train_ids, data, val_test_ids, None, None)

            val_ids, test_ids = train_test_split(
                val_test_ids,
                train_size=(self.ratio[1]/(self.ratio[1]+self.ratio[2]))
            )
            return (data, train_ids,
                    data, val_ids,
                    data, test_ids)

    @property
    def train_dataloader(self):
        """Get the train_dataloader

        Raises:
            When failed to fetch train data

        Returns:
            The train data loader
        """
        if not self._train_ids:
            return None

        return torch.utils.data.DataLoader(
            Dataset(self._train_data,
                    self._train_ids),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )

    @property
    def val_dataloader(self):
        """Get the val_dataloader

        Raises:
            When failed to fetch validation data

        Returns:
            The validation data loader
        """
        if not self._val_ids:
            return None

        return torch.utils.data.DataLoader(
            Dataset(self._val_data,
                    self._val_ids),
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    @property
    def test_dataloader(self):
        """Get the test_dataloader

        Raises:
            When failed to fetch test data

        Returns:
            The test data loader
        """
        if not self._test_ids:
            return None

        return torch.utils.data.DataLoader(
            Dataset(self._test_data,
                    self._test_ids),
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )
