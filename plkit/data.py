"""Data manager for plkit"""
import torch
from sklearn.model_selection import train_test_split
from pytorch_lightning import seed_everything
from .exceptions import PlkitDataException

class Dataset(torch.utils.data.Dataset):
    """The dataset"""
    def __init__(self, data, targets, ids, with_name=False):
        self.data = data
        self.with_name = with_name
        self.targets = targets
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        gid = self.ids[idx]
        if self.with_name:
            return torch.Tensor(self.data[gid]), self.targets[gid], gid
        return torch.Tensor(self.data[gid]), self.targets[gid]

class Data:
    """Data manager

    You have to define data_reader, to read the data and labels from the sources
    Then you will be able to get train_dataloader, val_dataloader and
    test_dataloader from this.
    """
    def __init__(self, sources, batch_size,
                 ratio=None, with_name=False, num_workers=1, seed=None):
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
            with_name (bool): Whether to return name with each sample.
                Normally, you can do `x, y = batch` to get the data and lables.
                However, in some cases, you can also trace the name of those
                samples: `x, y, names = batch`

                To do this, you will have to return dictionaries for data and
                labels, with the sample names as keys.
        """
        seed_everything(seed)
        self.sources = sources
        self.batch_size = batch_size
        self.ratio = ratio
        self.with_name = with_name
        self.num_workers = num_workers

        self._train_dataloader = None
        self._val_dataloader = None
        self._test_dataloader = None

        alldata = self.data_reader(sources)

        (self._train_data, self._train_labels, self._train_ids,
         self._val_data, self._val_labels, self._val_ids,
         self._test_data, self._test_labels, self._test_ids) = (
             self._parse_data_read(alldata)
         )

    def data_reader(self, sources):
        """Read the data and labels"""
        raise NotImplementedError()

    def _parse_data_read(self, data):
        """Check the returned value from data_reader"""

        def check_with_name(dat):
            """Check if the dat is a dictionary when self.with_name = True"""
            if self.with_name and not isinstance(dat, dict):
                raise PlkitDataException("Expecting a dictionary when "
                                         "with_name = True")
            if not self.with_name and not isinstance(dat, (tuple, list)):
                raise PlkitDataException("Expecting a tuple/list when "
                                         "with_name = False")

        if not self.ratio:
            if (
                    not isinstance(data, dict) or
                    set(data.keys()) - {'train', 'val', 'test'}
                ):
                raise PlkitDataException(
                    "No ratio specified, expecting data_reader to return "
                    "a dictionary with `train`, `val`, and/or `test` as keys."
                )
            for which, value in data.items():
                if len(value) != 2:
                    raise PlkitDataException("Expecting data and labels "
                                             f"for {which} data.")
                check_with_name(value[0])
                check_with_name(value[1])

            train_datalabels = data.get('train', (None, None))
            train_ids = (list(train_datalabels[0].keys())
                         if isinstance(train_datalabels[0], dict)
                         else list(range(len(train_datalabels[0])))
                         if isinstance(train_datalabels[0], (tuple, list))
                         else None)

            val_datalabels = data.get('val', (None, None))
            val_ids = (list(val_datalabels[0].keys())
                       if isinstance(val_datalabels[0], dict)
                       else list(range(len(val_datalabels[0])))
                       if isinstance(val_datalabels[0], (tuple, list))
                       else None)

            test_datalabels = data.get('test', (None, None))
            test_ids = (list(test_datalabels[0].keys())
                        if isinstance(test_datalabels[0], dict)
                        else list(range(len(test_datalabels[0])))
                        if isinstance(test_datalabels[0], (tuple, list))
                        else None)

            return (
                train_datalabels[0], train_datalabels[1], train_ids,
                val_datalabels[0], val_datalabels[1], val_ids,
                test_datalabels[0], test_datalabels[1], test_ids,
            )

        else: # ratio specified
            if not (isinstance(self.ratio, (list, tuple)) and
                    1 <= len(self.ratio) <= 3 and
                    sum(self.ratio) == 1.0):
                raise PlkitDataException("Ratio should be a 2- or 3-element "
                                         "tuple with sum of 1.0")

            if len(data) != 2:
                raise PlkitDataException("Expecting data and targets from "
                                         "data_reader")
            check_with_name(data[0])
            check_with_name(data[1])

            all_ids = (list(data[0].keys())
                       if isinstance(data[0], dict)
                       else list(range(len(data[0])))
                       if isinstance(data[0], (tuple, list))
                       else None)

            # all are training data
            if len(self.ratio) == 1 and self.ratio[0] == 1.0:
                return (data[0], data[1], all_ids,
                        None, None, None,
                        None, None, None)

            train_ids, val_test_ids = train_test_split(
                all_ids, train_size=self.ratio[0]
            )
            if len(self.ratio) == 2:
                return (data[0], data[1], train_ids,
                        data[0], data[1], val_test_ids,
                        None, None, None)

            val_ids, test_ids = train_test_split(
                val_test_ids,
                train_size=(self.ratio[1]/(self.ratio[1]+self.ratio[2]))
            )
            return (data[0], data[1], train_ids,
                    data[0], data[1], val_ids,
                    data[0], data[1], test_ids)

    @property
    def train_dataloader(self):
        """Get the train_dataloader

        Raises:
            When failed to fetch train data

        Returns:
            The train data loader
        """
        if not self._train_data:
            return None

        return torch.utils.data.DataLoader(
            Dataset(self._train_data,
                    self._train_labels,
                    self._train_ids,
                    with_name=self.with_name),
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
        if not self._val_data:
            return None

        return torch.utils.data.DataLoader(
            Dataset(self._val_data,
                    self._val_labels,
                    self._val_ids,
                    with_name=self.with_name),
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
        if not self._test_data:
            return None

        return torch.utils.data.DataLoader(
            Dataset(self._test_data,
                    self._test_labels,
                    self._test_ids,
                    with_name=self.with_name),
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )
