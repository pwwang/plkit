"""Data module for plkit"""
from types import GeneratorType
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union
from itertools import islice
from diot import FrozenDiot
from pytorch_lightning import LightningDataModule
from torch.utils.data import (
    DataLoader,
    Dataset as TorchDataset,
    IterableDataset as TorchIterableDataset,
    random_split
)
from .exceptions import PlkitDataException
from .utils import (
    normalize_tvt_ratio,
    check_config,
    logger
)

# pylint: disable=unused-argument

# The ids or keys for the data
DatasetType = Union[TorchDataset, TorchIterableDataset]

class Dataset(TorchDataset):
    """The dataset that used internally by Data class

    Examples:
    >>> ds = Dataset(data=[('a', 'x'), ('b', 'y'), ('c', 'z')], ids=[1, 2])
    >>> len(ds) == 2
    >>> ds[0] == ('b', 'y')
    >>> ds[1] == ('c', 'z')
    >>> # The features are what you get by
    >>> # x, y = batch

    Args:
        data: The data for the dataset.
            It could be a tuple of features. Each one should be an iterable,
            which could be accessed by index
        ids: The ids or keys of the data, which should be in the same order
            of each feature in the iterable.
    """
    def __init__(self,
                 data: Iterable[tuple],
                 ids: Optional[List[int]] = None) -> None:
        self.data = data
        self.ids = ids or list(range(len(data)))

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: Union[int, str]) -> Tuple[Any]:
        data_id = self.ids[idx]
        return self.data[data_id]

class IterDataset(TorchIterableDataset):
    """Iterable dataset

    The iterable dataset where each feature of the data is an iterable

    Examples:
    >>> feat1 = (x for x in range(10)
    >>> feat2 = (x for x in range(10)
    >>> ds = IterDataset(zip(feat1, feat2), ids=[4,3])
    >>> next(ds) == (0, 0)

    Args:
        data: a tuple of iterable features
        length: The length of the iterables
    """
    # pylint: disable=abstract-method
    def __init__(self,
                 data: Iterable[tuple],
                 length: int) -> None:
        self.data = data
        self.length = length

    def __iter__(self) -> Iterable[Any]:
        return iter(self.data)

class DataModule(LightningDataModule):
    """Data module for plkit"""
    def __init__(self,
                 train_transforms=None,
                 val_transforms=None,
                 test_transforms=None,
                 config: Optional[FrozenDiot] = None) -> None:
        super().__init__(train_transforms=train_transforms,
                         val_transforms=val_transforms,
                         test_transforms=test_transforms)
        self.config = config or FrozenDiot()

        check_config(self.config, 'batch_size')
        self.num_workers = self.config.get('data_num_workers', 0)
        self.tvt_ratio = normalize_tvt_ratio(self.config.get('data_tvt'))
        self.data = None
        self.splits = None
        self._length = None

    @property
    def length(self) -> int:
        """The length of the data

        This is required when `self.data_reader()` yields (it is a generator)

        Returns:
            The length of the data.
        """
        return self._length

    def data_reader(self) -> Union[Iterable[Any], Tuple[Iterator[Any]]]:
        """Read the data

        Returns:
            A tuple of iterables of features. Or it yields the following

        Yields:
            An iterable of tuple of features. In such a case, self.length
            property is required to be defined.
        """
        raise NotImplementedError # pragma: no cover

    def _split_data_generator(
            self, data: Iterable[tuple]
    ) -> Dict[str, Union[TorchIterableDataset, List[TorchIterableDataset]]]:
        ret = {}
        is_ratio = self.tvt_ratio[0] <= 1.0
        if is_ratio and self.length is None:
            raise PlkitDataException(
                'Got generator from `data_reader` and ratios from '
                '`config.data_tvt`, `self.length` should be recorded '
                'in `data_reader`.'
            )
        # split using islice
        start = 0
        train_len = (round(self.tvt_ratio[0] * float(self.length))
                     if is_ratio else self.tvt_ratio[0])
        ret['train'] = IterDataset(islice(data, start, train_len), train_len)
        start += train_len

        if self.tvt_ratio[1]:
            ret['val'] = []
            for val_ratio in self.tvt_ratio[1]:
                val_len = (round(val_ratio * float(self.length))
                           if is_ratio else val_ratio)
                ret['val'].append(IterDataset(
                    islice(data, start, start + val_len), val_len
                ))
                start += val_len

        if self.tvt_ratio[2]:
            ret['test'] = []
            for test_ratio in self.tvt_ratio[2]:
                test_len = (round(test_ratio * float(self.length))
                            if is_ratio else test_ratio)
                ret['test'].append(IterDataset(
                    islice(data, start, start + test_len), test_len
                ))
                start += test_len
        return ret

    def _split_data_list(
            self, data: List[Any]
    ) -> Dict[str, Union[IterDataset, List[IterDataset]]]:
        ret = {}
        is_ratio = self.tvt_ratio[0] <= 1.0
        self._length = len(data)
        all_ids = range(self.length)
        train_len = (round(self.tvt_ratio[0] * float(self.length))
                     if is_ratio else self.tvt_ratio[0])
        train_ids, rest_ids = random_split(
            all_ids, [train_len, len(all_ids) - train_len]
        )

        ret['train'] = Dataset(data, train_ids)

        if self.tvt_ratio[1]:
            ret['val'] = []
            for val_ratio in self.tvt_ratio[1]:
                val_len = (round(val_ratio * float(self.length))
                           if is_ratio else val_ratio)
                val_ids, rest_ids = random_split(
                    rest_ids, [val_len, len(rest_ids) - val_len]
                )
                ret['val'].append(Dataset(data, val_ids))

        if self.tvt_ratio[2]:
            ret['test'] = []
            for test_ratio in self.tvt_ratio[2]:
                test_len = (round(test_ratio * float(self.length))
                            if is_ratio else test_ratio)
                test_ids, rest_ids = random_split(
                    rest_ids, [test_len, len(rest_ids) - test_len]
                )
                ret['test'].append(Dataset(data, test_ids))
        return ret

    def data_splits( # pylint: disable=unused-argument
            self,
            data: Optional[Iterable[tuple]] = None,
            stage: Optional[str] = None
    ) -> Dict[str, Union[DatasetType, List[DatasetType]]]:
        """Split data from data_source for each dataloader

        Args:
            data: The data read by self.data_reader()
            stage: The stage argument same as the one from
                `LightningDataModule.setup(...)`

        Returns:
            A dictionary with keys `train`, `val` and `test`, and values a
            Dataset or an IterDataset (config.data_tvt will be ignored)

            Or if config.data_tvt is specified, one could just return an
            iterable of features, then the dataset will be automatically
            split by config.data_tvt
        """
        if not self.tvt_ratio:
            return None

        data = data or self.data

        if isinstance(data, GeneratorType):
            return self._split_data_generator(data)

        return self._split_data_list(data)

    def prepare_data(self, *args, **kwargs) -> None:
        """Prepare data"""
        logger.info('Reading data ...')
        self.data = self.data_reader()

    def setup(self, stage: Optional[str] = None) -> None:
        """Setup data"""
        if stage == 'fit':
            # Only do it once.
            # If you want it to be separate
            # redefine this method
            logger.info('Splitting data ...')

            self.splits = self.data_splits(self.data, stage)
            if not self.tvt_ratio and not self.splits:
                raise PlkitDataException(
                    'No train-val-test ratio (data-tvt) specified in '
                    'configuration, then `data_splits` method should be '
                    'implemented for DataModule.'
                )

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        """Train data loaders"""
        if 'train' not in self.splits:
            return None

        return DataLoader(self.splits['train'],
                          batch_size=self.config.batch_size,
                          num_workers=self.num_workers)

    def val_dataloader(self,
                       *args,
                       **kwargs) -> Union[DataLoader, List[DataLoader]]:
        """Validation data loaders"""
        if 'val' not in self.splits:
            return None
        ret = []
        for val_data in self.splits['val']:
            ret.append(DataLoader(val_data,
                                  batch_size=self.config.batch_size,
                                  num_workers=self.num_workers))
        return ret[0] if len(ret) == 1 else ret

    def test_dataloader(self,
                        *args,
                        **kwargs) -> Union[DataLoader, List[DataLoader]]:
        """Test data loaders"""
        if 'test' not in self.splits:
            return None
        ret = []
        for test_data in self.splits['test']:
            ret.append(DataLoader(test_data,
                                  batch_size=self.config.batch_size,
                                  num_workers=self.num_workers))
        return ret[0] if len(ret) == 1 else ret
