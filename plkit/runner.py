"""Run jobs via non-local runners."""
import os
import sys
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type
from diot import FrozenDiot
import cmdy
from .data import DataModule
from .module import Module
from .optuna import Optuna
from .trainer import Trainer
from .utils import logger, warning_to_logging, plkit_seed_everything

class Runner(ABC):
    """The base class for runner"""
    @abstractmethod
    def run(self,
            config: Dict[str, Any],
            data_class: Type[DataModule],
            model_class: Type[Module],
            optuna: Optional[Optuna] = None) -> Trainer:
        """Run the whole pipeline using the runner

        Args:
            config: A dictionary of configuration, must have following items:
                - batch_size: The batch size
                - num_classes: The number of classes for classification
                    1 means regression
            data_class: The data class subclassed from `Data`
            model_class: The model class subclassed from `Module`
            optuna: The optuna object
            runner: The runner object

        Returns:
            The trainer object
        """


class LocalRunner(Runner):
    """The local runner for the pipeline"""

    def run(self,
            config: Dict[str, Any],
            data_class: Type[DataModule],
            model_class: Type[Module],
            optuna: Optional[Optuna] = None) -> Trainer:
        """Run the pipeline locally"""
        if not isinstance(config, FrozenDiot):
            config = FrozenDiot(config)

        if optuna: # pragma: no cover
            return optuna.run(config, data_class, model_class)

        plkit_seed_everything(config)

        data = data_class(config=config)
        model = model_class(config)
        trainer = Trainer.from_config(config)
        with warning_to_logging():
            trainer.fit(model, data)

        if hasattr(data, 'test_dataloader'):
            test_dataloader = data.test_dataloader()
        else: # pragma: no cover
            test_dataloader = None

        if test_dataloader:
            with warning_to_logging():
                trainer.test(test_dataloaders=test_dataloader)
        return trainer

class SGERunner(LocalRunner):
    """The SGE runner for the pipeline

    Args:
        opts: The options for SGE runner, which will be translated as arguments
            for `qsub`. For example `opts={'notify': True}` will be translated
            as `qsub --notify ...` from command line.

            there are two special options `qsub` and `workdir`. `qsub` specified
            the path to `qsub` executable and `workdir` specifies a location to
            save outputs, errors and scripts of each job.

    Attributes:
        qsub: The path to qsub executable
        workdir: The path to the workdir
    """

    ENV_FLAG_PREFIX = "PLKIT_SGE_RUNNER_"

    def __init__(self, *args, **opts):
        self.qsub = opts.pop("qsub", "qsub") # type: str
        self.workdir = opts.pop("workdir", "./workdir") # type: str
        os.makedirs(self.workdir, exist_ok=True)

        self.args = args
        self.opts = opts
        self.uid = uuid.uuid5(uuid.NAMESPACE_DNS, str(sys.argv))
        self.envname = SGERunner.ENV_FLAG_PREFIX + str(self.uid).split('-')[0]


    def run(self,
            config: Dict[str, Any],
            data_class: Type[DataModule],
            model_class: Type[Module],
            optuna: Optional[Optuna] = None) -> Trainer:
        """Run the job depending on the env flag"""
        if not os.environ.get(self.envname):
            logger.info('Wrapping up the job ...')
            workdir = os.path.join(self.workdir, f'plkit-{self.uid}')
            os.makedirs(workdir, exist_ok=True)
            logger.info('  - Workdir: %s', workdir)

            script = os.path.join(workdir, 'job.sge.sh')
            logger.info('  - Script: %s', script)
            with open(script, 'w') as fscript:
                fscript.write("#!/bin/sh\n\n")
                cmd = cmdy._(*sys.argv, _exe=sys.executable).h.strcmd
                fscript.write(f"{self.envname}=1 {cmd}\n")

            opts = self.opts.copy()
            opts.setdefault('o', os.path.join(workdir, 'job.stdout'))
            opts.setdefault('cwd', True)
            opts.setdefault('j', 'y')
            opts.setdefault('notify', True)
            opts.setdefault('N', os.path.basename(workdir))

            logger.info('Submitting the job ...')
            cmd = cmdy.qsub(*self.args,
                            opts,
                            script,
                            cmdy_dupkey=True,
                            cmdy_prefix='-',
                            cmdy_exe=self.qsub).h()
            logger.info('  - Running: %s', cmd.strcmd)
            logger.info('  - %s', cmd.run().stdout.strip())

            cmdy.touch(opts['o'])
            logger.info('Streaming content from %s', opts['o'])
            cmdy.tail(f=True, _=opts['o']).fg()
            return None # pragma: no cover

        return super().run(config, data_class, model_class, optuna)
