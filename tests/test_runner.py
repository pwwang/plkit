import pytest

import os
from pathlib import Path
from threading import Thread

from diot import Diot
from plkit import *
from .test_optuna import Data, Model

@pytest.fixture
def config(tmp_path):
    return Diot(
        datadir=str(tmp_path),
        batch_size=32,
        max_epochs=5,
        # take a small set
        data_tvt=(300, 100, 100),
        learning_rate=OptunaSuggest(1e-4, 'float', 1e-5, 1e-3)
    )

def test_run(config):

    trainer = run(config, Data, Model)
    assert isinstance(trainer, Trainer)

def test_local_run(config):
    local = LocalRunner()
    trainer = local.run(config, Data, Model)
    assert isinstance(trainer, Trainer)

def test_sge_run(config, tmp_path):
    sge = SGERunner(qsub='echo', workdir=tmp_path/'workdir')

    thr = Thread(target=sge.run, args=(config, Data, Model), daemon=True)
    thr.start()
    thr.join(2)
    script = Path(sge.workdir) / f'plkit-{sge.uid}' / 'job.sge.sh'
    assert script.is_file()
    assert script.open().readline().startswith('#!/bin/sh')

def test_sge_run_env_true(config, tmp_path):
    sge = SGERunner(qsub='echo', workdir=tmp_path/'workdir')
    os.environ[sge.envname] = '1'
    trainer = sge.run(config, Data, Model)
    assert isinstance(trainer, Trainer)
