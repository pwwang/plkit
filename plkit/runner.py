"""Run jobs via non-local runners."""
import os
import sys
from uuid import uuid4
import cmdy
from .utils import logger

ENV_FLAG = "PLKIT_RUNNER"

class LocalRunner:

    def run(self, config, data_class, model_class):
        from . import run as pkrun
        pkrun(config, data_class, model_class)

class SGERunner(LocalRunner):
    """Run job on SGE by qsub"""
    def __init__(self, **opts):
        self.qsub = opts.get("qsub", "qsub")
        self.opts = {}
        for key, value in opts.items():
            if isinstance(value, dict):
                self.opts.update(value)
            else:
                self.opts[key] = value
        self.workdir = self.opts.pop("workdir", "./workdir")
        os.makedirs(self.workdir, exist_ok=True)

    def run(self, config, data_class, model_class):
        """Run the job depending on the env flag"""
        if not os.environ.get(ENV_FLAG):
            logger.info('Wrapping up the job ...')
            workdir = os.path.join(self.workdir, f'plkit-{uuid4()}')
            os.makedirs(workdir, exist_ok=True)
            logger.info('  - Workdir: %s', workdir)

            script = os.path.join(workdir, 'job.sh')
            logger.info('  - Script: %s', script)
            with open(script, 'w') as fscript:
                fscript.write("#!/bin/sh\n\n")
                cmd = cmdy._(*sys.argv, _exe=sys.executable).h.strcmd
                fscript.write(f"{ENV_FLAG}=1 {cmd}\n")

            opts = self.opts.copy()
            opts.setdefault('o', os.path.join(workdir, 'job.stdout'))
            opts.setdefault('cwd', True)
            opts.setdefault('j', 'y')
            opts.setdefault('notify', True)
            opts.setdefault('N', os.path.basename(workdir))

            logger.info('Submitting the job ...')
            cmd = cmdy.qsub(opts,
                            script,
                            _dupkey=True,
                            _prefix='-',
                            _exe=self.qsub)
            logger.info('  - %s', cmd.stdout.strip())

            cmdy.touch(opts['o'])
            logger.info('Streaming content from %s', opts['o'])
            cmdy.tail(f=True, _=opts['o']).fg()

        else:
            super().run(config, data_class, model_class)
