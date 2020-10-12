There are two runners supported: `LocalRunner` and `SGERunner`.

The first one submits the job locally and uses local CPUs/GPUs, and the second one uses grid CPUs/GPUs.

## Local runner
It is easy to initialize a local runner: `runner = LocalRunner()`

## SGE runner
But to initialize an sge runner, you need pass in a couple of arguments, which will be translated as command line arguments for `qsub`.

For example:

```python
from plkit import SGERunner

sge = SGERunner(o='/path/to/stdout')
# will be calling qsub like this:
# $ qsub -o /path/to/stdout
```

You can also specify the path to the `qsub` executable and the path to a `workdir` for the runner to save the outputs, errors and scripts for each jobs:

!!! note

    The job is submitted in background, and the main process will be then streaming the stdandard output from the job. You may have to end the program yourself after it's done.

```python
sge = SGERunner(qsub='/path/to/qsub', workdir='/path/to/workdir')
```
