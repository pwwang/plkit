"""Optuna wrapper for plkit"""
from diot import FrozenDiot
from torch import Tensor
from pytorch_lightning.callbacks import ModelCheckpoint
import optuna
from .trainer import Trainer
from .utils import log_config, logger, plkit_seed_everything

# supress optuna logging
optuna.logging._get_library_root_logger().handlers.clear()

class OptunaSuggest:
    """Optuna suggests for configuration items

    Args:
        default (any): The default value, which the value will be collapsed to
            when optuna is opted out. So that you don't have to change your
            code if you don't run optuna.
        suggtype (str): The type of suggestion
            For example, `cat` refers to `trial.suggest_categorical`
            The mappings are:
            cat -> 'suggest_categorical',
            categorical -> 'suggest_categorical',
            distuni -> 'suggest_discrete_uniform',
            dist_uni -> 'suggest_discrete_uniform',
            discrete_uniform -> 'suggest_discrete_uniform',
            float -> 'suggest_float',
            int -> 'suggest_int',
            loguni -> 'suggest_loguniform',
            log_uni -> 'suggest_loguniform',
            uni -> 'suggest_uniform'
        *args: The args used in `trial.suggest_xxx(name, *args, **kwargs)`
        **kwargs: The kwargs used in `trial.suggest_xxx(name, *args, **kwargs)`

    Attributes:
        default (any): The default from Args
        suggfunc (str): The transformed suggestion name according to `suggtype`
        args: *args from Args
        kwargs: **kwargs from Args
    """
    def __init__(self, default, suggtype, *args, **kwargs):
        self.default = default
        self.suggfunc = dict(
            cat='suggest_categorical',
            categorical='suggest_categorical',
            distuni='suggest_discrete_uniform',
            dist_uni='suggest_discrete_uniform',
            discrete_uniform='suggest_discrete_uniform',
            float='suggest_float',
            int='suggest_int',
            loguni='suggest_loguniform',
            log_uni='suggest_loguniform',
            uni='suggest_uniform'
        )[suggtype]
        self.args = args
        self.kwargs = kwargs

    def suggest(self, name, trial):
        """Get the suggested value

        This is used in Optuna class, you don't have to call this

        Args:
            name (str): The name of the parameter
            trial (optuna.Trial): The trial to get the suggested value from

        Returns:
            Any: The suggested value
        """
        return getattr(trial, self.suggfunc)(name, *self.args, **self.kwargs)


class Optuna:
    """The class uses optuna to automate hyperparameter tuning

    Example:
        >>> from plkit import Data, Module, Optuna
        >>> class MyData(Data):
        >>>     ...
        >>> class MyModel(Module):
        >>>     ...
        >>> class MyOptuna(Optuna):
        >>>     def suggests(self, config):
        >>>         ...
        >>>         return new_config
        >>> optuna = MyOptuna('val_loss', 100)
        >>> optuna.run(config, model_class, data_class)

    Args:
        on (str): On which value to optimize. Should be one of the keys of
            dictionary that is returned from `validation_epoch_end`.
            `val_acc` or `val_loss` for example.
        n_trials (int): Number of trials
        **kwargs: Other keyword arguments for `optuna.create_study`

    Attributes:
        on (str): on from Args
        n_trials (int): n_trials from Args
        study (optuna.Study): study object created from kwargs
        trainers (list): list of trainers to keep track of the best one
    """

    def __init__(self,
                 on,
                 n_trials,
                 **kwargs):
        self.on = on
        self.n_trials = n_trials
        self._best_trainer = None
        self._best_model = None
        self.study = optuna.create_study(**kwargs)
        # trainers, used for retrieve the best one
        self.trainers = []

    def _create_objective(self, config, data, model_class):
        """Create objective function for the study to optimize

        The objective function is built to return the best value from
        `validation_epoch_end` based on `self.on`. To implement this, a
        `ModelCheckpoint` callback is used and `best_model_score` is returned
        from it.

        Args:
            config (dict): The configuration dictionary
            data_class (class): The data class subclassed from `plkit.Data`
                Note it's the class itself, not instantiated object
            model_class (class): The model class subclassed from `plkit.Module`
                Note it's the class itself, not instantiated object

        Returns:
            callable: The objective function
        """
        def _objective(trial):
            logger.info('--------------------------------')
            logger.info('Start optuna trial #%s / %s',
                        len(self.trainers), self.n_trials - 1)
            logger.info('--------------------------------')
            suggested = self.suggests(trial, config)

            config_copy = config.copy()

            with config_copy.thaw():
                config_copy.update(suggested)

            log_config(suggested, "Tunable parameters")

            model = model_class(config_copy)
            model.hparams.update(suggested)

            # expose filepath argument?
            checkpoint_callback = ModelCheckpoint(monitor=self.on)
            trainer = Trainer.from_config(
                config_copy,
                checkpoint_callback=checkpoint_callback
            )

            trainer.fit(model, data)
            best_score = checkpoint_callback.best_model_score

            self.trainers.append((checkpoint_callback.best_model_path,
                                  config_copy,
                                  best_score))

            logger.info('')
            logger.info("'Optuna': trial #%s done with  %s = %s "
                        "and parameters: %s",
                        trial.number,
                        self.on,
                        best_score,
                        trial.params)
            number, score, params = self._current_best
            params = {key: val for key, val in params.items()
                      if key in suggested}
            logger.info("'Optuna': the best is #%s with %s = %s "
                        "and parameters: %s",
                        number,
                        self.on,
                        score,
                        params)
            logger.info('')
            return best_score

        return _objective

    @property
    def _current_best(self):
        """Get the current best trial number and parameters when the tuning
        is incomplete"""
        func = (max
                if self.study.direction == optuna.study.StudyDirection.MAXIMIZE
                else min)
        values = [trainer[2] if not isinstance(trainer[2], Tensor)
                  else trainer[2].cpu()
                  for trainer in self.trainers]
        number = values.index(func(values))
        trainer = self.trainers[number]
        return number, trainer[2], trainer[1]

    def suggests(self, trial, conf):
        """Collect the hyperparameters from the trial suggestions
        if any configuration item is an `OptunaSuggest` object

        Args:
            trial (optuna.Trial): the trial object
            conf (dict): The configuration dictionary

        Returns:
            dict: A dictionary of suggested parameters
        """

        return {key: val.suggest(key, trial)
                for key, val in conf.items()
                if isinstance(val, OptunaSuggest)}

    def run(self, config, data_class, model_class, **kwargs):
        # pylint: disable=line-too-long
        """Run the optimization

        The optimization is running on fit of the trainer. If test data is
        provided. Test will be performed as well.

        Args:
            config (dict): The configuation dictionary
            data_class (class): The data class subclassed from `plkit.Data`
                Note that this is the class itself, not the instantized object.
            model_class (class): The data class subclassed from `plkit.Module`
                Note that this is the class itself, not the instantized object.
            **kwargs: Other arguments for `study.optimize` other than
                `func` and `n_trials`.
                See: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study.optimize
        """
        # pylint: enable=line-too-long
        if not isinstance(config, FrozenDiot):
            config = FrozenDiot(config)

        plkit_seed_everything(config)

        data = data_class(config=config)
        objective = self._create_objective(config, data, model_class)
        self.study.optimize(objective, self.n_trials, **kwargs)

        self._best_trainer = Trainer.from_config(
            self.trainers[self.best_trial.number][1]
        )

        self._best_model = model_class.load_from_checkpoint(
            self.trainers[self.best_trial.number][0],
            # https://github.com/PyTorchLightning/pytorch-lightning/issues/2550
            config=self.trainers[self.best_trial.number][1]
        )

        if hasattr(data, 'test_dataloader'):
            test_dataloaders = data.test_dataloader()
        else: # pragma: no cover
            test_dataloaders = None

        if test_dataloaders:
            logger.info('')
            logger.info('---------------------------------')
            logger.info('Testing using best trial: #%s', self.best_trial.number)
            logger.info('---------------------------------')
            self.best_trainer.test(self.best_model,
                                   test_dataloaders=test_dataloaders)

        return self.best_trainer

    optimize = run

    @property
    def best_params(self):
        """The best parameters from the study

        Returns:
            dict: A dictionary containing parameters of the best trial.
        """
        return self.study.best_params

    @property
    def best_trial(self):
        """The best trial from the study

        Returns:
            optuna.FrozenTrial: A FrozenTrial object of the best trial.
        """
        return self.study.best_trial

    @property
    def trials(self):
        """The trials

        Returns:
            list: A list of FrozenTrial objects.
        """
        return self.study.trials

    @property
    def best_trainer(self):
        """Get the best trainer

        Returns:
            Trainer: The trainer object of the best trial.
        """
        return self._best_trainer

    @property
    def best_model(self):
        """Get the model from best trainer

        Returns:
            Module: The model of the best trial
        """
        return self._best_model
