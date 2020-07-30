"""Optuna wrapper for plkit"""
from pytorch_lightning.callbacks import ModelCheckpoint
import optuna
from .trainer import Trainer
from .utils import log_config, logger

class OptunaSuggest:
    """Optuna suggests for configuration items


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
        """Get the suggested value"""
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

    Attributes
    """

    def __init__(self,
                 on,
                 n_trials,
                 **kwargs):
        self.on = on
        self.n_trials = n_trials
        self.study = optuna.create_study(**kwargs)
        # trainers, used for retrieve the best one
        self.trainers = []

    def _create_data(self, data_class, conf):
        return data_class(conf)

    def _create_model(self, model_class, conf):
        return model_class(conf)

    def _create_objective(self, config, data_class, model_class):

        data = self._create_data(data_class, config)
        def _objective(trial):
            suggested = self.suggests(trial, config)
            config_copy = config.copy()
            config_copy.update(suggested)

            log_config(suggested, "Tunable parameters")
            if 'batch_size' in config_copy:
                data.batch_size = config_copy['batch_size']

            model = self._create_model(model_class, config_copy)
            model.hparams.update(suggested)

            # expose filepath argument?
            checkpoint_callback = ModelCheckpoint(monitor=self.on)
            trainer = Trainer.from_config(
                config_copy,
                data=data,
                checkpoint_callback=checkpoint_callback
            )
            self.trainers.append(trainer)
            trainer.fit(model)
            return checkpoint_callback.best_model_score

        return _objective

    def suggests(self, trial, conf):
        """Collect the hyperparameters from the trial suggestions"""

        return {key: val.suggest(key, trial)
                for key, val in conf.items()
                if isinstance(val, OptunaSuggest)}

    def run(self, config, data_class, model_class, **kwargs):
        """Run the optimization"""
        objective = self._create_objective(config, data_class, model_class)
        self.study.optimize(objective, self.n_trials, **kwargs)
        logger.info('Testing if test_dataloader exists '
                    'using best trial: #%s', self.best_trial.number)
        self.best_trainer.test()

    optimize = run

    @property
    def best_params(self):
        """The best parameters from the study"""
        return self.study.best_params

    @property
    def best_trial(self):
        """The best trial from the study"""
        return self.study.best_trial

    @property
    def trials(self):
        """The trials"""
        return self.study.trials

    @property
    def best_trainer(self):
        """Get the best trainer"""
        return self.trainers[self.best_trial.number]

    @property
    def best_model(self):
        """Get the model from best trainer"""
        return self.best_trainer.model
