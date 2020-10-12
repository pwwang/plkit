`plkit` integrates the `optuna` framework to tune the hyperparameters. You don't have to modify your code to much to enable it by following our [best practice][1].

To enable optuna, you just need to pass an optuna object to `plkit.run` or `runner.run` method.

To initialize an optuna object, you need to pass in following arguments:

- `on`: telling `plkit` which metric to optimize on.
  You need to log that metric inside `validation_epoch_end` for the object to fetch it.
- `n_trials`: How many trials you want to run
- `**kwargs`: Other arguments for `optuna.create_study` (see [optuna.study.create_study][2]).

## OptunaSuggest

Defines how the parameters should be tuning. You can specify an OptunaSuggest object to a parameter in configuration like this:

```python
configuration = {
    #                             default, type, low, high
    'learning_rate': OptunaSuggest(1e-3, 'float', 1e-5, 5e-2)
}
```

This way, yon don't have to modify the configuration when you want to enable or disable optuna integration.

When optuna is enabled, a value will be generated for `learning_rate` using `trial.suggest_float('learning_rate', 1e-5, 5e-2)` and the default value is ignored. While it's disabled, it will fall back to the default value.

See all avaiable suggest types [here][3] and corresponding names (aliases) for the types as the second argument of `OptunaSuggest`

| Name/Alias | `optuna.trial.Trial.suggest_xxx`|
|-|-|
|`cat`|`suggest_categorical`|
|`categorical`|`suggest_categorical`|
|`distuni`|`suggest_discrete_uniform`|
|`dist_uni`|`suggest_discrete_uniform`|
|`discrete_uniform`|`suggest_discrete_uniform`|
|`float`|`suggest_float`|
|`int`|`suggest_int`|
|`loguni`|`suggest_loguniform`|
|`log_uni`|`suggest_loguniform`|
|`uni`|`suggest_uniform`|


[1]: ../home/best-practice
[2]: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.create_study.html#optuna.study.create_study
[3]: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial
