"""Wrapper for Catboost based models."""

import json
import os
from glob import glob

import numpy as np

from ..decorators import flatten_batch, get_defaults_from_self, get_Xy
from ..utils import dict_pop, softmax
from .models import (
    Classifier,
    EnsembleClassifier,
    EnsembleModel,
    EnsembleRegressor,
    Output,
    Regressor,
    get_output_type,
)

# pylint: disable=import-outside-toplevel

catboost_fit_kwargs = [
    "cat_features",
    "text_features",
    "embedding_features",
    "pairs",
    "sample_weight",
    "group_id",
    "group_weight",
    "subgroup_id",
    "pairs_weight",
    "baseline",
    "use_best_model",
    "eval_set",
    "verbose",
    "logging_level",
    "plot",
    "plot_file",
    "column_description",
    "verbose_eval",
    "metric_period",
    "silent",
    "early_stopping_rounds",
    "save_snapshot",
    "snapshot_file",
    "snapshot_interval",
    "init_model",
    "log_cout=sys.stdout",
    "log_cerr=sys.stderr",
]


def fitted_model(model, X, y, eval_set=None):
    """Fit a model according to features X and targets y."""
    model.fit(X, y, eval_set=eval_set)
    return model


class CatBoostModel(EnsembleModel):
    """
    Wraps the CatBoostModel class of of the CatBoost package.
    This is a decision-tree-based, gradient-boosting model, which
    incorporates advanced features such as probabilistic fitting
    (to capture the 'noise' or 'data uncertainty'), and most importantly,
    can sample models from the Bayesian posterior, which is ideal for
    generating ensembles of models for predicting knowledge (or
    'epistemic') uncertainty.

    :param X: Optionally, you can provide a reference to a training
        dataset, which will be stored, and called upon during
        fitting.
    :param y: reference to labels in training dataset. If X is given but
        y is not, then the inputs are X[:, :-1] and the labels
        are y = X[:, -1]
    :param random_seed: A random state initializer, defaults to None.

    :param kwargs: ...
        learning rate
        iterations

    """

    def __init__(
        self,
        X=None,
        y=None,
        val=None,
        n_models=None,
        ensemble_size=None,
        virtual_ensemble_limit=100,
        shape=None,
        random_seed=None,
        prefer_threads=False,
        n_jobs=1,
        task=None,
        loss_function=None,
        verbose=True,
        **kwargs,
    ):
        super().__init__(ensemble_size=ensemble_size, shape=shape, random_seed=random_seed, X=X, y=y, **kwargs)
        if n_models and ensemble_size and n_models != ensemble_size:
            raise ValueError("n_models and ensemble_size must be the same if both are given.")
        self.val = val
        self.virtual_ensemble_limit = virtual_ensemble_limit
        self.n_models = n_models if n_models else ensemble_size if ensemble_size else 10
        self.n_jobs = n_jobs
        self.prefer_threads = prefer_threads
        self.verbose = verbose
        self.fit_kwargs = dict_pop(kwargs, *catboost_fit_kwargs)
        self.ensemble = []
        if task is None:
            if isinstance(self, Regressor):
                task = "regression"
            elif isinstance(self, Classifier):
                task = "classification"
            else:
                assert task, "You must specify a task: 'regression' or 'classification'"
        self.loss_function = loss_function or ("MultiClass" if task.startswith("c") else "RMSE")
        self.task = task

    @get_Xy
    @get_defaults_from_self
    def fit_model(
        self,
        X=None,
        y=None,
        val=None,
        random_seed=None,
        verbose=True,
        **kwargs,
    ):
        """
        See models.Model.fit for details.

        :param kwargs: ...

        learning rate
        iterations
        """
        if self.task.startswith("r"):
            from catboost import CatBoostRegressor as CBModel
        elif self.task.startswith("c"):
            from catboost import CatBoostClassifier as CBModel
        from joblib import Parallel, delayed

        kwargs_copy = self.fit_kwargs.copy()
        kwargs_copy.update(kwargs)

        X, y = np.asarray(X), np.asarray(y)

        if "model_shrink_rate" not in kwargs_copy:
            kwargs_copy["model_shrink_rate"] = 1 / (2.0 * len(X))

        if random_seed is None:
            random_seed = self.rng.integers(1e8)

        self.ensemble = [
            CBModel(
                random_seed=random_seed,
                loss_function=self.loss_function,
                langevin=True,
                diffusion_temperature=len(X),
                verbose=False,
                **kwargs_copy,
            )
            for i in range(self.n_models)
        ]
        if self.n_jobs is None or self.n_jobs == 1:
            for i, model in enumerate(self.ensemble):
                if verbose:
                    print(f"Fitting model {i}...")
                model.fit(X, y, eval_set=val)
            if verbose:
                print("Done.")
        else:
            self.ensemble = Parallel(
                n_jobs=self.n_jobs,
                verbose=50 if verbose else 0,
                prefer="threads" if self.prefer_threads else "processes",
            )(delayed(fitted_model)(model, X, y, eval_set=val) for model in self.ensemble)
        self.trained = True

    @flatten_batch
    @get_defaults_from_self
    def predict_fixed_ensemble(self, X, virtual_ensemble_count=None, **kwargs):
        if not self.ensemble:
            raise ValueError("Can't make predictions, because model has not been trained yet.")

        X = np.asarray(X)

        # TODO: implement classfier Output variable returns (class, logprob, probs)
        if virtual_ensemble_count:
            prediction_type = kwargs.pop(
                "prediction_type", None
            )  # virtual ensembles only output RawFormulaVal (logits?)
            preds = [
                model.virtual_ensembles_predict(X, virtual_ensembles_count=virtual_ensemble_count, **kwargs)
                for model in self.ensemble
            ]
            rv = np.concatenate(preds, axis=1)
            if prediction_type == "Probability":
                rv = softmax(rv, -1)
            elif prediction_type == "Class":
                rv = rv.argmax(-1)
            return rv
        else:
            preds = [model.predict(X, **kwargs) for model in self.ensemble]
            return np.stack(preds, axis=1)

    def predict_samples(self, X, *args, n=None, use_virtual_ensembles=True, **kwargs):
        if not self.ensemble:
            raise ValueError("Can't make predictions, because model has not been trained yet.")

        n = n or self.ensemble_size
        if n <= self.n_models:
            return super().predict_samples(X, *args, n=n, ensemble_size=self.n_models, **kwargs)

        X = np.asarray(X)

        if not use_virtual_ensembles:
            raise ValueError(
                f"You are asking for {n} samples, but you only have an ensemble of"
                f"{self.n_models} trained models. If you set `use_virtual_ensembles=True`, then"
                f"you can get more than {self.n_models} samples."
            )
        elif n > self.n_models * self.virtual_ensemble_limit:
            raise ValueError(
                f"You are asking for {n} samples, but you only have an ensemble of"
                f"{self.n_models} trained models, and virtual_ensemble_limit = "
                f"{self.virtual_ensemble_limit}, giving a max ensemble of {self.n_models} x "
                f"{self.virtual_ensemble_limit}."
            )

        virtual_count = np.array([int(n // self.n_models)] * self.n_models)
        virtual_remain = n % self.n_models
        virtual_count[:virtual_remain] += 1

        preds = []
        prediction_type = kwargs.pop("prediction_type", None)
        for ve_count, model in zip(virtual_count, self.ensemble):
            if ve_count > 1:
                preds.append(model.virtual_ensembles_predict(X, *args, virtual_ensembles_count=ve_count, **kwargs))
            else:
                preds.append(model.predict(X, *args, prediction_type="RawFormulaVal", **kwargs)[:, None])
        rv = np.concatenate(preds, axis=1)
        if prediction_type == "Probability":
            rv = softmax(rv, -1)
        elif prediction_type == "Class":
            rv = rv.argmax(-1)
        return rv

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "params.json"), "w", encoding="UTF-8") as file:
            json.dump(
                {
                    "shape": self.shape,
                    "n_jobs": self.n_jobs,
                    "prefer_threads": self.prefer_threads,
                    "kwargs": self.fit_kwargs,
                    "task": self.task,
                },
                file,
            )

        ensemble_len = len(str(len(self.ensemble)))
        for i, model in enumerate(self.ensemble):
            model.save_model(os.path.join(path, f"model_{i:0{ensemble_len}d}.cbm"))

    @staticmethod
    def load(path):
        assert os.path.isdir(path), f"Local path doesn't exist: {path} "
        filenames = sorted(glob(os.path.join(path, "*.cbm")))

        with open(os.path.join(path, "params.json"), "r", encoding="UTF-8") as file:
            params = json.load(file)

        task = params.pop("task", "regression")
        if task.startswith("r"):
            from catboost import CatBoostRegressor as CBModel

            ALModel = CatBoostRegressor
        elif task.startswith("c"):
            from catboost import CatBoostClassifier as CBModel

            ALModel = CatBoostClassifier

        ensemble_model = ALModel(n_models=len(filenames), **params)
        ensemble_model.ensemble = [CBModel() for _ in filenames]
        for filename, model in zip(filenames, ensemble_model.ensemble):
            model.load_model(filename)

        ensemble_model.trained = True
        return ensemble_model


class CatBoostRegressor(EnsembleRegressor, CatBoostModel):
    """The name is descriptive"""

    @flatten_batch
    def predict(self, X, **kwargs):
        if not self.ensemble:
            raise ValueError("Can't make predictions, because model has not been trained yet.")
        return np.asarray([model.predict(np.asarray(X), **kwargs) for model in self.ensemble]).mean(axis=0)

    def _forward(self, X, *args, **kwargs):
        """Raises NameError."""
        raise NameError("This method is not implemented for CatBoostRegressor.")
    
    def _prepare_batch(self, X):
        """Raises NameError."""
        raise NameError("CatBoostRegressor does not support batching. Use fit instead.")


class CatBoostClassifier(EnsembleClassifier, CatBoostModel):
    """The name is descriptive"""

    PREDICT_FN_MAPPING = {
        Output.LOGIT: "RawFormulaVal",
        Output.PROB: "Probability",
        Output.CLASS: "Class",
    }

    @get_defaults_from_self
    def predict_fixed_ensemble(self, *args, output=None, **kwargs):
        if "prediction_type" not in kwargs:
            kwargs["prediction_type"] = self.PREDICT_FN_MAPPING[get_output_type(output)]
        return super().predict_fixed_ensemble(*args, **kwargs)

    @get_defaults_from_self
    def predict_samples(self, *args, output=None, **kwargs):
        if "prediction_type" not in kwargs:
            kwargs["prediction_type"] = self.PREDICT_FN_MAPPING[get_output_type(output)]
        return super().predict_samples(*args, **kwargs)

    def predict_logit_samples(self, *args, **kwargs):
        return self.predict_samples(*args, output=Output.LOGIT)

    def predict_prob_samples(self, *args, **kwargs):
        return self.predict_samples(*args, output=Output.PROB)

    def predict_class_samples(self, *args, **kwargs):
        return self.predict_samples(*args, output=Output.CLASS)

    @flatten_batch
    def predict(self, X, *args, output=None, **kwargs):
        preds = self.predict_fixed_ensemble(X, *args, output=Output.PROB, **kwargs).mean(axis=1)
        output = (
            get_output_type(output)
            if output is not None
            else getattr(self, "output", getattr(self, "wrapped_output", Output.CLASS))
        )
        if output == Output.PROB:
            return preds
        elif output == Output.LOGIT:
            return np.log(preds)
        # Output.CLASS
        return (preds > 0.5).astype(int) if (preds.ndim == 1 or preds.shape[-1] == 1) else np.argmax(preds, axis=1)

    def predict_logit(self, *args, **kwargs):
        return self.predict(*args, output=Output.LOGIT, **kwargs)

    def predict_prob(self, *args, **kwargs):
        return self.predict(*args, output=Output.PROB, **kwargs)

    def predict_class(self, *args, **kwargs):
        return self.predict(*args, output=Output.CLASS, **kwargs)

    def _forward(self, X, *args, **kwargs):
        """Raises NameError."""
        raise NameError("This method is not implemented for CatBoostClassifier.")
    
    def _prepare_batch(self, X):
        """Raises NameError."""
        raise NameError("CatBoostClassifier does not support batching. Use fit instead.")
