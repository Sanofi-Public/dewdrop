"""Model classes for wrapping Keras-based models into ALiEN format."""

import numpy as np

from ...config import INIT_SEED_INCREMENT, default_training_epochs
from ...decorators import flatten_batch, get_defaults_from_self, get_Xy
from ...utils import dict_pop, seed_all, shift_seed, update_copy
from ..mc_dropout import MCDropoutClassifier, MCDropoutModel, MCDropoutRegressor

# pylint: disable=import-outside-toplevel


class KerasModel(MCDropoutModel):  # , WrappedModel):
    """Base Class for wrapped Keras models."""

    def __init__(self, model=None, X=None, y=None, **kwargs):
        self.compile_kwargs = dict_pop(
            kwargs,
            "metrics",
            "loss_weights",
            "weighted_metrics",
            "run_eagerly",
            "steps_per_execution",
            "jit_compile",
            optimizer="adam",
            loss="mse",
        )
        self.fit_kwargs = dict_pop(
            kwargs,
            "batch_size",
            "verbose",
            "callbacks",
            "validation_split",
            "validation_data",
            "shuffle",
            "class_weight",
            "sample_weight",
            "initial_epoch",
            "steps_per_epoch",
            "validation_steps",
            "validation_batch_size",
            "validation_freq",
            "max_queue_size",
            "workers",
            "use_multiprocessing",
            epochs=default_training_epochs,
        )
        global tf  # pylint: disable=global-statement
        import tensorflow as tf

        self.initial_weights = None
        self.compiled = False
        self.model = model
        # self.training = training
        super().__init__(X=X, y=y, **kwargs)

    def fix_dropouts(self):
        from .utils import modify_dropout, subobjects

        for m in subobjects(self.model, skip=self.nodropout_layers):
            if modify_dropout(m):
                self.dropouts.append(m)

    @get_defaults_from_self
    def reinitialize(self, init_seed=None, sample_input=None):
        if init_seed is not None:
            seed_all(init_seed)
            self.init_seed = shift_seed(init_seed, INIT_SEED_INCREMENT)

        self.save_initial_weights(sample_input)

        # Actually reinitializing the weights in keras is almost impossible,
        # so instead we shuffle the saved initial weights (within each tensor),
        # giving the same distribution
        self.set_weights(
            [np.random.default_rng(init_seed).permutation(w.flat).reshape(w.shape) for w in self.initial_weights]
        )

    def get_weights(self):
        """Return model weights."""
        weights = [w.copy() for w in self.model.get_weights()]
        return None if len(weights) == 0 else weights

    @get_defaults_from_self
    def set_weights(self, initial_weights=None):
        """Set model weights according to initial_weights"""
        self.model.set_weights([w.copy() for w in initial_weights])

    def save_initial_weights(self, sample_input=None):
        """Save initial weights in self.initial_weights object."""
        if self.initial_weights is not None:
            return
        if sample_input is None:
            if self.X is None:
                raise ValueError("Can't initialize weights for the first time without sample input.")
            sample_input = self.X[:1]  # select first row
        self.predict(sample_input)
        self.initial_weights = self.get_weights().copy()

    @get_Xy
    @get_defaults_from_self
    def fit_model(self, X=None, y=None, **kwargs):
        # pylint: disable=undefined-variable
        if not self.compiled:
            self.model.compile(**self.compile_kwargs)
            self.compiled = True
        if not (isinstance(X, tf.Tensor) or isinstance(X, np.ndarray)):
            X, y = np.asarray(X), np.asarray(y)
        self.model.fit(x=X, y=y, **update_copy(self.fit_kwargs, **kwargs))

    @flatten_batch
    def predict(self, X, *args, **kwargs):
        return self.forward(X, *args, training=True, **kwargs)

    def _prepare_batch(self, X):
        # pylint: disable=undefined-variable
        if not isinstance(X, (tf.Tensor, np.ndarray)):
            X = np.asarray(X)
        return X

    def _forward(self, X, *args, **kwargs):
        preds = self.model(X, *args, **kwargs)
        if getattr(preds, "ndim", 1) == 2 and preds.shape[-1] == 1:
            preds = preds[:, 0]
        return preds.numpy()

    @flatten_batch
    def predict_samples(self, *args, n=None, **kwargs):
        import tensorflow as tf

        X = args[0]

        n = n or self.ensemble_size
        preds = []
        seed = self.rng.integers(int(1e6))  # single seed int

        if not (isinstance(X, tf.Tensor) or isinstance(X, np.ndarray)):
            X = np.asarray(X)

        for i in range(n):
            tf.random.set_seed(shift_seed(seed, i * 7))
            preds.append(self._forward(X, training=1))

        if isinstance(preds[0], tf.Tensor) or isinstance(preds[0], np.ndarray):
            return tf.stack(preds, axis=1)
        else:
            return zip(preds)


class KerasRegressor(KerasModel, MCDropoutRegressor):
    """Like the name"""


class KerasClassifier(KerasModel, MCDropoutClassifier):
    """Like the name"""
