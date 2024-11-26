import numpy as numpy

from .. import tumpy as tp
from ..decorators import get_defaults_from_self
from ..matrices import EnsembleMatrix
from ..models import Classifier, Regressor
from ..stats import entropy, joint_entropy
from ..utils import concatenate, default_max, default_min, diagonal
from .selector import SampleSelector, optimize_batch

MIN_STARTS = 4
DEFAULT_BUFFER_SIZE = 1000


class EntropySelector(SampleSelector):
    """
    Batch selection based on entropy of the joint distribution of predictions.
    Args:
        model: An instance of :class:`models.EntropyModel`. Will be used to
            determine prediction entropy for proposed batches. Alternatively,
            you can provide a precomputed joint entropy matrix via `joint_entropy`.
        samples: The sample pool to select from. Can be a numpy-style addressable
            array (with first dimension indexing samples, and other dimensions indexing
            features)---note that :class:`data.Dataset` serves this purpose---or an instance of
            :class:`sample_generation.SampleGenerator`, in which case the `num_samples` parameter is
            in effect.
        joint_entropy (optional): If you have precomputed the pairwise joint entropy
            of the sample pool, you can provide it here. This can save time.

        precompute_entropy: If `True`, the joint entropy matrix is computed before
            the selection process begins. If `False`, the joint entropy matrix is computed
            on the fly, only computing the rows it needs. *This can save memory* (and time
            too, if the sample pool is large). Default `True`.
        use_prob: If `True`, the model is assumed to be a classifier, and the
            independent probabilities of each class are used to compute the joint entropy.
            This tends to underestimate correlations, so we advise against it. Default False.

        See :class:`SampleSelector` for additional arguments.
    """

    def __init__(
        self,
        model=None,
        samples=None,
        num_samples=float("inf"),
        joint_entropy=None,
        batch_size=1,
        prior=1,
        prior_scale=1,
        prefilter=None,
        random_seed=None,
        #fast_opt=True,
        n_tuples=None,
        n_rounds=None,
        n_starts=None,
        parallel_starts=None,
        use_prob=None,
        precompute_entropy=True,
        buffer_size=None,
        use_buffer=None,
        **kwargs
    ):
        super().__init__(
            model=model,
            samples=samples,
            num_samples=num_samples,
            batch_size=batch_size,
            prior=prior,
            prior_scale=prior_scale,
            prefilter=prefilter,
            **kwargs
        )
        self.joint_entropy = joint_entropy

        if n_starts is not None:
            n_tuples = default_max(n_tuples, n_starts)
        if buffer_size is None and n_tuples is None and use_buffer:
            buffer_size = DEFAULT_BUFFER_SIZE

        self.n_tuples = n_tuples or default_max(
            n_starts, int(buffer_size // batch_size) if buffer_size else None, MIN_STARTS
        )
        self.buffer_size = default_max(buffer_size, n_tuples * batch_size if n_tuples else None)
        self.n_starts = n_starts
        self.parallel_starts = default_min(parallel_starts, n_starts)

        self.n_rounds = n_rounds
        self.fast_opt = True
        self.random_seed = random_seed
        self.use_prob = use_prob
        self.precompute_entropy = precompute_entropy
        self.buffer_size = buffer_size

    @get_defaults_from_self
    def _select(  # NOSONAR
        self,
        samples=None,
        fixed_samples=None,
        joint_entropy=None,
        batch_size=None,
        prior=None,
        #fixed_prior=None,
        verbose=None,
        **kwargs
    ):
        if samples is None or self.model is None:
            assert joint_entropy is not None
        if fixed_samples is not None:
            if isinstance(fixed_samples, int):
                n_fixed = fixed_samples
            else:
                n_fixed = len(fixed_samples)
                samples = concatenate(fixed_samples, samples)
            if prior is not None and not isinstance(prior, float):
                if len(prior) == len(samples):
                    prior[:n_fixed] = 1
                else:
                    prior = concatenate(tp.ones(
                        n_fixed, 
                        dtype=prior.dtype, 
                        device=tp.device(prior)),
                        prior)
        else:
            n_fixed = 0

        # Pairwise joint entropy
        joint_entropy = self.get_joint_entropy(
            joint_entropy=joint_entropy,
            model=self.model,
            samples=samples,
            prior=prior,
            verbose=verbose,
        )

        def jen_fn(indices):
            """Returns the approximate joint entropy of the batches indexed by `indices`."""
            if indices.shape[-1] == 1:
                return joint_entropy.diagonal()[indices[..., None]]

            # select sub-matrix of joint entropy matrix
            i_0, i_1 = tp.broadcast_arrays(indices[..., None, :], indices[..., :, None])
            jen_batch = joint_entropy[i_0, i_1]
            assert jen_batch.shape == (*indices.shape, indices.shape[-1])

            return jen_batch.sum(axis=(-1, -2))
            # TODO: Profile this, or the alternative without tril_indices
            # return jen_batch[..., tp.tril_indices(indices.shape[-1], -1)].sum(axis=-1)

        def jen_fn_opt_step(indices, samples=None):
            """Returns the information benefit of adding a given sample to the batch."""
            # select rows of entropy matrix
            jen_rows = joint_entropy[indices, :]

            # sum rows
            rv =  jen_rows.sum(axis=-2)
            #breakpoint()
            return rv

        def debug_callback(round, i, batch):
            #breakpoint()
            assert batch.shape[0] == 1
            batch = batch.squeeze()
            print(f'\nRound {round}, place {i}')
            print(batch)
            print(joint_entropy[batch[:,None], batch[None,:]], flush=True)

        if verbose:
            print("Optimizing batches...")
        batch_indices = optimize_batch(
            jen_fn,
            batch_size,
            samples=len(joint_entropy),
            n_fixed=n_fixed,
            scoring_opt_step=jen_fn_opt_step if self.fast_opt else None,
            n_rounds=self.n_rounds,
            random_seed=numpy.random.default_rng(self.random_seed).integers(1e8),
            n_tuples=self.n_tuples,
            n_starts=self.n_starts,
            parallel_starts=self.parallel_starts,
            verbose=verbose,
            scoring_capacity=self.buffer_size,
            #callback=debug_callback,
        )

        return batch_indices
    
    def get_joint_entropy(self, joint_entropy=None, model=None, samples=None, prior=None, verbose=None):
        if self.precompute_entropy or joint_entropy is not None:
            if joint_entropy is None:
                if verbose:
                    print("Computing pairwise joint entropies...")
                joint_entropy = self.model.joint_entropy(
                    samples,
                    block_size=int(numpy.sqrt(self.compute_batch_size)) if self.compute_batch_size else None,
                    pbar=verbose and bool(self.compute_batch_size),
                    **({"use_prob": self.use_prob} if isinstance(self.model, Classifier) else {})
                )
                if verbose:
                    print("  Done.")
            if isinstance(prior, numpy.ndarray):
                prior = tp.asarray(prior)
            if prior is not None:
                joint_entropy = prior[None, :] * joint_entropy * prior[:, None]
        else:
            joint_entropy = JointEntropyMatrix(
                self.model.predict_prob_or_class_samples(
                    samples, 
                    **({"use_prob": self.use_prob} if isinstance(self.model, Classifier) else {})),
                prior=prior,
                buffer_size=self.buffer_size,
            )
            self.buffer_size = joint_entropy.buffer_size

        return joint_entropy

    

class InfoFootprintSelector(EntropySelector):
    """
    Batch selection based on entropy of the joint distribution of predictions.
    Args:
        model: An instance of :class:`models.EntropyModel`. Will be used to
            determine prediction entropy for proposed batches. Alternatively,
            you can provide a precomputed joint entropy matrix via `joint_entropy`.
    """
    @get_defaults_from_self
    def _select(  # NOSONAR
        self,
        samples=None,
        fixed_samples=None,
        joint_entropy=None,
        batch_size=None,
        prior=None,
        fixed_prior=None,
        verbose=None,
        **kwargs
    ):
        if fixed_samples is not None:
            raise NotImplementedError("Fixed samples not yet implemented for entropy selection.")
        if samples is None or self.model is None:
            assert joint_entropy is not None
        if isinstance(fixed_samples, int):
            n_fixed = fixed_samples
        elif fixed_samples is not None:
            n_fixed = len(fixed_samples)
            samples = concatenate(fixed_samples, samples)
        else:
            n_fixed = 0

        # Pairwise joint entropy
        joint_entropy = self.get_joint_entropy(
            joint_entropy=joint_entropy,
            model=self.model,
            samples=samples,
            verbose=verbose,
        )

        diag = joint_entropy.diagonal()    # (N,)
        mut_info = diag[:, None] + diag[None, :] - joint_entropy   # (N, N)
        overlap = .5 * (mut_info / diag)    # diag varies along last axis
        overlap[:, diag==0] = 0.
        diagonal(overlap)[:] = 0.
        if prior is not None:
            footprint = prior * mut_info.sum(axis=-1)  # (N,)
        else:
            footprint = mut_info.sum(axis=-1)

        #breakpoint()

        def info_footprint(indices):
            """???"""
            if indices.shape[-1] == 1:
                return footprint[indices].sum(axis=-1)

            # select sub-matrix of joint entropy matrix
            i0, i1 = tp.broadcast_arrays(indices[..., None, :], indices[..., :, None])
            overlap_batch = overlap[i0, i1]   # (K, K)
            rescale = 1 - overlap_batch.sum(axis=-2)       # (K,)
            assert rescale.shape == indices.shape 

            batch_footprint = (footprint[indices] * rescale).sum(axis=-1)
            return batch_footprint
    
        def info_footprint_opt_step(indices, samples=None):  # indices: (n_starts, K)
            """Returns the information benefit of adding a given sample to the batch."""
            # overlap[indices]   :  (n_starts, K, N)
            # (1 - overlap[indices].sum(axis=-2)) * footprint   :  (n_starts, N)
            return (1 - overlap[indices].sum(axis=-2)) * footprint \
                - (tp.moveaxis(overlap[:, indices], 0, -1) * footprint[indices, None]).sum(axis=-2)

        def debug_callback(round, i, batch):
            #breakpoint()
            assert batch.shape[0] == 1
            batch = batch.squeeze()
            print(f'\nRound {round}, place {i}')
            print(batch)
            i0, i1 = tp.broadcast_arrays(batch[..., None, :], batch[..., :, None])
            overlap_batch = overlap[i0, i1]   # (K, K)
            rescale = 1 - overlap_batch.sum(axis=-2)       # (K,)
            print(footprint[batch])
            print(rescale)
            #print(mut_info[batch[:,None], batch[None,:]], flush=True)

        if verbose:
            print("Optimizing batches...")
        batch_indices = optimize_batch(
            info_footprint,
            batch_size,
            samples=len(joint_entropy),
            n_fixed=n_fixed,
            scoring_opt_step=info_footprint_opt_step if self.fast_opt else None,
            n_rounds=self.n_rounds,
            random_seed=numpy.random.default_rng(self.random_seed).integers(1e8),
            n_tuples=self.n_tuples,
            n_starts=self.n_starts,
            parallel_starts=self.parallel_starts,
            verbose=verbose,
            scoring_capacity=self.buffer_size,
            #callback=debug_callback,
        )
        return batch_indices        


class JointEntropyMatrix(EnsembleMatrix):
    def __init__(self, ensemble, prior=None, block_size=None, buffer_size=None, device=None):
        buffer_size = max(buffer_size or 0, ensemble.shape[0])
        super().__init__(
            ensemble,
            shape=2 * (ensemble.shape[0],),
            prior=prior,
            block_size=block_size,
            buffer_size=buffer_size,
            device=device,
        )

    def compute(self, indices, block_size=None):
        return joint_entropy(self.ensemble[indices], self.ensemble, block_size=block_size)

    def compute_diagonal(self, indices=None):
        if indices is None:
            return entropy(self.ensemble)
        return entropy(self.ensemble[indices])