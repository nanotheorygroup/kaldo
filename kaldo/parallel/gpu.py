"""GPU strategy layer for kaldo's TensorFlow-heavy operations.

Wraps ``tf.distribute`` strategies to provide a simple interface for
single-GPU, multi-GPU, and CPU-only execution of kaldo's anharmonic
calculations (einsum, sparse_dense_matmul, gather_nd, etc.).

This layer is orthogonal to the task executor — the executor distributes
work units (atoms, modes) across processes/nodes, while the GPU strategy
controls how TensorFlow operations within each work unit are placed on
devices.

Usage::

    from kaldo.parallel.gpu import GPUStrategy

    gpu = GPUStrategy('single-gpu')
    with gpu.scope():
        # TF operations here run on GPU
        result = tf.einsum('ij,jk->ik', a, b)

    # Or for multi-GPU distribution of per-mode calculations:
    gpu = GPUStrategy('multi-gpu')
    gpu.distribute_modes(mode_indices, compute_fn, shared_data)
"""

import warnings


class GPUStrategy:
    """Manages TensorFlow device placement for kaldo computations.

    Parameters
    ----------
    strategy : str
        One of:
        - ``'default'``: Use TF's default strategy (no-op, respects
          ``CUDA_VISIBLE_DEVICES``).
        - ``'single-gpu'``: Place ops on ``/GPU:0`` if available, else CPU.
        - ``'multi-gpu'``: Use ``tf.distribute.MirroredStrategy`` to
          replicate computation across all visible GPUs.
        - ``'cpu-only'``: Force all ops to CPU regardless of GPU availability.
    """

    def __init__(self, strategy='default'):
        self._strategy_name = strategy
        self._strategy = None  # Lazy init to avoid importing TF at module load

    def _init_strategy(self):
        """Lazily initialize the TF distribute strategy."""
        if self._strategy is not None:
            return

        import tensorflow as tf

        if self._strategy_name == 'default':
            self._strategy = tf.distribute.get_strategy()

        elif self._strategy_name == 'single-gpu':
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                self._strategy = tf.distribute.OneDeviceStrategy('/GPU:0')
            else:
                self._strategy = tf.distribute.get_strategy()

        elif self._strategy_name == 'multi-gpu':
            gpus = tf.config.list_physical_devices('GPU')
            if len(gpus) > 1:
                self._strategy = tf.distribute.MirroredStrategy()
            elif len(gpus) == 1:
                warnings.warn(
                    "Only 1 GPU available; 'multi-gpu' strategy falls back to "
                    "single-GPU. Use 'single-gpu' to suppress this warning.",
                    RuntimeWarning,
                    stacklevel=3,
                )
                self._strategy = tf.distribute.OneDeviceStrategy('/GPU:0')
            else:
                warnings.warn(
                    "No GPUs available; 'multi-gpu' strategy falls back to CPU.",
                    RuntimeWarning,
                    stacklevel=3,
                )
                self._strategy = tf.distribute.get_strategy()

        elif self._strategy_name == 'cpu-only':
            self._strategy = tf.distribute.OneDeviceStrategy('/CPU:0')

        else:
            raise ValueError(
                f"Unknown GPU strategy {self._strategy_name!r}. "
                "Choose from 'default', 'single-gpu', 'multi-gpu', 'cpu-only'."
            )

    def scope(self):
        """Return a context manager for TF operations under this strategy.

        Returns
        -------
        context : tf.distribute.Strategy.scope
            Context manager. TF variables and ops created inside this scope
            are placed according to the strategy.

        Example
        -------
        >>> gpu = GPUStrategy('single-gpu')
        >>> with gpu.scope():
        ...     result = tf.einsum('ij,jk->ik', a, b)
        """
        self._init_strategy()
        return self._strategy.scope()

    def distribute_modes(self, mode_indices, compute_fn, batch_size=100, **shared_data):
        """Distribute per-mode computation across GPUs.

        Splits ``mode_indices`` into batches of ``batch_size`` and distributes
        them across GPUs via the underlying ``tf.distribute`` strategy.

        Parameters
        ----------
        mode_indices : array-like
            Indices of phonon modes to compute (only physical modes).
        compute_fn : callable
            Function signature: ``compute_fn(mode_batch, **shared_data) -> results``.
            Called once per batch with a slice of mode indices.
        batch_size : int
            Number of modes per batch. Larger batches amortize GPU kernel
            launch overhead but use more memory. Default 100.
        **shared_data
            Read-only data broadcast to all GPUs (eigenvectors, frequencies,
            third-order FC, etc.).

        Returns
        -------
        results : list
            Collected results from all batches, in order.
        """
        self._init_strategy()
        all_results = []

        with self.scope():
            for batch_start in range(0, len(mode_indices), batch_size):
                batch = mode_indices[batch_start:batch_start + batch_size]
                batch_result = compute_fn(batch, **shared_data)
                all_results.append(batch_result)

        return all_results

    @property
    def num_replicas(self):
        """Number of GPU replicas in the current strategy."""
        self._init_strategy()
        return self._strategy.num_replicas_in_sync

    def __repr__(self):
        return f"GPUStrategy({self._strategy_name!r})"
