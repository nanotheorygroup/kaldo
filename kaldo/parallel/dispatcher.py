"""Dispatch with resume: the loop that's identical across kaldo's
parallel-with-disk-checkpointing call sites.

Why it exists: ``calculate_second``, ``calculate_third`` (FD), and
``_project_crystal`` (k-point projection) all do the same thing:
filter out units that already have a ``.done`` sentinel on disk,
submit the rest to a process pool, and wait for them to complete.
That loop was copy-pasted three times before this module existed.

What it does NOT do: choose the on-disk file format, assemble results
into the caller's output shape, or keep results in memory past the
``yield``. Workers write their own data; the helper only writes the
sentinel after the worker returns successfully (atomicity guarantee:
no sentinel without a complete result).
"""
import os
from concurrent.futures import as_completed

from kaldo.helpers.logger import get_logger
from kaldo.parallel.executor import get_executor

logging = get_logger()


def dispatch_with_resume(
    work_ids,
    worker_fn,
    *,
    n_workers,
    output_dir=None,
    sentinel_prefix="unit_",
    log_progress=True,
):
    """Dispatch ``worker_fn(unit_id)`` across workers with optional resume.

    Parameters
    ----------
    work_ids : iterable of int
        All unit identifiers the caller would dispatch in a fresh run.
        Integers only; used in ``f"{sentinel_prefix}{uid:05d}.done"``.
    worker_fn : callable
        Picklable callable taking a single ``unit_id`` argument and
        returning the per-unit result. If ``output_dir`` is set, the worker
        is responsible for writing its own output file under ``output_dir``
        before returning.
    n_workers : int or None
        Forwarded to ``get_executor``. ``1`` runs the serial backend;
        anything else picks the process backend.
    output_dir : str or os.PathLike or None
        Directory used both for resume detection (skipping any unit whose
        ``<sentinel_prefix>NNNNN.done`` already exists) and as the
        destination for the sentinel files. Created if missing.
    sentinel_prefix : str
        Prefix for the sentinel filename. Default ``"unit_"``.
    log_progress : bool
        When True, log one line per completed unit.

    Yields
    ------
    (unit_id, result) tuples in completion order.
    """
    work_ids = list(work_ids)
    n_total = len(work_ids)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        pending = [
            uid for uid in work_ids
            if not os.path.exists(
                os.path.join(output_dir, f"{sentinel_prefix}{uid:05d}.done")
            )
        ]
        n_resumed = n_total - len(pending)
        if n_resumed and log_progress:
            logging.info(
                f"Resuming: skipping {n_resumed} already-computed unit(s)"
            )
    else:
        pending = list(work_ids)

    use_parallel = n_workers is None or n_workers > 1
    backend = "process" if use_parallel else "serial"
    executor_workers = n_workers if use_parallel else None

    with get_executor(backend=backend, n_workers=executor_workers) as executor:
        futures = {executor.submit(worker_fn, uid): uid for uid in pending}
        for future in as_completed(futures):
            uid = futures[future]
            result = future.result()
            if output_dir is not None:
                # Sentinel only AFTER worker_fn returns. If a worker crashes
                # mid-write its .done file is missing and the next run
                # recomputes that unit — same contract the FD code had.
                open(
                    os.path.join(
                        output_dir, f"{sentinel_prefix}{uid:05d}.done"
                    ),
                    "w",
                ).close()
            if log_progress:
                logging.info(f"Completed unit {uid}")
            yield uid, result
