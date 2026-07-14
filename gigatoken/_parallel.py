"""Call-time detection of multiprocessing worker processes.

The Rust batch encodes fan out on a process-global rayon thread pool. Inside
a Python multiprocessing worker that is wasteful (every worker would size its
pool to all of the machine's cores) and, after os.fork, unsafe: a rayon pool
built before the fork has no worker threads in the child, so injecting work
into it waits forever. The batch methods therefore default to the sequential
Rust paths (which never touch the pool) when the current process is detected
to be a worker; passing an explicit ``parallel=`` overrides the detection.
"""

from __future__ import annotations

import multiprocessing
import os

_forked_child = False


def _mark_forked_child() -> None:
    global _forked_child
    _forked_child = True


if hasattr(os, "register_at_fork"):
    # Fires in the child of any os.fork(), including multiprocessing's
    # "fork" start method; "spawn" and "forkserver" children (fresh
    # interpreters, so no inherited rayon pool, but every worker would
    # still oversubscribe the cores) are caught by parent_process().
    os.register_at_fork(after_in_child=_mark_forked_child)


def in_worker_process() -> bool:
    """Whether this process is a multiprocessing worker or a forked child."""
    return _forked_child or multiprocessing.parent_process() is not None


def resolve_parallel(parallel: bool | None) -> bool:
    """Resolve a batch method's ``parallel`` argument: None means auto —
    parallel except inside a multiprocessing worker or forked child."""
    return not in_worker_process() if parallel is None else parallel
