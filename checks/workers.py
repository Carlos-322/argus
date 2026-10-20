"""Running one function over every episode on worker processes, so that a worker that dies costs its own episode only.

A worker process killed by a signal or by running out of memory breaks the whole process pool: the episode it was on
and every episode still waiting come back unfinished, not only the one that killed it. run_each runs those again, each
in a worker process of its own, and records an episode as errored only when its worker dies on that retry too. A check
that maps over a pool without this stops at the first dead worker, and every rerun stops the same way."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Callable, Iterable, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def run_each(fn: Callable[[T], R], items: Iterable[T], jobs: int, record: Callable[[R], None],
             died: Callable[[T, Exception], R]) -> list[T]:
    """Call record(fn(item)) for every item, fn running on a pool of jobs worker processes, each answer recorded as it
    finishes. The items a broken pool left unfinished run again, each in a worker process of its own, and an item whose
    worker fails on that retry too is recorded as record(died(item, error)). Every item is recorded exactly once.
    fn must be picklable (a module-level function or a functools.partial of one). Returns the items that were retried,
    in their original order."""
    items = list(items)
    lost = []
    with ProcessPoolExecutor(max(1, jobs)) as ex:
        futures = {ex.submit(fn, item): i for i, item in enumerate(items)}
        for fut in as_completed(futures):
            try:
                res = fut.result()
            except Exception:  # noqa: BLE001 - the pool broke: retried alone below
                lost.append(futures[fut])
                continue
            record(res)
    retry = [items[i] for i in sorted(lost)]
    if retry:
        print(f"retrying {len(retry)} episodes a worker's death left unfinished, each in a process of its own",
              flush=True)

        def alone(item: T) -> R:
            try:
                with ProcessPoolExecutor(1) as one:
                    return one.submit(fn, item).result()
            except Exception as e:  # noqa: BLE001 - its worker died again: recorded as errored, never skipped
                return died(item, e)

        with ThreadPoolExecutor(max(1, jobs)) as ex:
            for res in ex.map(alone, retry):
                record(res)
    return retry
