import time
import traceback
from dataclasses import dataclass
from itertools import islice
from typing import Any, Callable, Generator, Iterable, List, Optional, Type, Union


def iter_batch(iterable: Union[Iterable, Generator], size: int) -> Iterable:
    """Iterate over an iterable in batches.

    >>> list(iter_batch([1,2,3,4,5], 3))
    [[1, 2, 3], [4, 5]]
    """
    source_iter = iter(iterable)
    while source_iter:
        b = list(islice(source_iter, size))
        if len(b) == 0:
            break
        yield b


def get_tqdm_iterable(items: Iterable, show_progress: bool, desc: str) -> Iterable:
    """Optionally get a tqdm iterable. Ensures tqdm.auto is used."""
    _iterator = items
    if show_progress:
        try:
            from tqdm import tqdm

            return tqdm(items, desc=desc)
        except ImportError:
            pass
    return _iterator


@dataclass
class ErrorToRetry:
    """Exception types that should be retried.

    Args:
        exception_cls (Type[Exception]): Class of exception.
        check_fn (Optional[Callable[[Any]], bool]]):
            A function that takes an exception instance as input and returns
            whether to retry.

    """

    exception_cls: Type[Exception]
    check_fn: Optional[Callable[[Any], bool]] = None


def retry_on_exceptions_with_backoff(
    lambda_fn: Callable,
    errors_to_retry: List[ErrorToRetry],
    max_tries: int = 10,
    min_backoff_secs: float = 0.5,
    max_backoff_secs: float = 60.0,
) -> Any:
    """Execute lambda function with retries and exponential backoff.

    The function is copied from llama_index.core.utils.py

    Args:
        lambda_fn (Callable): Function to be called and output we want.
        errors_to_retry (List[ErrorToRetry]): List of errors to retry.
            At least one needs to be provided.
        max_tries (int): Maximum number of tries, including the first. Defaults to 10.
        min_backoff_secs (float): Minimum amount of backoff time between attempts.
            Defaults to 0.5.
        max_backoff_secs (float): Maximum amount of backoff time between attempts.
            Defaults to 60.

    """
    if not errors_to_retry:
        raise ValueError("At least one error to retry needs to be provided")

    error_checks = {
        error_to_retry.exception_cls: error_to_retry.check_fn
        for error_to_retry in errors_to_retry
    }
    exception_class_tuples = tuple(error_checks.keys())

    backoff_secs = min_backoff_secs
    tries = 0

    while True:
        try:
            return lambda_fn()
        except exception_class_tuples as e:
            traceback.print_exc()
            tries += 1
            if tries >= max_tries:
                raise
            check_fn = error_checks.get(e.__class__)
            if check_fn and not check_fn(e):
                raise
            time.sleep(backoff_secs)
            backoff_secs = min(backoff_secs * 2, max_backoff_secs)
