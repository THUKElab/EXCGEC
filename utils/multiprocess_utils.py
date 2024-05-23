import math
from multiprocessing import Pool
from typing import Any, Callable, List


def multiprocess_helper(func: Callable, args: Any, num_worker: int = 1) -> List[Any]:
    split_input = args[0]
    step = math.ceil(len(split_input) / num_worker)

    returns, results = [], []
    with Pool(processes=num_worker) as p:
        for i in range(0, len(split_input), step):
            results.append(p.apply_async(func, (split_input[i : i + step], *args[1:])))
        for res in results:
            returns.append(res.get())
    return returns
