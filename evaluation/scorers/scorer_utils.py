from typing import List


def compute_prf(tp: float, fp: float, fn: float, beta: float = 0.5) -> float:
    p = float(tp) / (tp + fp) if fp else 1.0
    r = float(tp) / (tp + fn) if fn else 1.0
    f = float((1 + (beta**2)) * p * r) / (((beta**2) * p) + r) if p + r else 0.0
    return round(p, 4), round(r, 4), round(f, 4)


def compute_acc(tp: float, fp: float, fn: float, tn: float) -> float:
    acc = float(tp + tn) / (tp + fp + fn + tn) if tp + fp + fn + tn else 0.0
    return round(acc, 4)


def gt_numbers(nums_1: List[float], nums_2: List[float]) -> bool:
    if len(nums_1) != len(nums_2):
        raise ValueError("Unequal length of two lists")
    for num1, num2 in zip(nums_1, nums_2):
        if num1 > num2:
            return True
        elif num1 < num2:
            return False
    return False
