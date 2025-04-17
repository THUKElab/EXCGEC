"""
Post-process mT5 models' outputs

(1) 半角符号改全角符号。

"""

from typing import List, Union

HALF2FULL = [
    (",", "，"),
    (":", "："),
    (";", "；"),
    ("?", "？"),
    ("!", "！"),
    ("(", "（"),
    (")", "）"),
    ("......", "……"),
]


def post_process(lines: Union[str, List[str]]):
    if isinstance(lines, str):
        lines = [lines]
    new_lines = []
    for line in lines:
        for pair in HALF2FULL:
            line = line.replace(pair[0], pair[1])
        new_lines.append(line)
    return new_lines
