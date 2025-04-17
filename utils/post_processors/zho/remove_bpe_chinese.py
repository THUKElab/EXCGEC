import sys
from multiprocessing import Pool

from tqdm.auto import tqdm


def remove_bpe(line: str) -> str:
    line = line.strip().replace("##", "").replace(" ", "")
    return line


with Pool(64) as pool:
    for ret in pool.imap(remove_bpe, tqdm(sys.stdin), chunksize=1024):
        print(ret)
