import gzip
import os
from pathlib import Path
from typing import Generator, List, Optional


def concat_dirs(dirname: str, basename: str) -> str:
    """
    Append basename to dirname, avoiding backslashes when running on windows.

    os.path.join(dirname, basename) will add a backslash before dirname if
    basename does not end with a slash, so we make sure it does.
    """
    dirname += "/" if dirname[-1] != "/" else ""
    return os.path.join(dirname, basename)


def smart_open(file: str, mode: str = "rt", encoding: str = "utf-8"):
    """Convenience function for reading compressed or plain text files.
    :param file: The file to read.
    :param mode: The file mode (read, write).
    :param encoding: The file encoding.
    """
    if file.endswith(".gz"):
        return gzip.open(file, mode=mode, encoding=encoding, newline="\n")
    return open(file, mode=mode, encoding=encoding, newline="\n")


def add_files(
    dir_dataset: str,
    exclude: Optional[List] = None,
    recursive: bool = False,
    num_files_limit: Optional[int] = None,
) -> List[Path]:
    dir_dataset = Path(dir_dataset)
    all_files, rejected_files = set(), set()

    if exclude is not None:
        for excluded_pattern in exclude:
            if recursive:
                # Recursive glob
                for file in dir_dataset.rglob(excluded_pattern):
                    rejected_files.add(Path(file))
            else:
                # Non-recursive glob
                for file in dir_dataset.glob(excluded_pattern):
                    rejected_files.add(Path(file))

    file_refs: Generator[Path, None, None]
    if recursive:
        file_refs = Path(dir_dataset).rglob("*")
    else:
        file_refs = Path(dir_dataset).glob("*")

    for ref in file_refs:
        is_dir = ref.is_dir()
        skip_because_excluded = ref in rejected_files
        if not is_dir and not skip_because_excluded:
            all_files.add(ref)

    new_input_files = sorted(all_files)

    if len(new_input_files) == 0:
        raise ValueError(f"No files found in {dir_dataset}.")

    if num_files_limit is not None and num_files_limit > 0:
        new_input_files = new_input_files[0:num_files_limit]

    # print total number of files added
    print(f"> Total files added from {dir_dataset}: {len(new_input_files)}")
    return new_input_files
