import copy
from itertools import chain
from typing import Dict, List, Tuple

from data import Chunk, Edit
from utils import get_logger

LOGGER = get_logger(__name__)


def chunk_list_to_text(chunks: List[Chunk], limiter=" ") -> Tuple[str, str]:
    src = limiter.join([limiter.join(x.src_tokens) for x in chunks])
    tgt = limiter.join([limiter.join(x.tgt_tokens) for x in chunks])
    return src, tgt


def chunk_src_list_to_text(chunks: List[Chunk], chunk_index: int, limiter=" ") -> str:
    tokens = []
    for idx, chunk in enumerate(chunks):
        if idx == chunk_index:
            tokens.extend(chunk.tgt_tokens)
        else:
            tokens.extend(chunk.src_tokens)
    return limiter.join(tokens)


def all_correct(chunk_list: List[Chunk]):
    """Whether all chunks change the original text."""
    if not set([x.chunk_index for x in chunk_list]) != 1:
        raise ValueError("Input chunks must have the same chunk index: {chunk_list}")
    return all([x.types for x in chunk_list])


def any_correct(chunk_list: List[Chunk]):
    """Whether there exist chunks change the original text."""
    if not set([x.chunk_index for x in chunk_list]) != 1:
        raise ValueError("Input chunks must have the same chunk index: {chunk_list}")
    return any([x.types for x in chunk_list])


def hit_operation(chunk_list: List[Chunk], hyp_chunk: Chunk):
    """ """
    for chunk in chunk_list:
        # 如果是1就返回true，只要有一个是true就说明需要修改  # 必要编辑并进行了纠这个动作
        if chunk.types and hyp_chunk.types:
            return True
    return False


def map_parallel(src_tokens: List[str], edit_list: List[Edit]) -> Dict[int, int]:
    """Map source indices to target indices."""
    token_mapping = {}
    src_idx, tgt_idx = 0, 0

    for edit in edit_list:
        token_mapping[edit.src_interval[0]] = edit.tgt_interval[0]
        token_mapping[edit.src_interval[1]] = edit.tgt_interval[1]
        while src_idx < edit.src_interval[0]:
            token_mapping[src_idx] = tgt_idx
            src_idx += 1
            tgt_idx += 1
        src_idx = edit.src_interval[1]
        tgt_idx = edit.tgt_interval[1]

    while src_idx <= len(src_tokens):
        token_mapping[src_idx] = tgt_idx
        src_idx += 1
        tgt_idx += 1
    return token_mapping


def merge_edit(
    src_sent: List[str],
    tgt_list: List[List[str]],
    edits_list: List[List[Edit]],
    token_mapping_total: List[Dict[int, int]],
    merge_distance: int = 0,
) -> Tuple[List[List[Edit]], List[List[int]]]:
    """Merge edits with overlapping or adjacent intervals into a single chunk.

    Args:
        src_sent (List[str]): Segmented source sentence
        tgt_list (List[List[str]]): Segmented target sentences
        edits_list (List[List[Edit]]): Edits given by target sentences
        token_mapping_total (List[Dict[int, int]]): Token mapping given by target sentences
        merge_distance (int, optional): Distance of merging edits. Defaults to 0.

    Returns:
        Tuple[List[List[Edit]], List[List[int]]]:
            merge_edits_list: Merged edits for targets
            shared_interval_list: Shared edit intervals for targets
    """
    edits_flat = list(chain(*edits_list))
    sorted_edit_list = sorted(edits_flat, key=lambda x: x.src_interval[0])

    # Merge interval
    shared_interval_list = []
    for edit in sorted_edit_list:
        if (
            shared_interval_list
            and edit.src_interval[0] <= shared_interval_list[-1][1] + merge_distance
        ):
            # overlap
            shared_interval_list[-1][1] = max(
                edit.src_interval[1], shared_interval_list[-1][1]
            )
        else:
            # non-overlap
            shared_interval_list.append(list(edit.src_interval))
    LOGGER.debug(f"shared_interval_list: {shared_interval_list}")

    def get_shared_interval(interval):
        for i in shared_interval_list:
            if interval[0] >= i[0] and interval[1] <= i[1]:
                return i
        return None

    # Construct token_mapping: src_token -> tgt_token
    merge_edits_list = []
    for tgt_idx in range(len(tgt_list)):
        token_mapping = token_mapping_total[tgt_idx]
        LOGGER.debug(f"{' '.join(src_sent)} || {' '.join(tgt_list[tgt_idx])}")
        LOGGER.debug(f"token_mapping[{tgt_idx}]={token_mapping}")

        for edit in edits_list[tgt_idx]:
            ori_edit = copy.deepcopy(edit)
            shared_interval = get_shared_interval(edit.src_interval)
            assert shared_interval is not None

            edit.src_interval = shared_interval

            if shared_interval[0] == 0:
                tgt_beg_idx = 0
            else:
                tgt_beg_idx = token_mapping[shared_interval[0] - 1] + 1

            tgt_end_idx = token_mapping[shared_interval[1]]
            edit.tgt_interval = [tgt_beg_idx, tgt_end_idx]
            edit.src_tokens = src_sent[shared_interval[0] : shared_interval[1]]
            edit.tgt_tokens = tgt_list[tgt_idx][tgt_beg_idx:tgt_end_idx]
            LOGGER.debug(f"Merge edits: {ori_edit} -> {edit}")

        # Deduplicate edits
        merge_edits = []
        for edit in edits_list[tgt_idx]:
            if edit not in merge_edits:
                merge_edits.append(edit)
            else:
                merge_edits[-1].types.extend(edit.types)
                LOGGER.debug(f"Merge the same edits: {merge_edits[-1]}")
        merge_edits_list.append(merge_edits.copy())
        LOGGER.debug(f"merge_edits: {merge_edits}")

    return merge_edits_list, shared_interval_list


def convert_edit_into_chunk(
    src_sent: List[str],
    tgt_sent_list: List[List[str]],
    merge_edits_list: List[List[Edit]],
    shared_interval_list: List[List[int]],
    token_mapping_total: List[Dict[int, int]],
) -> List[List[Chunk]]:
    """Convert edits to chunks

    Args:
        src_sent (List[str]): Segmented source sentence
        tgt_sent_list (List[List[str]]): Segmented target sentences
        merge_edits_list (List[List[Edit]]): Merged edits for targets
        shared_interval_list (List[List[int]]): Shared edit intervals for targets
        token_mapping_total (List[Dict[int, int]]): Token mapping given by target sentences

    Returns:
        List[List[Chunk]]: Chunk sequences
    """
    valid_interval_list = []
    src_idx, interval_idx = 0, 0

    # Acquire chunk intervals
    while src_idx <= len(src_sent):
        if interval_idx == len(shared_interval_list):
            if src_idx != len(src_sent):
                valid_interval_list.append([src_idx, len(src_sent)])
            break

        curr_interval = shared_interval_list[interval_idx]
        if src_idx == curr_interval[0] == curr_interval[1]:
            # Dummy Chunk
            valid_interval_list.append(curr_interval.copy())
            interval_idx += 1
        elif curr_interval[0] <= src_idx < curr_interval[1]:
            # Corrected Chunk
            valid_interval_list.append(curr_interval.copy())
            src_idx = curr_interval[1]
            interval_idx += 1
        else:
            # Unchanged Chunk
            valid_interval_list.append([src_idx, curr_interval[0]])
            src_idx = curr_interval[0]

    LOGGER.debug(f"valid_interval_list: {valid_interval_list}")

    # Add chunk
    chunk_list_total = []
    for tgt_idx in range(len(merge_edits_list)):
        tgt_sent = tgt_sent_list[tgt_idx]
        edit_list = merge_edits_list[tgt_idx]
        token_mapping = token_mapping_total[tgt_idx]
        chunk_list = []
        for chunk_idx, chunk_interval in enumerate(valid_interval_list):
            is_ungrammatical = False
            for edit in edit_list:
                if chunk_interval == edit.src_interval:
                    # Corrected/Dummy Chunk
                    chunk_list.append(
                        Chunk(
                            chunk_index=chunk_idx,
                            src_interval=edit.src_interval.copy(),
                            tgt_interval=edit.tgt_interval.copy(),
                            src_tokens=edit.src_tokens.copy(),
                            tgt_tokens=edit.tgt_tokens.copy(),
                            types=edit.types.copy(),
                        )
                    )
                    is_ungrammatical = True
                    break

            if not is_ungrammatical:
                # Unchanged Chunk
                tgt_beg_idx = token_mapping[chunk_interval[0]]
                tgt_end_idx = (
                    0
                    if chunk_interval[1] == 0
                    else token_mapping[chunk_interval[1] - 1] + 1
                )
                chunk_list.append(
                    Chunk(
                        chunk_index=chunk_idx,
                        src_interval=chunk_interval.copy(),
                        tgt_interval=[tgt_beg_idx, tgt_end_idx],
                        src_tokens=src_sent[chunk_interval[0] : chunk_interval[1]],
                        tgt_tokens=tgt_sent[tgt_beg_idx:tgt_end_idx],
                        types=[],
                    )
                )

        chunk_list_total.append(chunk_list.copy())
        LOGGER.debug(f"chunk_list: {chunk_list}")

    # Sanity check: all chunk_list should have the same length
    chunk_len = len(chunk_list_total[0])
    for chunk_list in chunk_list_total:
        assert len(chunk_list) == chunk_len, f"{len(chunk_list)} != {chunk_len}"
    return chunk_list_total
