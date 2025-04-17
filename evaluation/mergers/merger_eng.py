from itertools import combinations, groupby
from re import sub
from string import punctuation
from typing import Any, List, Sequence, Tuple

import spacy.symbols as POS
from rapidfuzz.distance import Indel

from data import Edit

from .merger_base import BaseMerger


class MergerEng(BaseMerger):
    """Merger for English (ENG)."""

    OPEN_POS = {POS.ADJ, POS.AUX, POS.ADV, POS.NOUN, POS.VERB}

    def signature(self):
        return "eng"

    def get_rule_edits(
        self,
        source: Sequence,
        target: Sequence,
        align_seq: List[Tuple],
        tgt_index: int = 0,
    ) -> List[Edit]:
        edits = []
        # Split alignment into groups of M, T and rest. (T has a number after it)
        for op, group in groupby(
            align_seq, lambda x: x[0][0] if x[0][0] in {"M", "T"} else False
        ):
            group = list(group)
            if op == "M":  # Ignore M
                continue
            elif op == "T":  # T is always split
                for align in group:
                    src_tokens_tok = source[align[1] : align[2]]
                    tgt_tokens_tok = target[align[3] : align[4]]
                    edits.append(
                        Edit(
                            tgt_index=tgt_index,
                            src_interval=[align[1], align[2]],
                            tgt_interval=[align[3], align[4]],
                            src_tokens=(
                                [x.text for x in src_tokens_tok]
                                if src_tokens_tok
                                else []
                            ),
                            tgt_tokens=(
                                [x.text for x in tgt_tokens_tok]
                                if tgt_tokens_tok
                                else []
                            ),
                            src_tokens_tok=src_tokens_tok,
                            tgt_tokens_tok=tgt_tokens_tok,
                        )
                    )
            else:  # Process D, I and S subsequence
                processed = self._process_seq(
                    source=source,
                    target=target,
                    align_seq=group,
                )
                # Turn the processed sequence into edits
                for merged in processed:
                    src_tokens_tok = source[merged[1] : merged[2]]
                    tgt_tokens_tok = target[merged[3] : merged[4]]
                    edits.append(
                        Edit(
                            tgt_index=tgt_index,
                            src_interval=[merged[1], merged[2]],
                            tgt_interval=[merged[3], merged[4]],
                            src_tokens=(
                                [x.text for x in src_tokens_tok]
                                if src_tokens_tok
                                else []
                            ),
                            tgt_tokens=(
                                [x.text for x in tgt_tokens_tok]
                                if tgt_tokens_tok
                                else []
                            ),
                            src_tokens_tok=src_tokens_tok,
                            tgt_tokens_tok=tgt_tokens_tok,
                        )
                    )
        return edits

    # Input 1: A sequence of adjacent D, I and/or S alignments
    # Input 2: An Alignment object
    # Output: A sequence of merged/split alignments
    def _process_seq(
        self, source: Sequence, target: Sequence, align_seq: List[Tuple]
    ) -> List[Any]:
        """Merge edits by heuristic rules.

        Args:
            source (Sequence): Source sentence
            target (Sequence): Target sentence
            align_seq (List[Tuple]): _description_

        Returns:
            List[Any]: Merged edits
        """
        # Return single alignments
        if len(align_seq) <= 1:
            return align_seq
        # Get the ops for the whole sequence
        ops = [op[0] for op in align_seq]
        # Merge all `D` xor `I` operations. (95% of human multi-token edits contain S).
        if set(ops) == {"D"} or set(ops) == {"I"}:
            return self.merge_edits(align_seq)

        content = False  # True if edit includes a content word
        # Get indices of all start-end combinations in the seq: 012 = 01, 02, 12
        combos = list(combinations(range(0, len(align_seq)), 2))
        # Sort them starting with the largest spans first
        combos.sort(key=lambda x: x[1] - x[0], reverse=True)
        # Loop through combos
        for start, end in combos:
            # Ignore ranges that do NOT contain a substitution.
            if "S" not in ops[start : end + 1]:
                continue
            # Get the tokens in source and target. They will now never be empty.
            src = source[align_seq[start][1] : align_seq[end][2]]
            tgt = target[align_seq[start][3] : align_seq[end][4]]

            # First token possessive suffixes
            if start == 0 and (src[0].tag_ == "POS" or tgt[0].tag_ == "POS"):
                return [align_seq[0]] + self._process_seq(
                    source=source,
                    target=target,
                    align_seq=align_seq[1:],
                )

            # Merge possessive suffixes: [friends -> friend 's]
            if src[-1].tag_ == "POS" or tgt[-1].tag_ == "POS":
                return (
                    self._process_seq(
                        source=source,
                        target=target,
                        align_seq=align_seq[: end - 1],
                    )
                    + self.merge_edits(align_seq[end - 1 : end + 1])
                    + self._process_seq(
                        source=source,
                        target=target,
                        align_seq=align_seq[end + 1 :],
                    )
                )

            # Case changes
            if src[-1].lower == tgt[-1].lower:
                # Merge first token I or D: [Cat -> The big cat]
                if start == 0 and (
                    (len(src) == 1 and tgt[0].text[0].isupper())
                    or (len(tgt) == 1 and src[0].text[0].isupper())
                ):
                    return self.merge_edits(
                        align_seq[start : end + 1]
                    ) + self._process_seq(
                        source=source,
                        target=target,
                        align_seq=align_seq[end + 1 :],
                    )
                # Merge with previous punctuation: [, we -> . We], [we -> . We]
                if (len(src) > 1 and is_punct(src[-2])) or (
                    len(tgt) > 1 and is_punct(tgt[-2])
                ):
                    return (
                        self._process_seq(
                            source=source,
                            target=target,
                            align_seq=align_seq[: end - 1],
                        )
                        + self.merge_edits(align_seq[end - 1 : end + 1])
                        + self._process_seq(
                            source=source,
                            target=target,
                            align_seq=align_seq[end + 1 :],
                        )
                    )

            # Merge whitespace/hyphens: [acat -> a cat], [sub - way -> subway]
            s_str = sub("['-]", "", "".join([tok.lower_ for tok in src]))
            t_str = sub("['-]", "", "".join([tok.lower_ for tok in tgt]))
            if s_str == t_str:
                return (
                    self._process_seq(
                        source=source,
                        target=target,
                        align_seq=align_seq[:start],
                    )
                    + self.merge_edits(align_seq[start : end + 1])
                    + self._process_seq(
                        source=source,
                        target=target,
                        align_seq=align_seq[end + 1 :],
                    )
                )

            # Merge same POS or auxiliary/infinitive/phrasal verbs:
            # [to eat -> eating], [watch -> look at]
            pos_set = set([tok.pos for tok in src] + [tok.pos for tok in tgt])
            if len(src) != len(tgt) and (
                len(pos_set) == 1 or pos_set.issubset({POS.AUX, POS.PART, POS.VERB})
            ):
                return (
                    self._process_seq(
                        source=source,
                        target=target,
                        align_seq=align_seq[:start],
                    )
                    + self.merge_edits(align_seq[start : end + 1])
                    + self._process_seq(
                        source=source,
                        target=target,
                        align_seq=align_seq[end + 1 :],
                    )
                )

            # Split rules take effect when we get to the smallest chunks
            if end - start < 2:
                # Split adjacent substitutions
                if len(src) == len(tgt) == 2:
                    return self._process_seq(
                        source=source,
                        target=target,
                        align_seq=align_seq[: start + 1],
                    ) + self._process_seq(
                        source=source,
                        target=target,
                        align_seq=align_seq[start + 1 :],
                    )
                # Split similar substitutions at sequence boundaries
                if (ops[start] == "S" and char_cost(src[0], tgt[0]) > 0.75) or (
                    ops[end] == "S" and char_cost(src[-1], tgt[-1]) > 0.75
                ):
                    return self._process_seq(
                        source=source,
                        target=target,
                        align_seq=align_seq[: start + 1],
                    ) + self._process_seq(
                        source=source,
                        target=target,
                        align_seq=align_seq[start + 1 :],
                    )
                # Split final determiners
                if end == len(align_seq) - 1 and (
                    (ops[-1] in {"D", "S"} and src[-1].pos == POS.DET)
                    or (ops[-1] in {"I", "S"} and tgt[-1].pos == POS.DET)
                ):
                    return self._process_seq(
                        source=source,
                        target=target,
                        align_seq=align_seq[:-1],
                    ) + [align_seq[-1]]

            # Set content word flag
            if not pos_set.isdisjoint(self.OPEN_POS):
                content = True
        # Merge sequences that contain content words
        if content:
            return self.merge_edits(align_seq)
        else:
            return align_seq


# Check whether token is punctuation
def is_punct(token) -> bool:
    return token.pos == POS.PUNCT or token.text in punctuation


# Calculate the cost of character alignment; i.e. char similarity
def char_cost(a, b) -> float:
    return 1 - Indel.normalized_distance(a.text, b.text)
