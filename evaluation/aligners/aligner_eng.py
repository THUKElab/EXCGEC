from typing import Any

import spacy.parts_of_speech as POS
from rapidfuzz.distance import Indel

from .aligner_base import BaseAligner


class AlignerEng(BaseAligner):
    _open_pos = {POS.ADJ, POS.ADV, POS.NOUN, POS.VERB}

    def signature(self) -> str:
        return "eng"

    def get_sub_cost(self, src_token: Any, tgt_token: Any) -> float:
        """Cost of Linguistic Damerau-Levenshtein.

        Args:
            src_token (str): A spacy source Token.
            tgt_token (str): A spacy target Token.

        Returns:
            float: A linguistic cost between 0 < x < 2.
        """
        # Short circuit if the only difference is case
        if src_token.lower == tgt_token.lower:
            return 0
        # Lemma cost
        if src_token.lemma == tgt_token.lemma:
            lemma_cost = 0
        else:
            lemma_cost = 0.499
        # POS cost
        if src_token.pos == tgt_token.pos:
            pos_cost = 0
        elif src_token.pos in self._open_pos and tgt_token.pos in self._open_pos:
            pos_cost = 0.25
        else:
            pos_cost = 0.5
        # Char cost
        char_cost = Indel.normalized_distance(src_token.text, tgt_token.text)
        # Combine the costs
        return lemma_cost + pos_cost + char_cost

    # # Alignment object string representation
    # def __str__(self):
    #     orig = " ".join(["Orig:"] + [tok.text for tok in source])
    #     cor = " ".join(["Cor:"] + [tok.text for tok in target])
    #     cost_matrix = "\n".join(["Cost Matrix:"] + [str(row) for row in self.cost_matrix])
    #     op_matrix = "\n".join(["Operation Matrix:"] + [str(row) for row in self.op_matrix])
    #     seq = "Best alignment: " + str([a[0] for a in self.align_seq])
    #     return "\n".join([orig, cor, cost_matrix, op_matrix, seq])
