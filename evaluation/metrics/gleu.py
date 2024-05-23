# This file is referenced from:
# Ground Truth for Grammatical Error Correction Metrics [ACL 2015]
# by Courtney Napoles, Keisuke Sakaguchi, Matt Post, and Joel Tetreault.
# <https://github.com/alopez/en600.468/blob/master/reranker/>

from collections import Counter, defaultdict
from typing import Dict, List

from data import Sample
from utils import get_logger

from ..constants import KEY_HYP_LEN, KEY_NGRAMS, KEY_REF_LEN
from .base import BaseMetric

LOGGER = get_logger(__name__)


class GLEU(BaseMetric):
    def __init__(self, scorer: str = "corpus_gleu"):
        super().__init__(scorer)
        self.order = self.scorer.order
        self.tokenizer = self.build_tokenizer(
            name=tokenizer if tokenizer else lang,
            **kwargs,
        )

    def evaluate_sample(self, sample_hyp: Sample, sample_ref: Sample) -> List[Dict]:
        src_split: List[str] = self.tokenizer(sample_hyp.source[0], to_list=True)
        hyp_split: List[str] = self.tokenizer(sample_hyp.target[0], to_list=True)
        refs_split: List[List[str]] = [
            self.tokenizer(x, to_list=True) for x in sample_ref.target
        ]

        src_ngrams = [
            self.get_ngram_counts(src_split, n) for n in range(1, self.order + 1)
        ]
        hyp_ngrams = [
            self.get_ngram_counts(hyp_split, n) for n in range(1, self.order + 1)
        ]
        refs_len = [len(x) for x in refs_split]

        results = []
        for ref_idx, ref_split in enumerate(refs_split):
            ngrams_precision = defaultdict()
            for n in range(1, self.order + 1):
                _src_ngrams = src_ngrams[n - 1]
                _hyp_ngrams = hyp_ngrams[n - 1]
                _ref_ngrams = self.get_ngram_counts(ref_split, n)
                src_ref_diff = self.get_ngram_diff(_src_ngrams, _ref_ngrams)

                numerator = max(
                    [
                        sum((_hyp_ngrams & _ref_ngrams).values())
                        - sum((_hyp_ngrams & src_ref_diff).values()),
                        0,
                    ]
                )
                denominator = max([len(hyp_split) + 1 - n, 0])
                ngrams_precision[n] = [numerator, denominator]
            results.append(
                {
                    KEY_HYP_LEN: len(hyp_split),
                    KEY_REF_LEN: refs_len[ref_idx],
                    KEY_NGRAMS: ngrams_precision,
                }
            )
        return results

    @staticmethod
    def get_ngram_counts(sentence: List[str], n: int) -> Counter:
        return Counter(
            [tuple(sentence[i : i + n]) for i in range(len(sentence) + 1 - n)]
        )

    @staticmethod
    def get_ngram_diff(a: Counter, b: Counter) -> Counter:
        """returns ngrams in `a` but not in `b`."""
        diff = Counter(a)
        for k in set(a) & set(b):
            del diff[k]
        return diff
