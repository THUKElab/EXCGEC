import math
import random
from typing import Any, Dict, List

import numpy as np
from pydantic import Field

from ..constants import KEY_HYP_LEN, KEY_NGRAMS, KEY_REF_LEN
from ..schema import OverallScorerResult
from .scorer_base import BaseScorer


class GLEUScorer(BaseScorer):
    order: int = Field(default=4, description="Maximum order of ngrams")
    num_iter: int = Field(default=500, description="Number of iterations to run")
    smoothing: bool = Field(default=False, description="Smoothing factor")

    def __call__(self, metric_results: List[List[Dict]]) -> Dict[str, Any]:
        # Corpus-level
        score_corpus = self.score_corpus(metric_results)
        # Sentence-level
        score_sentence = self.score_sentence(metric_results)

        result = OverallScorerResult(
            num_sample=len(metric_results),
            scores={
                "gleu_corpus": score_corpus,
                "gleu_sentence": score_sentence,
            },
        )
        if self.table_print:
            self.print_result_table(result, num_sample=result.num_sample)
        return result

    def score_corpus(self, metric_results: List[List[Dict]]) -> float:
        total_hyp_len, total_ref_len = 0, 0
        total_ngrams = [0] * (len(metric_results[0][0][KEY_NGRAMS]) * 2)
        for sample_result in metric_results:
            for _ in range(self.num_iter):
                ref_idx = random.randint(0, len(sample_result) - 1)
                total_hyp_len += sample_result[ref_idx][KEY_HYP_LEN]
                total_ref_len += sample_result[ref_idx][KEY_REF_LEN]
                for n, precision in sample_result[ref_idx][KEY_NGRAMS].items():
                    assert len(precision) == 2
                    total_ngrams[2 * n - 2] += precision[0]
                    total_ngrams[2 * n - 1] += precision[1]

        # smooth 0 counts for sentence-level scores
        if self.smoothing:
            total_hyp_len = [x if x != 0 else 1 for x in total_ngrams]

        assert len(list(filter(lambda x: x == 0, total_ngrams))) == 0
        log_gleu_prec = (
            sum(
                [
                    math.log(float(x) / y)
                    for x, y in zip(total_ngrams[0::2], total_ngrams[1::2])
                ]
            )
            / 4
        )
        score = math.exp(
            min([0, 1 - float(total_ref_len) / total_hyp_len]) + log_gleu_prec
        )
        return score

    def score_sentence(self, scorer_inputs: List[List[Dict]]) -> float:
        total_scores = []
        for sample_idx, sample_result in enumerate(scorer_inputs):
            sample_score = []
            for ref_idx, ref_result in enumerate(sample_result):
                ref_len = ref_result[KEY_REF_LEN] if ref_result[KEY_REF_LEN] != 0 else 1
                hyp_len = ref_result[KEY_HYP_LEN] if ref_result[KEY_HYP_LEN] != 0 else 1
                log_gleu_prec = 0.0
                for n, precision in ref_result[KEY_NGRAMS].items():
                    numerator = precision[0] if precision[0] != 0 else 1
                    denominator = precision[1] if precision[1] != 0 else 1
                    log_gleu_prec += math.log(float(numerator) / denominator)
                log_gleu_prec /= self.order
                ref_score = math.exp(
                    min([0, 1 - float(ref_len) / hyp_len]) + log_gleu_prec
                )
                total_scores.append(ref_score)
                sample_score.append(ref_score)
        # return {
        #     "score": np.average(total_scores),
        #     "std": np.std(total_scores),
        #     "ci": scipy.stats.norm.interval(0.95, loc=mean, scale=std)
        # }
        return np.average(total_scores)

    # def print_result_table(
    #     self, result: OverallScorerResult, sout: TextIO = sys.stdout, **kwargs: Any
    # ) -> None:
    #     """Visulize results as a table."""
    #     tabular_data = defaultdict(list)
    #     for key, score_result in result.scores.items():
    #         assert isinstance(score_result, HEUOEditScorerResult)
    #         tabular_data["metric"].append(key)
    #         tabular_data["COM"].append(score_result.com)
    #         tabular_data["HIT"].append(score_result.hit)
    #         tabular_data["ERR"].append(score_result.err)
    #         tabular_data["UND"].append(score_result.und)
    #         tabular_data["OVE"].append(score_result.ove)

    #         tabular_data["TP"].append(score_result.tp)
    #         tabular_data["FP"].append(score_result.fp)
    #         tabular_data["FN"].append(score_result.fn)
    #         tabular_data["TN"].append(score_result.tn)
    #         tabular_data["FP_NE"].append(score_result.fp_ne)
    #         tabular_data["FP_UN"].append(score_result.fp_un)
    #         tabular_data["NE"].append(score_result.necessary)

    #     table = tabulate(
    #         tabular_data,
    #         tablefmt="fancy_grid",
    #         headers="keys",
    #         floatfmt=".4f",
    #         missingval="N/A",
    #         numalign="left",
    #     )
    #     sout.write("\n" + table + "\n")
    #     for k, v in kwargs.items():
    #         sout.write(f"{k}: {v}\n")
    #     sout.write("\n")
