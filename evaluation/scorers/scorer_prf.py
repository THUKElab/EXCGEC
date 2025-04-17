import sys
from collections import defaultdict
from typing import Any, List, TextIO, Tuple

import numpy as np
from pydantic import Field
from tabulate import tabulate

from data import Dataset

from ..schema import (
    BaseChunkMetricResult,
    BaseEditMetricResult,
    EditScorerResult,
    OverallScorerResult,
    SampleMetricResult,
)
from .scorer_base import BaseScorer
from .scorer_utils import compute_acc, compute_prf, gt_numbers


class PRFEditScorer(BaseScorer):
    """Traditional Edit-based Scorer."""

    factor_beta: float = Field(
        default=0.5, description="Trade-off factor of Precision and Recall"
    )
    table_print: bool = Field(default=True)

    def score(
        self,
        dataset_hyp: Dataset,
        dataset_ref: Dataset,
        metric_results: List[SampleMetricResult],
    ) -> OverallScorerResult:
        dataset_scorer_results, dataset_scorer_results_weighted = [], []
        for sample_metric_result in metric_results:
            sample_scorer_result, sample_scorer_result_weighted = self.score_sample(
                sample_metric_result
            )
            dataset_scorer_results.append(sample_scorer_result)
            dataset_scorer_results_weighted.append(sample_scorer_result_weighted)

        # Corpus-level
        score_corpus = self.score_corpus(dataset_scorer_results)
        score_corpus_weighted = self.score_corpus(dataset_scorer_results_weighted)

        # Sentence-level
        score_sentence = self.score_sentence(dataset_scorer_results)
        score_sentence_weighted = self.score_sentence(dataset_scorer_results_weighted)

        result = OverallScorerResult(
            num_sample=len(metric_results),
            scores={
                "prf_corpus_unweighted": score_corpus,
                "prf_corpus_weighted": score_corpus_weighted,
                "prf_sentence_unweighted": score_sentence,
                "prf_sentence_weighted": score_sentence_weighted,
            },
        )
        if self.table_print:
            self.print_result_table(result, num_sample=result.num_sample)
        return result

    def score_sample(
        self, metric_result: SampleMetricResult
    ) -> Tuple[List[EditScorerResult], List[EditScorerResult]]:
        results, results_weighted = [], []
        for ref_result in metric_result.ref_results:
            if isinstance(ref_result, BaseEditMetricResult):
                tp = len(ref_result.tp_edits)
                fp = len(ref_result.fp_edits)
                fn = len(ref_result.fn_edits)
                tn = len(ref_result.tn_edits)
            elif isinstance(ref_result, BaseChunkMetricResult):
                tp = len(ref_result.tp_chunks)
                fp = len(ref_result.fp_chunks)
                fn = len(ref_result.fn_chunks)
                tn = len(ref_result.tn_chunks)
            else:
                raise ValueError()
            p, r, f = compute_prf(tp, fp, fn, beta=self.factor_beta)
            acc = compute_acc(tp, fp, fn, tn)
            result = EditScorerResult(
                tp=tp, fp=fp, fn=fn, tn=tn, p=p, r=r, f=f, acc=acc
            )
            results.append(result)

            # Compute weighted result
            if isinstance(ref_result, BaseEditMetricResult):
                tp = sum([x.weight for x in ref_result.tp_edits])
                fp = sum([x.weight for x in ref_result.fp_edits])
                fn = sum([x.weight for x in ref_result.fn_edits])
                tn = sum([x.weight for x in ref_result.tn_edits])
            elif isinstance(ref_result, BaseChunkMetricResult):
                tp = sum([x.weight for x in ref_result.tp_chunks])
                fp = sum([x.weight for x in ref_result.fp_chunks])
                fn = sum([x.weight for x in ref_result.fn_chunks])
                tn = sum([x.weight for x in ref_result.tn_chunks])
            else:
                raise ValueError()
            p, r, f = compute_prf(tp, fp, fn, beta=self.factor_beta)
            acc = compute_acc(tp, fp, fn, tn)
            result_weighted = EditScorerResult(
                tp=tp, fp=fp, fn=fn, tn=tn, p=p, r=r, f=f, acc=acc
            )
            results_weighted.append(result_weighted)
        return results, results_weighted

    def score_corpus(
        self, dataset_scorer_results: List[List[EditScorerResult]]
    ) -> EditScorerResult:
        """Calculate corpus-level PEF Scores."""
        total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0
        for sample_scorer_result in dataset_scorer_results:
            best_f, best_tp, best_fp, best_fn, best_tn = -1.0, 0, 0, 0, 0
            for ref_scorer_result in sample_scorer_result:
                _tp = ref_scorer_result.tp
                _fp = ref_scorer_result.fp
                _fn = ref_scorer_result.fn
                _tn = ref_scorer_result.tn
                _p, _r, _f = compute_prf(
                    tp=total_tp + _tp,
                    fp=total_fp + _fp,
                    fn=total_fn + _fn,
                    beta=self.factor_beta,
                )
                if gt_numbers(
                    [_f, _tp, -_fp, -_fn, _tn],
                    [best_f, best_tp, -best_fp, -best_fn, best_tn],
                ):
                    best_f, best_tp, best_fp, best_fn, best_tn = _f, _tp, _fp, _fn, _tn

            total_tp += best_tp
            total_fp += best_fp
            total_fn += best_fn
            total_tn += best_tn

        final_p, final_r, final_f = compute_prf(
            total_tp, total_fp, total_fn, beta=self.factor_beta
        )
        final_acc = compute_acc(total_tp, total_fp, total_fn, total_tn)
        return EditScorerResult(
            tp=total_tp,
            fp=total_fp,
            fn=total_fn,
            tn=total_tn,
            p=final_p,
            r=final_r,
            f=final_f,
            acc=final_acc,
        )

    def score_sentence(
        self, dataset_scorer_results: List[List[EditScorerResult]]
    ) -> EditScorerResult:
        """Calculate sentence-level PEF Scores."""
        total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0
        total_f, total_p, total_r, total_acc = [], [], [], []
        for sample_scorer_result in dataset_scorer_results:
            best_f, best_p, best_r, best_acc = -1.0, -1.0, -1.0, -1.0
            best_tp, best_fp, best_fn, best_tn = 0, 0, 0, 0

            for ref_scorer_result in sample_scorer_result:
                _tp = ref_scorer_result.tp
                _fp = ref_scorer_result.fp
                _fn = ref_scorer_result.fn
                _tn = ref_scorer_result.tn
                _p, _r, _f = compute_prf(_tp, _fp, _fn)
                _acc = compute_acc(_tp, _fp, _fn, _tn)

                if gt_numbers([_f, _p, _r, _acc], [best_f, best_p, best_r, best_acc]):
                    best_f, best_p, best_r, best_acc = _f, _p, _r, _acc
                    best_tp = _tp
                    best_fp = _fp
                    best_fn = _fn
                    best_tn = _tn

            total_tp += best_tp
            total_fp += best_fp
            total_fn += best_fn
            total_tn += best_tn
            total_f.append(best_f)
            total_p.append(best_p)
            total_r.append(best_r)
            total_acc.append(best_acc)
        return EditScorerResult(
            tp=total_tp,
            fp=total_fp,
            fn=total_fn,
            tn=total_tn,
            p=np.average(total_p),
            r=np.average(total_r),
            f=np.average(total_f),
            acc=np.average(total_acc),
        )

    def print_result_table(
        self, result: OverallScorerResult, sout: TextIO = sys.stdout, **kwargs: Any
    ) -> None:
        """Visulize results as a table."""
        tabular_data = defaultdict(list)
        for key, score_result in result.scores.items():
            assert isinstance(score_result, EditScorerResult)
            tabular_data["metric"].append(key)
            tabular_data["F"].append(score_result.f)
            tabular_data["P"].append(score_result.p)
            tabular_data["R"].append(score_result.r)
            tabular_data["ACC"].append(score_result.acc)

            tabular_data["TP"].append(score_result.tp)
            tabular_data["FP"].append(score_result.fp)
            tabular_data["FN"].append(score_result.fn)
            tabular_data["TN"].append(score_result.tn)

        table = tabulate(
            tabular_data,
            tablefmt="fancy_grid",
            headers="keys",
            floatfmt=".4f",
            missingval="N/A",
            numalign="left",
        )
        sout.write("\n" + table + "\n")
        for k, v in kwargs.items():
            sout.write(f"{k}: {v}\n")
        sout.write("\n")
