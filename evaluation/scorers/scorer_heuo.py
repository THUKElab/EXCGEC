import sys
from collections import defaultdict
from typing import Any, List, TextIO, Tuple

import numpy as np
from pydantic import Field
from tabulate import tabulate

from data import Dataset, Sample
from utils import get_logger

from ..schema import (
    BaseChunkMetricResult,
    EditScorerResult,
    HEUOEditScorerResult,
    OverallScorerResult,
    SampleMetricResult,
)
from .scorer_base import BaseScorer
from .scorer_utils import gt_numbers

LOGGER = get_logger(__name__)


class HEUOEditScorer(BaseScorer):
    """Disentangled Edit-based Scorer, which generate interpretable scores."""

    factor_hit: float = Field(default=0.50)
    factor_err: float = Field(default=0.40)
    factor_und: float = Field(default=0.05)
    factor_ove: float = Field(default=0.05)
    table_print: bool = Field(default=True)
    max_value: float = Field(default=0.01)

    def __init__(
        self,
        factor_hit: float = 0.50,
        factor_err: float = 0.40,
        factor_und: float = 0.05,
        factor_ove: float = 0.05,
        max_value: float = 0.01,
        **kwargs: Any,
    ) -> None:
        factor_combined = factor_hit + factor_err + factor_und + factor_ove
        if abs(factor_combined - 1.0) > 1e-5:
            raise ValueError("Invalid factors: combination of factors must be 1.0")
        super().__init__(
            factor_hit=factor_hit,
            factor_err=factor_err,
            factor_und=factor_und,
            factor_ove=factor_ove,
            max_value=max_value,
            **kwargs,
        )

    def compute_comprehensive_indicator(
        self, tp: float, fn: float, fp_ne: float, fp_un: float
    ) -> Tuple:
        """Compute comprehensive indicator in the form of linear weighted sum.

        Returns:
            Tuple: hit, err, und, ove, com.
        """
        ne = tp + fp_ne + fn
        fp = fp_ne + fp_un

        hit = float(tp) / ne if ne else 0.0
        err = float(fp_ne) / ne if ne else 0.0
        und = float(fn) / ne if ne else 0.0
        ove = float(fp_un) / (tp + fp) if tp + fp else 0.0

        # NOTE: Linear weighted sum
        com = (
            self.factor_hit * hit
            + self.factor_err * (1 - err)
            + self.factor_und * (1 - und)
            + self.factor_ove * (1 - ove)
        )
        # It will slightly change results if you use round
        # return round(hit, 4), round(err, 4), round(und, 4), round(ove, 4), round(com, 4)
        return hit, err, und, ove, com

    def score(
        self,
        dataset_hyp: Dataset,
        dataset_ref: Dataset,
        metric_results: List[SampleMetricResult],
        adapt: bool = False,
    ) -> OverallScorerResult:
        LOGGER.info(self)
        dataset_scorer_results, dataset_scorer_results_weighted = [], []
        for sample_idx, sample_metric_result in enumerate(metric_results):
            sample_scorer_result, sample_scorer_result_weighted = self.score_sample(
                sample_hyp=dataset_hyp[sample_idx],
                sample_ref=dataset_ref[sample_idx],
                metric_result=sample_metric_result,
            )
            dataset_scorer_results.append(sample_scorer_result)
            dataset_scorer_results_weighted.append(sample_scorer_result_weighted)

        # Corpus-level unweighted
        if adapt:
            self.factor_hit = 0.60
            self.factor_err = 0.10
            self.factor_und = 0.10
            self.factor_ove = 0.20
        score_corpus = self.score_corpus(dataset_scorer_results)

        # Corpus-level weighted
        if adapt:
            self.factor_hit = 0.60
            self.factor_err = 0.30
            self.factor_und = 0.05
            self.factor_ove = 0.05
        score_corpus_weighted = self.score_corpus(dataset_scorer_results_weighted)

        # Sentence-level unweighted
        if adapt:
            self.factor_hit = 0.60
            self.factor_err = 0.10
            self.factor_und = 0.05
            self.factor_ove = 0.25
        score_sentence = self.score_sentence(dataset_scorer_results)

        # Sentence-level weighted
        if adapt:
            self.factor_hit = 0.15
            self.factor_err = 0.35
            self.factor_und = 0.35
            self.factor_ove = 0.15
        score_sentence_weighted = self.score_sentence(dataset_scorer_results_weighted)

        result = OverallScorerResult(
            num_sample=len(metric_results),
            scores={
                "heuo_corpus_unweighted": score_corpus,
                "heuo_corpus_weighted": score_corpus_weighted,
                "heuo_sentence_unweighted": score_sentence,
                "heuo_sentence_weighted": score_sentence_weighted,
            },
        )
        if self.table_print:
            self.print_result_table(result, num_sample=result.num_sample)
        return result

    def score_sample(
        self,
        sample_hyp: Sample,
        sample_ref: Sample,
        metric_result: SampleMetricResult,
    ) -> Tuple[List[EditScorerResult], List[EditScorerResult]]:
        results, results_weighted = [], []
        for ref_idx, ref_result in enumerate(metric_result.ref_results):
            if not isinstance(ref_result, BaseChunkMetricResult):
                raise ValueError

            # NOTE: Trick: Remove references with dramatic revision
            # if ref_result.sim_src_tgt < 0.80:
            #     continue

            tp = len(ref_result.tp_chunks)
            fp = len(ref_result.fp_chunks)
            fn = len(ref_result.fn_chunks)
            tn = len(ref_result.tn_chunks)
            fp_ne = len(ref_result.fp_ne_chunks)
            fp_un = len(ref_result.fp_un_chunks)

            # TODO
            # fn = 0
            # for chunk in ref_result.fn_chunks:
            #     ref_tokens = sample_ref.chunks[0][ref_idx][chunk.chunk_index].tgt_tokens
            #     if max(len(chunk.tgt_tokens), len(ref_tokens)) <= 2:
            #         fn += 1

            hit, err, und, ove, com = self.compute_comprehensive_indicator(
                tp=tp, fn=fn, fp_ne=fp_ne, fp_un=fp_un
            )
            result = HEUOEditScorerResult(
                tp=tp,
                fp=fp,
                fn=fn,
                tn=tn,
                fp_ne=fp_ne,
                fp_un=fp_un,
                necessary=tp + fp_ne + fn,
                unnecessary=tp + fp,
                hit=hit,
                err=err,
                und=und,
                ove=ove,
                com=com,
            )
            results.append(result)

            # Compute weighted result. NOTE: Apply max clipper
            tp = sum([x.weight for x in ref_result.tp_chunks])
            fp = sum([x.weight for x in ref_result.fp_chunks])
            fn = sum([x.weight for x in ref_result.fn_chunks])
            tn = sum([x.weight for x in ref_result.tn_chunks])
            fp_ne = sum([x.weight for x in ref_result.fp_ne_chunks])
            fp_un = sum([x.weight for x in ref_result.fp_un_chunks])

            # TODO: FN should be applied
            # fn = 0.0
            # for chunk in ref_result.fn_chunks:
            #     ref_tokens = sample_ref.chunks[0][ref_idx][chunk.chunk_index].tgt_tokens
            #     # if len(ref_tokens) > 2:
            #     #     fn += chunk.weight / max(1, len(ref_tokens))
            #     if max(len(chunk.tgt_tokens), len(ref_tokens)) <= 2:
            #         fn += chunk.weight

            hit, err, und, ove, com = self.compute_comprehensive_indicator(
                tp=tp, fn=fn, fp_ne=fp_ne, fp_un=fp_un
            )
            result_weighted = HEUOEditScorerResult(
                tp=tp,
                fp=fp,
                fn=fn,
                tn=tn,
                fp_ne=fp_ne,
                fp_un=fp_un,
                necessary=tp + fp_ne + fn,
                unnecessary=fp_un,
                hit=hit,
                err=err,
                und=und,
                ove=ove,
                com=com,
            )
            results_weighted.append(result_weighted)
        return results, results_weighted

    def score_corpus(
        self, dataset_scorer_results: List[List[HEUOEditScorerResult]]
    ) -> HEUOEditScorerResult:
        """Calculate corpus-level HEUO Scores."""
        total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0
        total_fp_ne, total_fp_un = 0, 0

        for sample_scorer_result in dataset_scorer_results:
            best_com = -1.0
            best_tp, best_fp, best_fn, best_tn = 0, 0, 0, 0
            best_fp_ne, best_fp_un = 0, 0
            for ref_scorer_result in sample_scorer_result:
                _tp = ref_scorer_result.tp
                _fp = ref_scorer_result.fp
                _fn = ref_scorer_result.fn
                _tn = ref_scorer_result.tn
                _fp_ne = ref_scorer_result.fp_ne
                _fp_un = ref_scorer_result.fp_un

                hit, err, und, ove, com = self.compute_comprehensive_indicator(
                    tp=total_tp + _tp,
                    fn=total_fn + _fn,
                    fp_ne=total_fp_ne + _fp_ne,
                    fp_un=total_fp_un + _fp_un,
                )

                # TODO: Maybe sorting by HEUO would be better.
                if gt_numbers(
                    [com, _tp, -_fp, -_fn, _tn],
                    [best_com, best_tp, -best_fp, -best_fn, best_tn],
                ):
                    best_com = com
                    best_tp = _tp
                    best_fp = _fp
                    best_fn = _fn
                    best_tn = _tn
                    best_fp_ne = _fp_ne
                    best_fp_un = _fp_un

                # NOTE: Error code
                # hit, err, und, ove, com = self.compute_comprehensive_indicator(
                #     tp=_tp, fn=_fn, fp_ne=_fp_ne, fp_un=_fp_un
                # )

                # if com > best_com:
                #     # best_com = com
                #     best_tp = _tp
                #     best_fp = _fp
                #     best_fn = _fn
                #     best_tn = _tn
                #     best_fp_ne = _fp_ne
                #     best_fp_un = _fp_un

            total_tp += best_tp
            total_fp += best_fp
            total_fn += best_fn
            total_tn += best_tn
            total_fp_ne += best_fp_ne
            total_fp_un += best_fp_un

        hit, err, und, ove, com = self.compute_comprehensive_indicator(
            tp=total_tp,
            fn=total_fn,
            fp_ne=total_fp_ne,
            fp_un=total_fp_un,
        )
        return HEUOEditScorerResult(
            tp=total_tp,
            fp=total_fp,
            fn=total_fn,
            tn=total_tn,
            fp_ne=total_fp_ne,
            fp_un=total_fp_un,
            necessary=total_tp + total_fp_ne + total_fn,
            unnecessary=total_fp_un,
            com=com,
            hit=hit,
            err=err,
            und=und,
            ove=ove,
        )

    def score_sentence(
        self, dataset_scorer_results: List[List[HEUOEditScorerResult]]
    ) -> HEUOEditScorerResult:
        """Calculate sentence-level HEUO Scores."""
        total_com, total_hit, total_err, total_und, total_ove = [], [], [], [], []
        total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0
        total_fp_ne, total_fp_un = 0, 0

        for sample_scorer_result in dataset_scorer_results:
            if not len(sample_scorer_result):
                continue

            best_com = -1.0
            best_hit, best_err, best_und, best_ove = -1.0, -1.0, -1.0, -1.0
            best_tp, best_fp, best_fn, best_tn = 0, 0, 0, 0
            best_fp_ne, best_fp_un = 0, 0
            for ref_scorer_result in sample_scorer_result:
                _tp = ref_scorer_result.tp
                _fp = ref_scorer_result.fp
                _fn = ref_scorer_result.fn
                _tn = ref_scorer_result.tn
                _fp_ne = ref_scorer_result.fp_ne
                _fp_un = ref_scorer_result.fp_un
                hit, err, und, ove, com = self.compute_comprehensive_indicator(
                    tp=_tp, fn=_fn, fp_ne=_fp_ne, fp_un=_fp_un
                )

                if gt_numbers(
                    [com, hit, -err, -und, -ove],
                    [best_com, best_hit, -best_err, -best_und, -best_ove],
                ):
                    best_com = com
                    best_hit = hit
                    best_err = err
                    best_und = und
                    best_ove = ove
                    best_tp = _tp
                    best_fp = _fp
                    best_fn = _fn
                    best_tn = _tn
                    best_fp_ne = _fp_ne
                    best_fp_un = _fp_un

            total_tp += best_tp
            total_fp += best_fp
            total_fn += best_fn
            total_tn += best_tn
            total_fp_ne += best_fp_ne
            total_fp_un += best_fp_un
            total_com.append(best_com)
            total_hit.append(best_hit)
            total_err.append(best_err)
            total_und.append(best_und)
            total_ove.append(best_ove)

        return HEUOEditScorerResult(
            tp=total_tp,
            fp=total_fp,
            fn=total_fn,
            tn=total_tn,
            fp_ne=total_fp_ne,
            fp_un=total_fp_un,
            necessary=total_tp + total_fp_ne + total_fn,
            unnecessary=total_fp_un,
            com=np.average(total_com),
            hit=np.average(total_hit),
            err=np.average(total_err),
            und=np.average(total_und),
            ove=np.average(total_ove),
        )

    def print_result_table(
        self, result: OverallScorerResult, sout: TextIO = sys.stdout, **kwargs: Any
    ) -> None:
        """Visulize results as a table."""
        tabular_data = defaultdict(list)
        for key, score_result in result.scores.items():
            assert isinstance(score_result, HEUOEditScorerResult)
            tabular_data["metric"].append(key)
            tabular_data["COM"].append(score_result.com)
            tabular_data["HIT"].append(score_result.hit)
            tabular_data["ERR"].append(score_result.err)
            tabular_data["UND"].append(score_result.und)
            tabular_data["OVE"].append(score_result.ove)

            tabular_data["TP"].append(score_result.tp)
            tabular_data["FP"].append(score_result.fp)
            tabular_data["FN"].append(score_result.fn)
            tabular_data["TN"].append(score_result.tn)
            tabular_data["FP_NE"].append(score_result.fp_ne)
            tabular_data["FP_UN"].append(score_result.fp_un)
            tabular_data["NE"].append(score_result.necessary)

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
