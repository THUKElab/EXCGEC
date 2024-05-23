from typing import Any, List

from data import Sample

from ..schema import BaseChunkMetricResult, BaseEditMetricResult, SampleMetricResult


class BaseWeigher:
    """Compute edit weights.

    Edit weigher is beneficial to improve correlations with human judgements.

    """

    def __init__(self) -> None:
        super().__init__()

    def signature(self) -> str:
        """Returns a signature for the tokenizer."""
        return "none"

    def setup(self, **kwargs: Any) -> None:
        """Prepare for edit weighting. Rewrite this function if necessary."""
        pass

    def __call__(
        self,
        sample_hyp: Sample,
        sample_ref: Sample,
        metric_result: SampleMetricResult,
        **kwargs: Any
    ) -> None:
        for ref_result in metric_result.ref_results:
            if isinstance(ref_result, BaseEditMetricResult):
                for tp_edit in ref_result.tp_edits:
                    tp_edit.weight = 1.0
                for fp_edit in ref_result.fp_edits:
                    fp_edit.weight = 1.0
                for fn_edit in ref_result.fn_edits:
                    fn_edit.weight = 1.0
                for tn_edit in ref_result.tn_edits:
                    tn_edit.weight = 1.0
                for fp_ne_edits in ref_result.fp_ne_edits:
                    fp_ne_edits.weight = 1.0
                for fp_un_edits in ref_result.fp_un_edits:
                    fp_un_edits.weight = 1.0
            elif isinstance(ref_result, BaseChunkMetricResult):
                for tp_chunk in ref_result.tp_chunks:
                    tp_chunk.weight = 1.0
                for fp_chunk in ref_result.fp_chunks:
                    fp_chunk.weight = 1.0
                for fn_chunk in ref_result.fn_chunks:
                    fn_chunk.weight = 1.0
                for tn_chunk in ref_result.tn_chunks:
                    tn_chunk.weight = 1.0
                for fp_ne_chunks in ref_result.fp_ne_chunks:
                    fp_ne_chunks.weight = 1.0
                for fp_un_chunks in ref_result.fp_un_chunks:
                    fp_un_chunks.weight = 1.0
            else:
                raise ValueError()

    # def weigh_batch(
    #     self,
    #     samples_hyp: List[Sample],
    #     samples_ref: List[Sample],
    #     metric_results: List[SampleMetricResult],
    # ) -> None:
    #     for metric_result in metric_results:
    #         self.__call__(metric_result=metric_result)

    def get_weights_batch(
        self,
        samples_hyp: List[Sample],
        samples_ref: List[Sample],
        metric_results: List[SampleMetricResult],
    ) -> None:
        for metric_result in metric_results:
            self.__call__(metric_result=metric_result)
