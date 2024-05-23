import copy

from data import Sample
from utils import get_logger

from ...schema import BaseChunkMetricResult, SampleMetricResult
from .cleme_base import CLEME
from .cleme_utils import all_correct, any_correct

LOGGER = get_logger(__name__)


class IndependentCLEME(CLEME):
    def evaluate_sample_correction(
        self, sample_hyp: Sample, sample_ref: Sample, in_place: bool = False
    ) -> SampleMetricResult:
        """Acquire TP, FP, FN, and TN correction counts.

        V1: Only consider one ref_result.
        V2: Consider every ref_result for each reference.

        Args:
            sample_hyp (Sample): Hyp sample.
            sample_ref (Sample): Ref sample.

        Returns:
            MetricSampleResult: Correction result of IndependentCLEME.
        """
        tp_chunks, fp_chunks, fn_chunks, tn_chunks = [], [], [], []
        fp_ne_chunks, fp_un_chunks = [], []

        chunks_hyp = sample_hyp.chunks[0][0]
        for chunk_idx, chunk_hyp in enumerate(chunks_hyp):
            chunks_same_index = [x[chunk_idx] for x in sample_ref.chunks[0]]

            if chunk_hyp.types:
                if chunk_hyp in chunks_same_index:
                    tp_chunks.append(chunk_hyp)
                else:
                    fp_chunks.append(chunk_hyp)
                    # Distinguish fp_ne and fp_un
                    if any_correct(chunks_same_index):
                        fp_ne_chunks.append(chunk_hyp)
                    else:
                        fp_un_chunks.append(chunk_hyp)
            else:
                if all_correct(chunks_same_index):
                    fn_chunks.append(chunk_hyp)
                else:
                    tn_chunks.append(chunk_hyp)

        ref_results = []
        for _ in sample_ref.target:
            if in_place:
                ref_result = BaseChunkMetricResult(
                    tp_chunks=tp_chunks,
                    fp_chunks=fp_chunks,
                    fn_chunks=fn_chunks,
                    tn_chunks=tn_chunks,
                    fp_ne_chunks=fp_ne_chunks,
                    fp_un_chunks=fp_un_chunks,
                )
            else:
                ref_result = BaseChunkMetricResult(
                    tp_chunks=copy.deepcopy(tp_chunks),
                    fp_chunks=copy.deepcopy(fp_chunks),
                    fn_chunks=copy.deepcopy(fn_chunks),
                    tn_chunks=copy.deepcopy(tn_chunks),
                    fp_ne_chunks=copy.deepcopy(fp_ne_chunks),
                    fp_un_chunks=copy.deepcopy(fp_un_chunks),
                )
            ref_results.append(ref_result)
        return SampleMetricResult(ref_results=ref_results)

    def evaluate_sample_detection(
        self, sample_hyp: Sample, sample_ref: Sample
    ) -> SampleMetricResult:
        # TODO
        pass
