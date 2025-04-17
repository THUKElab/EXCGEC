import copy
import logging
from typing import List

from data import Sample
from utils import get_logger

from ...schema import BaseChunkMetricResult, SampleMetricResult
from .cleme_base import CLEME
from .cleme_utils import chunk_list_to_text

LOGGER = get_logger(__name__, level=logging.INFO)


class DependentCLEME(CLEME):
    def evaluate_sample_correction(
        self, sample_hyp: Sample, sample_ref: Sample, in_place: bool = False
    ) -> SampleMetricResult:
        """Acquire TP, FP, FN, and TN correction counts.

        Args:
            sample_hyp (Sample): Hyp sample.
            sample_ref (Sample): Ref sample.

        Returns:
            MetricSampleResult: Correction result of DependentCLEME.
        """
        ref_results: List[BaseChunkMetricResult] = []
        for chunks_ref in sample_ref.chunks[0]:
            tp_chunks, fp_chunks, fn_chunks, tn_chunks = [], [], [], []
            fp_ne_chunks, fp_un_chunks = [], []
            for chunk_index, chunk_hyp in enumerate(sample_hyp.chunks[0][0]):
                if chunk_hyp.types:
                    if chunk_hyp == chunks_ref[chunk_index]:
                        tp_chunks.append(chunk_hyp)
                    else:
                        fp_chunks.append(chunk_hyp)
                        # Distinguish fp_ne and fp_un
                        if chunks_ref[chunk_index].types:
                            fp_ne_chunks.append(chunk_hyp)
                        else:
                            fp_un_chunks.append(chunk_hyp)
                else:
                    if chunk_hyp != chunks_ref[chunk_index]:
                        fn_chunks.append(chunk_hyp)
                    else:
                        tn_chunks.append(chunk_hyp)

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

            src, ref = chunk_list_to_text(chunks_ref)
            LOGGER.debug(f"SRC: {src}")
            LOGGER.debug(f"REF: {ref}")
            LOGGER.debug(
                f"tp={len(tp_chunks)}, fp={len(fp_chunks)}, "
                f"fn={len(fn_chunks)}, tn={len(tn_chunks)}"
            )

        return SampleMetricResult(ref_results=ref_results)

    def evaluate_sample_detection(
        self, sample_hyp: Sample, sample_ref: Sample
    ) -> SampleMetricResult:
        pass
