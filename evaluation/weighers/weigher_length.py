import logging
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from data import Chunk, Dataset
from utils import get_logger

from ..schema import BaseChunkMetricResult, SampleMetricResult
from .weigher_base import BaseWeigher

LOGGER = get_logger(__name__, level=logging.INFO)


class LengthWeigher(BaseWeigher, BaseModel):
    """Compute edit weights by length.

    For TP, FN and TN, the longer is the edit, the greater is the weight.
    For FP, the longer is the edit, the smaller is the weight.

    For more details, refer to the following paper:
    CLEME: De-biasing Multi-reference Evaluation for Grammatical Error Correction [EMNLP 2023]

    """

    tp_alpha: float = Field(default=2.0, description="Scale factor of tp_edits")
    tp_bias: float = Field(default=0.0, description="Bias factor of tp_edits")
    tp_min_value: float = Field(default=0.75, description="Minimum weight of tp_edits")
    tp_max_value: float = Field(default=1.25, description="Maximum weight of tp_edits")

    fp_alpha: float = Field(default=2.0, description="Scale factor of fp_edits")
    fp_bias: float = Field(default=0.0, description="Bias factor of fp_edits")
    fp_min_value: float = Field(default=0.75, description="Minimum weight of fp_edits")
    fp_max_value: float = Field(default=1.25, description="Maximum weight of fp_edits")

    fn_alpha: float = Field(default=2.0, description="Scale factor of fn_edits")
    fn_bias: float = Field(default=0.0, description="Bias factor of fn_edits")
    fn_min_value: float = Field(default=0.75, description="Minimum weight of fn_edits")
    fn_max_value: float = Field(default=1.25, description="Maximum weight of fn_edits")

    tn_alpha: float = Field(default=2.0, description="Scale factor of tn_edits")
    tn_bias: float = Field(default=0.0, description="Bias factor of tn_edits")
    tn_min_value: float = Field(default=1.0, description="Minimum weight of tn_edits")
    tn_max_value: float = Field(default=1.0, description="Maximum weight of tn_edits")

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()

    def setup(self, dataset_ref: Dataset) -> None:
        # Calculate the average length of correct chunks (i.e., Unchanged Chunks)
        # and incorrect chunks (i.e., Corrected/Dummy Chunks)
        chunk_len_correct, chunk_len_incorrect = [], []

        for sample in dataset_ref:
            for chunks in sample.chunks[0]:
                for chunk in chunks:
                    if chunk.types:  # Corrected/Dummy Chunk
                        # chunk_len = (len(chunk.src_tokens) + len(chunk.tgt_tokens)) / 2
                        chunk_len = max(len(chunk.src_tokens), len(chunk.tgt_tokens))
                        chunk_len_incorrect.append(chunk_len)
                    else:  # Unchanged Chunk
                        chunk_len_correct.append(len(chunk.src_tokens))
        avg_chunk_len_correct = np.average(chunk_len_correct)
        avg_chunk_len_incorrect = np.average(chunk_len_incorrect)

        LOGGER.info(
            f"avg_chunk_len_correct={round(avg_chunk_len_correct, 2)}, "
            f"avg_chunk_len_incorrect={round(avg_chunk_len_incorrect, 2)}"
        )

        self.tp_bias = avg_chunk_len_incorrect
        self.fp_bias = avg_chunk_len_incorrect
        self.fn_bias = avg_chunk_len_incorrect
        self.tn_bias = avg_chunk_len_incorrect

    def __call__(self, metric_result: SampleMetricResult, **kwargs: Any) -> None:
        """Generate edit weights for SampleMetricResult."""
        for ref_result in metric_result.ref_results:
            assert isinstance(ref_result, BaseChunkMetricResult)

            for tp_chunk in ref_result.tp_chunks:
                tp_chunk.weight = self.weigh_edit(
                    chunk=tp_chunk,
                    alpha=self.tp_alpha,
                    bias=self.tp_bias,
                    min_value=self.tp_min_value,
                    max_value=self.tp_max_value,
                    reverse=False,
                )
                LOGGER.debug(f"TP: {tp_chunk}")
            for fp_chunk in ref_result.fp_chunks:
                fp_chunk.weight = self.weigh_edit(
                    chunk=fp_chunk,
                    alpha=self.fp_alpha,
                    bias=self.fp_bias,
                    min_value=self.fp_min_value,
                    max_value=self.fp_max_value,
                    reverse=True,
                )
                LOGGER.debug(f"FP: {fp_chunk}")
            for fp_ne_chunk in ref_result.fp_ne_chunks:
                fp_ne_chunk.weight = self.weigh_edit(
                    chunk=fp_ne_chunk,
                    alpha=self.fp_alpha,
                    bias=self.fp_bias,
                    min_value=self.fp_min_value,
                    max_value=self.fp_max_value,
                    reverse=True,
                )
                LOGGER.debug(f"FP_NE: {fp_ne_chunk}")
            for fp_un_chunk in ref_result.fp_un_chunks:
                fp_un_chunk.weight = self.weigh_edit(
                    chunk=fp_un_chunk,
                    alpha=self.fp_alpha,
                    bias=self.fp_bias,
                    min_value=self.fp_min_value,
                    max_value=self.fp_max_value,
                    reverse=True,
                )
                LOGGER.debug(f"FP_UN: {fp_un_chunk}")

            for fn_chunk in ref_result.fn_chunks:
                fn_chunk.weight = self.weigh_edit(
                    chunk=fn_chunk,
                    alpha=self.fn_alpha,
                    bias=self.fn_bias,
                    min_value=self.fn_min_value,
                    max_value=self.fn_max_value,
                    reverse=False,
                )
                LOGGER.debug(f"FN: {fn_chunk}")
            for tn_chunk in ref_result.tn_chunks:
                tn_chunk.weight = self.weigh_edit(
                    chunk=tn_chunk,
                    alpha=self.tn_alpha,
                    bias=self.tn_bias,
                    min_value=self.tn_min_value,
                    max_value=self.tn_max_value,
                    reverse=False,
                )
                # LOGGER.debug(f"TN: {tn_chunk}")

    @classmethod
    def weigh_edit(
        cls,
        chunk: Chunk,
        alpha: float,
        bias: float,
        min_value: float,
        max_value: float,
        reverse: bool,
    ) -> float:
        edit_len = max(len(chunk.src_tokens), len(chunk.tgt_tokens))
        if reverse:
            weight = alpha * (1 / (1 + (alpha - 1) * np.exp(edit_len - bias)))
        else:
            weight = alpha * (1 / (1 + (alpha - 1) * np.exp(-edit_len + bias)))
        return np.clip(weight, min_value, max_value).item()
