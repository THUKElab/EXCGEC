import copy
import sys
from typing import Any, List, TextIO, Tuple, Union

import numpy as np
from tabulate import tabulate

from data import Chunk, Dataset, Edit, Sample
from utils import get_logger

from ...aligners import AlignerType
from ...classifers import ClassifierType
from ...mergers import MergerType, MergeStrategy
from ...schema import SampleMetricResult
from ...scorers import ScorerType
from ...tokenizers import TokenizerType
from ...weighers import LengthWeigher, WeigherType, get_weigher
from ..base import BaseEditMetric
from ..errant import Errant
from .cleme_utils import convert_edit_into_chunk, map_parallel, merge_edit

LOGGER = get_logger(__name__)


class CLEME(BaseEditMetric):
    """Evaluate data with multiple references unbiasly.

    For more details, refer to the following paper:
    CLEME: De-biasing Multi-reference Evaluation for Grammatical Error Correction [EMNLP 2023]
    """

    def __init__(
        self,
        lang: str,
        scorer_type: ScorerType = ScorerType.PRF,
        weigher_type: WeigherType = WeigherType.NONE,
        weigher_model_name: str = None,
        weigher_model_layer: int = None,
        tokenizer_type: TokenizerType = None,
        aligner_type: AlignerType = None,
        aligner_standard: bool = False,
        merger_type: MergerType = None,
        merger_strategy: MergeStrategy = None,
        classifier_type: ClassifierType = None,
        enable_tqdm: bool = True,
        merge_distance: int = 0,
        output_visualize: Union[str, TextIO] = None,
        **kwargs: Any,
    ) -> None:
        tokenizer_type = tokenizer_type or lang
        aligner_type = aligner_type or lang
        merger_type = merger_type or lang
        classifier_type = classifier_type or lang

        super().__init__(
            lang=lang,
            tokenizer_type=tokenizer_type,
            scorer_type=scorer_type,
            enable_tqdm=enable_tqdm,
            **kwargs,
        )
        self.errant = Errant(
            lang=lang,
            tokenizer_type=tokenizer_type,
            aligner_type=aligner_type,
            aligner_standard=aligner_standard,
            merger_type=merger_type,
            merger_strategy=merger_strategy,
            classifier_type=classifier_type,
            scorer_type=scorer_type,
            enable_tqdm=enable_tqdm,
        )
        self.weigher = get_weigher(
            weigher_type, model_name=weigher_model_name, model_layer=weigher_model_layer
        )
        self.merge_distance = merge_distance
        self.output_visualize = output_visualize

    def parallel_to_edits(self, sample: Sample) -> List[List[List[Edit]]]:
        return self.errant.parallel_to_edits(sample=sample)

    def prepare_datasets(
        self, dataset_hyp: Dataset, dataset_ref: Dataset
    ) -> Tuple[Dataset, Dataset]:
        """Prepare datasets for chunk-level evaluation.

        1) Acquire Edits using Errant.
        2) Chunk partition.
        2) Compute average chunk length.

        Args:
            dataset_hyp (Dataset): Hyp dataset.
            dataset_ref (Dataset): Ref dataset.
            num_workers (int): _description_. Defaults to 1.
            write_hyp_m2 (str): _description_. Defaults to None.
            write_ref_m2 (str): _description_. Defaults to None.

        Returns:
            Tuple[Dataset, Dataset]: Prepared dataset with chunk partition.
        """
        # Extract edits
        dataset_hyp, dataset_ref = self.errant.prepare_datasets(
            dataset_hyp=dataset_hyp, dataset_ref=dataset_ref
        )

        # Remove empty references to get rid of chunk partition collapse
        for sample in dataset_ref:
            valid_target_indices = [i for i, x in enumerate(sample.target) if x]
            if len(valid_target_indices) != len(sample.target):
                LOGGER.warning(f"Remove empty reference in {sample}")
                if len(valid_target_indices) == 0:
                    valid_target_indices = [0]
                sample.target = [sample.target[x] for x in valid_target_indices]
                sample.edits[0] = [sample.edits[0][x] for x in valid_target_indices]

        # Merge dataset_hyp and dataset_ref into one dataset for convenience
        merge_data = copy.deepcopy(dataset_hyp)
        for sample_idx, sample in enumerate(merge_data):
            sample.target.extend(dataset_ref[sample_idx].target)
            sample.edits[0].extend(copy.deepcopy(dataset_ref[sample_idx].edits[0]))

        # Chunk partition
        chunk_dataset = self.chunk_partition(
            merge_data, merge_distance=self.merge_distance
        )
        for sample_chunk, sample_hyp, sample_ref in zip(
            chunk_dataset, dataset_hyp, dataset_ref
        ):
            assert len(chunk_dataset[sample_idx]) > 1
            sample_hyp.chunks = [[sample_chunk[0]]]
            sample_ref.chunks = [sample_chunk[1:]]

        # Visualize chunk partition
        if self.output_visualize:
            sout = self.output_visualize
            if isinstance(sout, str):
                sout = open(sout, "r", encoding="utf-8")
            self.visualize(
                merge_data,
                chunk_dataset=chunk_dataset,
                sout=sout,
                delimiter="" if self.lang == "zho" else "",
            )
            if isinstance(self.output_visualize, str):
                sout.close()

        # Prepare LengthWeigher
        if isinstance(self.weigher, LengthWeigher):
            self.setup_length_weigher(dataset_ref)

        return dataset_hyp, dataset_ref

    def setup_length_weigher(self, dataset_ref: Dataset) -> None:
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

        if isinstance(self.weigher, LengthWeigher):
            self.weigher.tp_bias = avg_chunk_len_incorrect
            self.weigher.fp_bias = avg_chunk_len_incorrect
            self.weigher.fn_bias = avg_chunk_len_incorrect
            self.weigher.tn_bias = avg_chunk_len_incorrect

    def post_metric_evaluation(
        self,
        dataset_hyp: Dataset,
        dataset_ref: Dataset,
        metric_results: List[SampleMetricResult],
    ) -> None:
        """Compute edit weights for metric_results.

        Args:
            dataset_hyp (Dataset): Hyp dataset.
            dataset_ref (Dataset): Ref dataset.
            metric_results (List[MetricSampleResult]): Results of metric.
        """
        if self.weigher is not None:
            self.weigher.weigh_batch(
                samples_hyp=dataset_hyp.samples,
                samples_ref=dataset_ref.samples,
                metric_results=metric_results,
            )

    def chunk_partition(
        self, dataset: Dataset, merge_distance: int = 0
    ) -> List[List[List[Chunk]]]:
        """Segment the source, hypothesis and references into chunk sequences.

        NOTE: Rewrite the function for higher efficiency.

        1) Construct token_mapping
        2) Merge edits with overlapping interval
        3) Convert edit into chunk

        Args:
            dataset (Dataset): Input dataset
            merge_distance (int): Maximum merging distance of two adjacent edits. Defaults to 0.

        Returns:
            List[List[List[Chunk]]]: Segmented chunks.
        """
        chunk_list_dataset = []
        for sample in dataset:
            # Segment sentence
            src_tokens = self.errant.tokenizer.segment(sample.source[0])
            tgt_tokens_list = [self.errant.tokenizer.segment(x) for x in sample.target]

            # Construct token_mapping
            edits_list, token_mapping_total = [], []
            for tgt_idx in range(len(sample.target)):
                edits = sample.edits[0][tgt_idx]
                edits = sorted(edits, key=lambda x: x.src_interval[0])
                edits_list.append(edits)
                token_mapping = map_parallel(src_tokens, edits)
                token_mapping_total.append(token_mapping)

            # Merge edits with overlapping interval
            merge_edits_list, shared_interval_list = merge_edit(
                src_tokens,
                tgt_tokens_list,
                edits_list,
                token_mapping_total,
                merge_distance=merge_distance,
            )

            # Convert edit into chunk
            chunk_list_total = convert_edit_into_chunk(
                src_tokens,
                tgt_tokens_list,
                merge_edits_list,
                shared_interval_list,
                token_mapping_total,
            )
            chunk_list_dataset.append(chunk_list_total)
        return chunk_list_dataset

    def visualize(
        self,
        dataset: Dataset,
        chunk_dataset: List[List[List[Chunk]]] = None,
        sout: Union[str, TextIO] = sys.stdout,
        show_types: bool = False,
        delimiter: str = " ",
        **kwargs: Any,
    ) -> None:
        """Visualize the results of chunk partition into output stream.

        tabular_data = {
            "sentence": [],
            "chunk-0": [],
            "chunk-1": [],
            "chunk-T": [],
            "chunk-N": [],
        }
        """

        if chunk_dataset is None:
            chunk_dataset = self.chunk_partition(dataset)

        for chunk_sample in chunk_dataset:
            tabular_data = {
                "sentence": ["source"]
                + [f"target-{x}" for x in range(len(chunk_sample))],
            }
            for chunk_idx in range(len(chunk_sample[0])):
                chunks = [delimiter.join(chunk_sample[0][chunk_idx].src_tokens)] + [
                    delimiter.join(x[chunk_idx].tgt_tokens) for x in chunk_sample
                ]
                if len(set(chunks)) > 1:
                    head_name = f"chunk-{chunk_idx} *"
                else:
                    head_name = f"chunk-{chunk_idx}"
                tabular_data[head_name] = chunks

                if show_types and len(set(chunks)) > 1:
                    types = [""] + [" ".join(x[chunk_idx].types) for x in chunk_sample]
                    tabular_data[f"Types-{chunk_idx}"] = types

            table = tabulate(
                tabular_data,
                tablefmt="fancy_grid",
                headers="keys",
                floatfmt=".3f",
                missingval="N/A",
                numalign="left",
            )
            sout.write("\n" + table + "\n")
            for k, v in kwargs.items():
                sout.write(f"{k}: {v}\n")
            sout.write("\n")
