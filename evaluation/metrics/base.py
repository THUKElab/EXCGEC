"""
`Metric` is an abstract class that enforces the implementation of a set of abstract methods,
so that a correctly implemented metric will work seamlessly with the rest of the codebase.

BaseMetric                              # Abstract Metric Class
  ├── NGramMetric                       # N-gram based Metric  i.e., GLEU
  |   └── GLEUMetric
  └── EditMetric                        # Edit-based Metric, including MaxMatch(M2), ERRANT
      ├── MaxMatch                      # Dynamic Programming based Metric
      ├── Errant                        # Linguistic-enhanced Metric
      └── CLEME                         # Chunk-based Metric, i.e, CLEME
          ├── DependentCLEME            # CLEME-dependent
          └── IndependentCLEME          # CLEME-independent
"""

import copy
import json
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

from data import Dataset, Edit, Sample
from utils import get_logger, get_tqdm_iterable

from ..schema import OverallScorerResult, SampleMetricResult
from ..scorers import ScorerType, get_scorer
from ..tokenizers import TokenizerType, get_tokenizer

LOGGER = get_logger(__name__)


class BaseMetric(ABC):
    def __init__(
        self,
        lang: str,
        tokenizer_type: TokenizerType,
        scorer_type: ScorerType,
        enable_tqdm: bool = True,
        table_print: bool = True,
        remove_unchanged_reference: bool = False,
    ) -> None:
        self.lang = lang
        self.tokenizer = get_tokenizer(tokenizer_type)
        self.scorer = get_scorer(scorer_type, table_print=table_print)
        self.enable_tqdm = enable_tqdm
        self.remove_unchanged = remove_unchanged_reference

    @property
    def classname(cls) -> str:
        return cls.__class__.__name__

    def prepare_datasets(
        self, dataset_hyp: Dataset, dataset_ref: Dataset
    ) -> Tuple[Dataset, Dataset]:
        self.check_datasets(dataset_hyp, dataset_ref)
        if self.remove_unchanged:
            dataset_ref = self.remove_unchanged_reference(dataset_ref)
        return dataset_hyp, dataset_ref

    @classmethod
    def check_datasets(cls, dataset_hyp: Dataset, dataset_ref: Dataset) -> None:
        if len(dataset_hyp) != len(dataset_ref):
            raise ValueError("Unequal source numbers for datasets.")

        for sample_hyp, sample_ref in zip(dataset_hyp, dataset_ref):
            if len(sample_hyp.source) != 1 or len(sample_ref.source) != 1:
                raise ValueError(
                    f"Each hyp_sample must have a single source: {sample_hyp}"
                )
            if sample_hyp.source[0] != sample_ref.source[0]:
                raise ValueError(
                    f"Sources of hyp and ref must be the same:\n"
                    f"hyp_source={sample_hyp.source[0]}\n"
                    f"ref_source={sample_ref.source[0]}"
                )
            if not sample_hyp.source[0]:
                raise ValueError("Source cannot be empty.")
            if len(sample_hyp.target) != 1:
                raise ValueError("The number of hyp target must be one.")
            if len(sample_ref.target) == 0:
                raise ValueError(f"No references for sample_{sample_ref.index}.")

    def remove_unchanged_reference(self, dataset: Dataset) -> Dataset:
        """Remove unchanged reference.

        Args:
            dataset (Dataset): Input dataset.

        Returns:
            Dataset: Output dataset.
        """
        new_dataset = copy.deepcopy(dataset)
        for sample in new_dataset:
            if len(sample.target) == 1:
                continue
            src = sample.source[0]
            valid_target_indices = [i for i, x in enumerate(sample.target) if x != src]
            if len(valid_target_indices) != len(sample.target):
                LOGGER.warning(f"Remove unchanged reference in {sample}")
                if len(valid_target_indices) == 0:
                    valid_target_indices = [0]
                sample.target = [sample.target[x] for x in valid_target_indices]
                sample.edits[0] = [sample.edits[0][x] for x in valid_target_indices]
        return new_dataset

    def evaluate(
        self, dataset_hyp: Dataset, dataset_ref: Dataset, persist_path: str = None
    ) -> Tuple[OverallScorerResult, List[SampleMetricResult]]:
        start_time = time.time()
        dataset_hyp, dataset_ref = self.prepare_datasets(
            dataset_hyp=dataset_hyp, dataset_ref=dataset_ref
        )
        prepare_time = time.time() - start_time

        # futures = []
        # executor = ProcessPoolExecutor(max_workers=num_workers)
        # for sample_hyp, sample_ref in zip(dataset_hyp, dataset_ref):
        #     future = executor.submit(
        #         self.evaluate_sample, sample_hyp, sample_ref, **kwargs
        #     )
        #     futures.append(future)

        # iterator = futures
        # if self.enable_tqdm:
        #     iterator = tqdm(
        #         iterator,
        #         total=len(futures),
        #         desc=f"{self.classname} Evaluate by {num_workers} workers",
        #     )
        # results = [future.result() for future in iterator]
        # executor.shutdown(wait=True)

        queue_with_progress = get_tqdm_iterable(
            items=zip(dataset_hyp, dataset_ref),
            show_progress=self.enable_tqdm,
            desc=f"{self.classname} Evaluating",
        )

        # Acquire metric results
        metric_results: List[SampleMetricResult] = []
        for sample_hyp, sample_ref in queue_with_progress:
            result = self.evaluate_sample(sample_hyp, sample_ref)
            metric_results.append(result)

        # Post process the metric results
        self.post_metric_evaluation(
            dataset_hyp=dataset_hyp,
            dataset_ref=dataset_ref,
            metric_results=metric_results,
        )

        # Acquire score results
        score_result: OverallScorerResult = self.scorer(
            dataset_hyp=dataset_hyp,
            dataset_ref=dataset_ref,
            metric_results=metric_results,
        )
        LOGGER.info(
            "{} Total samples: {}, Total time: {:.3f} seconds; Preparation time: {:.3f}".format(
                self.classname, len(dataset_hyp), time.time() - start_time, prepare_time
            )
        )

        # Save datasets and results
        if persist_path is not None:
            self.persist(
                dataset_hyp=dataset_hyp,
                persist_path=persist_path,
                score_result=score_result,
                metric_results=metric_results,
            )
        return score_result, metric_results

    def post_metric_evaluation(self, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def evaluate_sample(self, *args, **kwargs) -> List[Dict[str, int]]:
        raise NotImplementedError

    @abstractmethod
    def persist(
        self,
        dataset_hyp: Dataset,
        persist_path: str,
        score_result: OverallScorerResult,
        metric_results: List[SampleMetricResult],
    ) -> None:
        raise NotImplementedError


class BaseEditMetric(BaseMetric):
    def prepare_dataset(self, dataset: Dataset, num_workers: int = 1) -> Dataset:
        # iterator = dataset
        # if self.enable_tqdm:
        #     iterator = tqdm(dataset, total=len(dataset), desc="Tokenizing")
        # for sample in iterator:
        #     sample.source_tokens = [self.tokenizer(source) for source in sample.source]
        #     sample.target_tokens = [self.tokenizer(target) for target in sample.target]

        # futures = []
        # executor = ProcessPoolExecutor(max_workers=num_workers)
        # for sample in dataset:
        #     future = executor.submit(self.prepare_edits, sample)
        #     futures.append(future)

        # iterator = zip(dataset, futures)
        # if self.enable_tqdm:
        #     iterator = tqdm(
        #         iterator,
        #         total=len(futures),
        #         desc=f"{self.classname} Preparing Dataset by {num_workers} workers",
        #     )
        # for sample, future in iterator:
        #     edits = future.result()
        #     if edits is not None:
        #         sample._edits = edits
        # executor.shutdown(wait=True)

        queue_with_progress = get_tqdm_iterable(
            items=dataset.samples,
            show_progress=self.enable_tqdm,
            desc=f"{self.classname} preparing edits",
        )

        for sample in queue_with_progress:
            if not sample.edits:
                sample.edits = self.parallel_to_edits(sample)
        return dataset

    def prepare_datasets(
        self, dataset_hyp: Dataset, dataset_ref: Dataset
    ) -> Tuple[Dataset, Dataset]:
        super().prepare_datasets(dataset_hyp=dataset_hyp, dataset_ref=dataset_ref)
        dataset_hyp = self.prepare_dataset(dataset_hyp)
        dataset_ref = self.prepare_dataset(dataset_ref)
        return dataset_hyp, dataset_ref

    def evaluate_sample(
        self, sample_hyp: Sample, sample_ref: Sample
    ) -> SampleMetricResult:
        # Calculate TP, FP and FN counts
        return self.evaluate_sample_correction(
            sample_hyp=sample_hyp, sample_ref=sample_ref
        )

    @abstractmethod
    def parallel_to_edits(self, sample: Sample) -> List[List[List[Edit]]]:
        raise NotImplementedError

    @abstractmethod
    def evaluate_sample_correction(
        self, sample_hyp: Sample, sample_ref: Sample
    ) -> SampleMetricResult:
        raise NotImplementedError

    @abstractmethod
    def evaluate_sample_detection(
        self, sample_hyp: Sample, sample_ref: Sample
    ) -> SampleMetricResult:
        raise NotImplementedError

    def persist(
        self,
        dataset_hyp: Dataset,
        persist_path: str,
        score_result: OverallScorerResult,
        metric_results: List[SampleMetricResult],
    ) -> None:
        persist_json = {
            "scores": score_result.dict(),
            "samples": dataset_hyp.dict()["samples"],
        }
        for sample, metric_result in zip(persist_json["samples"], metric_results):
            sample.pop("edits")
            sample.pop("chunks")
            sample["metric_result"] = metric_result.dict()
        with open(persist_path, "w", encoding="utf-8") as f:
            json.dump(persist_json, f, indent=2, ensure_ascii=False)
