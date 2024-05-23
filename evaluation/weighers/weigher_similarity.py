import copy
from itertools import chain
from typing import List

from bert_score import BERTScorer

from data import Chunk, Sample
from utils import get_logger

from ..schema import BaseChunkMetricResult, SampleMetricResult
from .weigher_base import BaseWeigher

LOGGER = get_logger(__name__)


class SimilarityWeigher(BaseWeigher):
    """Compute edit weights using BERTScore.

    Args:
        model_name (str, optional): Model name or path for BERTScore.
            Defaults to DEFAULT_MODEL_NAME. Recommend different default_model for languages.
        model_layer (int, optional): Number of layers for BERTScore. Defaults to None.
        batch_size (int, optional): Number of samples per batch. Defaults to 128.
        device (str, optional): Computing device. Defaults to None.
        verbose (bool, optional): Whether to print debugging info. Defaults to False.
        show_progress (bool, optional): Whether to show progress. Defaults to False.
    """

    DEFAULT_MODEL_NAME = "bert-base-uncased"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        model_layer: int = None,
        batch_size: int = 128,
        device: str = None,
        verbose: bool = False,
        show_progress: bool = False,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.model_layer = model_layer
        self.batch_size = batch_size
        self.verbose = verbose
        self.show_progress = show_progress
        # TODO: Determine the best num_layers
        self.model = BERTScorer(
            model_type=self.model_name,
            num_layers=self.model_layer,
            batch_size=batch_size,
            device=device,
        )

        LOGGER.info(
            f"Similarity weigher driven by {self.model_name}, Layers: {self.model._num_layers}"
        )

    @DeprecationWarning
    def __call__(
        self,
        sample_hyp: Sample,
        sample_ref: Sample,
        metric_result: SampleMetricResult,
        for_chunk: bool = True,
    ) -> None:
        """Generate weights for a single MetricSampleResult.

        Not recommended, use self.get_weights_batch instead if you have a pile of samples.
        """
        if not for_chunk:
            raise ValueError("Only support for_chunk = True")

        src = sample_hyp.source[0]
        for ref, ref_result in zip(sample_ref.target, metric_result.ref_results):
            todo_hyps = []
            # Build pseudo hyp that
            assert isinstance(ref_result, BaseChunkMetricResult)
            todo_chunks = (
                ref_result.tp_chunks
                + ref_result.fp_chunks
                + ref_result.fn_chunks
                + ref_result.tn_chunks
                + ref_result.fp_ne_chunks
                + ref_result.fp_un_chunks
            )
            for chunk in todo_chunks:
                todo_hyps.append(
                    src_chunks_to_text(
                        chunks=sample_hyp.chunks[0][0], chunk_index=chunk.chunk_index
                    )
                )
            weights = self._get_weights_sample(
                src=src, ref=ref, todo_hyps=todo_hyps, verbose=self.verbose
            )
            for chunk, weight in zip(todo_chunks, weights):
                chunk.weight = weight
            print(todo_chunks)

    def _get_weights_sample(
        self, src: str, ref: str, todo_hyps: List[str]
    ) -> List[float]:
        """Generate weights by using BERTScorer.

        Args:
            src (str): Source sentence.
            ref (str): Reference sentence.
            todo_hyps (List[str]): Pseudo hypothesis sentences.

        Returns:
            List[float]: Generated edit weights, with each corresponding to each todo_hyp.
        """
        processed_hyps = [src] + todo_hyps
        processed_refs = [ref] * len(todo_hyps)

        fscores = self.model.score(
            cands=processed_hyps, refs=processed_refs, verbose=self.verbose
        )[-1]
        anchor = fscores[0].item()
        return [abs(anchor - x.item()) for x in fscores[1:]]

    def get_weights_batch(
        self,
        samples_hyp: List[Sample],
        samples_ref: List[Sample],
        metric_results: List[List[SampleMetricResult]],
    ) -> None:
        """Generate weights for batch MetricSampleResults. Inspired by PT-M2."""

        srcs: List[str] = []
        refs: List[str] = []
        todo_hyps_list: List[List[str]] = []
        todo_chunks_list: List[List[Chunk]] = []

        # Compute similarities between sources and targets
        self.get_similarity(samples=samples_ref, metric_results=metric_results)

        for sample_hyp, sample_ref, metric_result in zip(
            samples_hyp, samples_ref, metric_results
        ):
            for ref_idx, ref_result in enumerate(metric_result.ref_results):
                # Build pseudo hyp that
                assert isinstance(ref_result, BaseChunkMetricResult)
                for chunk in ref_result.tn_chunks:
                    chunk.weight = 1.0

                todo_hyps = []
                todo_chunks = (
                    ref_result.tp_chunks
                    + ref_result.fp_chunks
                    + ref_result.fp_ne_chunks
                    + ref_result.fp_un_chunks
                )
                for chunk in todo_chunks:
                    todo_hyps.append(
                        src_chunks_to_text(
                            chunks=sample_hyp.chunks[0][0],
                            chunk_index=chunk.chunk_index,
                        )
                    )
                for chunk in ref_result.fn_chunks:
                    todo_hyps.append(
                        src_chunks_to_text(
                            chunks=sample_ref.chunks[0][ref_idx],
                            chunk_index=chunk.chunk_index,
                        )
                    )
                if todo_hyps:  # Only consider samples with edits
                    srcs.append(sample_hyp.source[0])
                    refs.append(sample_ref.target[ref_idx])
                    todo_hyps_list.append(todo_hyps)
                    todo_chunks_list.append(todo_chunks + ref_result.fn_chunks)

        # Weigh all the edits in batch
        weights_list = self.get_weights_batch(
            srcs=srcs, refs=refs, todo_hyps_list=todo_hyps_list
        )
        for chunks, weights in zip(todo_chunks_list, weights_list):
            for chunk, weight in zip(chunks, weights):
                chunk.weight = weight

        if self.verbose:
            for sample_hyp, sample_ref, metric_result in zip(
                samples_hyp, samples_ref, metric_results
            ):
                for ref, ref_result in zip(
                    sample_ref.target, metric_result.ref_results
                ):
                    print(f"SRC: {sample_hyp.source[0]}")
                    print(f"HYP: {sample_hyp.target[0]}")
                    print(f"REF: {ref}")
                    print(f"TP Chunks: {ref_result.tp_chunks}")
                    print(f"FP Chunks: {ref_result.fp_chunks}")
                    print(f"FN Chunks: {ref_result.fn_chunks}")
                    # print(f"TN Chunks: {ref_result.tn_chunks}")
                    print(f"FP_NE Chunks: {ref_result.fp_ne_chunks}")
                    print(f"FP_UN Chunks: {ref_result.fp_un_chunks}")
                    print()

    def get_weights_batch_v2(
        self,
        samples_hyp: List[Sample],
        samples_ref: List[Sample],
        metric_results: List[List[SampleMetricResult]],
    ) -> None:
        """Generate weights for batch MetricSampleResults."""

        cands: List[str] = []
        refs: List[str] = []
        todo_chunks: List[Chunk] = []

        for sample_hyp, sample_ref, metric_result in zip(
            samples_hyp, samples_ref, metric_results
        ):
            for ref_idx, ref_result in enumerate(metric_result.ref_results):
                assert isinstance(ref_result, BaseChunkMetricResult)
                for chunk in ref_result.tn_chunks:
                    chunk.weight = 1.0

                for chunk in ref_result.tp_chunks:
                    # Apply the chunk on the source
                    src = sample_hyp.source[0]
                    src_post = src_chunks_to_text(
                        chunks=sample_hyp.chunks[0][0], chunk_index=chunk.chunk_index
                    )
                    cands.append(src)
                    refs.append(src_post)
                    todo_chunks.append(chunk)

                for chunk in (
                    ref_result.fn_chunks
                    + ref_result.fp_un_chunks
                    + ref_result.fp_ne_chunks
                ):
                    # Apply reversely the chunk on the reference
                    cand_chunks = sample_ref.chunks[0][ref_idx]
                    cand_chunks[chunk.chunk_index] = sample_hyp.chunks[0][0][
                        chunk.chunk_index
                    ]
                    cand_tokens = list(chain(*[x.tgt_tokens for x in cand_chunks]))
                    cand_tokens = list(filter(None, cand_tokens))
                    # NOTE: Delimiter may change for different languages
                    cand = " ".join(cand_tokens)
                    cands.append(cand)
                    refs.append(sample_ref.target[ref_idx])
                    todo_chunks.append(chunk)

        # Compute sentence similarity
        weights = self.model.score(cands=cands, refs=refs, verbose=self.verbose)[-1]
        for chunk, weight in zip(todo_chunks, weights):
            chunk.weight = 1.0 - weight.item()

        for metric_result in metric_results:
            for ref_result in metric_result.ref_results:
                ref_result.fp_chunks = copy.deepcopy(
                    ref_result.fp_ne_chunks + ref_result.fp_un_chunks
                )
                ref_result.fp_chunks.sort(key=lambda x: x.chunk_index)

        if self.verbose:
            for sample_hyp, sample_ref, metric_result in zip(
                samples_hyp, samples_ref, metric_results
            ):
                for ref, ref_result in zip(
                    sample_ref.target, metric_result.ref_results
                ):
                    print(f"SRC: {sample_hyp.source[0]}")
                    print(f"HYP: {sample_hyp.target[0]}")
                    print(f"REF: {ref}")
                    print(f"TP Chunks: {ref_result.tp_chunks}")
                    print(f"FP Chunks: {ref_result.fp_chunks}")
                    print(f"FN Chunks: {ref_result.fn_chunks}")
                    # print(f"TN Chunks: {ref_result.tn_chunks}")
                    print(f"FP_NE Chunks: {ref_result.fp_ne_chunks}")
                    print(f"FP_UN Chunks: {ref_result.fp_un_chunks}")
                    print()

    def _get_weights_batch(
        self, srcs: List[str], refs: List[str], todo_hyps_list: List[List[str]]
    ) -> List[List[float]]:
        """Generate weights in batch by using BERTScorer. Weigh edits through similarity increase.

        Args:
            srcs (List[str]): Source sentences.
            refs (List[str]): Reference sentences.
            todo_hyps (List[List[str]]): Pseudo hypothesis sentences.

        Returns:
            List[List[float]]: Generated edit weights, with each corresponding to each todo_hyp.
        """
        if len(srcs) != len(refs) != len(todo_hyps_list):
            raise ValueError("The input sentences should consist of the same number")

        processed_hyps = []
        processed_refs = []
        anchor_indices = []  # Save the weights for the pair (src, ref)
        for src, ref, todo_hyps in zip(srcs, refs, todo_hyps_list):
            for idx, hyp in enumerate(todo_hyps):
                if not hyp:
                    LOGGER.warning(
                        f"Empty Sentences for weigher\n"
                        f"SRC:{src}\nREF:{ref}\nHYPs:{todo_hyps}"
                    )
                    todo_hyps[idx] = src

            anchor_indices.append(len(processed_hyps))
            processed_hyps.extend([src] + todo_hyps)
            processed_refs.extend([ref] * (1 + len(todo_hyps)))

        # Compute sentence similarity
        fscores = self.model.score(
            cands=processed_hyps, refs=processed_refs, verbose=self.verbose
        )[-1]

        # print(f"srcs: {srcs}")
        # print(f"refs: {refs}")
        # print(f"todo_hyps_list: {todo_hyps_list}")
        # print(fscores)

        # Compute edit weights
        weights_list = []
        for idx, anchor_idx in enumerate(anchor_indices):
            num_sent = len(todo_hyps_list[idx])
            anchor = fscores[anchor_idx].item()
            weights = [
                abs(anchor - x.item())
                for x in fscores[anchor_idx + 1 : anchor_idx + 1 + num_sent]
            ]
            weights_list.append(weights)
        return weights_list

    def get_similarity(
        self,
        samples: List[Sample],
        metric_results: List[List[SampleMetricResult]],
        verbose: bool = False,
    ) -> List[float]:
        srcs, tgts = [], []
        for sample in samples:
            src = sample.source[0]
            for tgt in sample.target:
                if not tgt:
                    LOGGER.warning(f"Empty Target: {sample}")
                srcs.append(src)
                tgts.append(tgt)

        # Compute sentence similarity
        fscores = self.model.score(cands=srcs, refs=tgts, batch_size=self.batch_size)
        fscores = fscores[-1].tolist()

        idx = 0
        for sample, metric_result in zip(samples, metric_results):
            if len(sample.target) != len(metric_result.ref_results):
                raise ValueError("Unequal results")
            for ref_result in metric_result.ref_results:
                ref_result.sim_src_tgt = fscores[idx]
                idx += 1
        if idx != len(fscores):
            raise ValueError()

        if verbose:
            for src, tgt, fscore in zip(srcs, tgts, fscores):
                print(f"Source: {src}")
                print(f"Target: {tgt}")
                print(f"Similarity: {fscore}")
        return fscores


def src_chunks_to_text(chunks: List[Chunk], chunk_index: int, limiter=" ") -> str:
    tokens = []
    for idx, chunk in enumerate(chunks):
        if idx == chunk_index:
            tokens.extend(chunk.tgt_tokens)
        else:
            tokens.extend(chunk.src_tokens)
    tokens = list(filter(None, tokens))
    return limiter.join(tokens)


def tgt_chunks_to_text(chunks: List[Chunk], chunk_index: int, limiter=" ") -> str:
    tokens = []
    for idx, chunk in enumerate(chunks):
        if idx == chunk_index:
            tokens.extend(chunk.src_tokens)
        else:
            tokens.extend(chunk.tgt_tokens)
    tokens = list(filter(None, tokens))
    return limiter.join(tokens)
