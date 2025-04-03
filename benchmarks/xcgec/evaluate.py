"""The file is for evaluation of EXGEC


在NLP当中，不同的评价指标，BLEU, METEOR, ROUGE和CIDEr的逻辑意义？
https://www.zhihu.com/question/304798594


用 Python 计算文本 BLEU 分数和 ROUGE 值
https://xiaosheng.blog/2020/08/13/calculate-bleu-and-rouge

"""

from typing import Any, Dict, List
from collections import defaultdict
from tabulate import tabulate
import pandas as pd
import io

import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score

# from rouge import Rouge
from rouge_chinese import Rouge
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    precision_recall_fscore_support,
)

from benchmarks.xcgec.objects import (
    VALID_ERROR_TYPES,
    XDataset,
    XSample,
    convert_dataset,
    convert_dataset_2,
)
from benchmarks.xcgec.objects_eval import (
    BaseExplanationMetricResult,
    SampleExplanationMetricResult,
)
from data import Dataset
from evaluation import DependentCLEME, ScorerType, WeigherType
from utils import get_logger, remove_space

LOGGER = get_logger(__name__)


def get_chunked_dataset(
    dataset: XDataset, merge_distance: int = 1, output_visualize: str = None
) -> Dataset:
    """Build dataset with chunks.

    Args:
        dataset (XDataset): Input XDataset.
        merge_distance (int, optional): Maximum interval of two ajacent edits. Defaults to 1.
        output_visualize (str, optional): Output file of visualization. Defaults to None.

    Returns:
        Dataset: Chunked dataset.
    """
    # Convert XDataset into conventional Dataset
    gec_dataset = convert_dataset(dataset=dataset)

    metric = DependentCLEME(
        lang="zho",
        scorer_type=ScorerType.PRF,
        weigher_type=WeigherType.LENGTH,
        output_visualize=output_visualize,
        merge_distance=merge_distance,
        )
    metric.prepare_dataset(gec_dataset)

    # Chunk partition
    chunk_dataset = metric.chunk_partition(
        dataset=gec_dataset, merge_distance=merge_distance
    )
    for sample_chunk, gec_sample in zip(chunk_dataset, gec_dataset):
        gec_sample.chunks = [sample_chunk]

    # Chunk visualization
    #if output_visualize:
    #    sout = output_visualize
    #    if isinstance(sout, str):
    #        sout = open(sout, "r", encoding="utf-8")
    #    metric.visualize(dataset=dataset, chunk_dataset=chunk_dataset, sout=sout)
    #    if isinstance(output_visualize, str):
    #        sout.close()

    return gec_dataset

def get_chunked_dataset_2(
    dataset: XDataset, merge_distance: int = 1, output_visualize: str = None
) -> Dataset:
    """Build dataset with chunks.

    Args:
        dataset (XDataset): Input XDataset.
        merge_distance (int, optional): Maximum interval of two ajacent edits. Defaults to 1.
        output_visualize (str, optional): Output file of visualization. Defaults to None.

    Returns:
        Dataset: Chunked dataset.
    """
    # Convert XDataset into conventional Dataset
    gec_dataset = convert_dataset_2(dataset=dataset)

    metric = DependentCLEME(
        lang="zho",
        scorer_type=ScorerType.PRF,
        weigher_type=WeigherType.LENGTH,
        output_visualize=output_visualize,
        merge_distance=merge_distance,
        )
    metric.prepare_dataset_2(gec_dataset)

    # Chunk partition
    chunk_dataset = metric.chunk_partition_2(
        dataset=gec_dataset, merge_distance=merge_distance
    )
    for sample_chunk, gec_sample in zip(chunk_dataset, gec_dataset):
        gec_sample.chunks = [sample_chunk]

    # Chunk visualization
    if output_visualize:
        sout = output_visualize
        if isinstance(sout, str):
            sout = open(sout, "w", encoding="utf-8")
        metric.visualize(dataset=dataset, chunk_dataset=chunk_dataset, sout=sout)
        if isinstance(output_visualize, str):
            sout.close()

    return gec_dataset


def check_dataset(dataset: XDataset) -> None:
    """Check the validity of dataset.

    Args:
        dataset (XDataset): Input dataset.
    """

    for sample in dataset:
        if not sample.source:
            raise ValueError(f"Empty source: {sample}")

        for edit in sample.edits:
            if edit.error_type not in VALID_ERROR_TYPES:
                raise ValueError(f"Invalid error type {edit.error_type}: {sample}")
            if edit.error_severity not in [1, 2, 3, 4, 5]:
                raise ValueError(
                    f"Invalid error severity {edit.error_severity}: {sample}"
                )

            if not edit.src_interval or not edit.tgt_interval:
                raise ValueError(f"None interval: {sample}")


def evaluate(dataset_hyp: XDataset, dataset_ref: XDataset) -> Dict[str, Any]:
    check_dataset(dataset_hyp)
    check_dataset(dataset_ref)
    scores = {}
    scores["gec"] = evaluate_gec(dataset_hyp=dataset_hyp, dataset_ref=dataset_ref)
    scores["exp"] = evaluate_exp(dataset_ref=dataset_ref, dataset_hyp=dataset_hyp)
    return scores


def evaluate_gec(
    dataset_hyp: XDataset,
    dataset_ref: XDataset,
    lang: str = "zho",
    merge_distance: int = 1,
    output_visualize: str = None,
    output_evaluation: str = None,
) -> Dict[str, Any]:
    """Evaluate explanations.

    Args:
        dataset_ref (XDataset): Reference dataset.
        dataset_hyp (XDataset): Hypothesis dataset.
        lang (str, optional): Language. Defaults to "zho".
        merge_distance (int, optional): Merge distance of CLEME. Defaults to 1.
        output_visualize (str, optional): Output filepath of chunk partition visualization.
            Defaults to None.
        output_evaluation (str, optional): Output filepath of correction evaluation results.
            Defaults to None.

    Returns:
        Dict[str, Any]: _description_
    """
    gec_dataset_hyp = convert_dataset(dataset=dataset_hyp)
    gec_dataset_ref = convert_dataset(dataset=dataset_ref)

    metric = DependentCLEME(
        lang=lang,
        scorer_type=ScorerType.PRF,
        weigher_type=WeigherType.LENGTH,
        output_visualize=output_visualize,
        merge_distance=merge_distance,
    )
    
    scorer_results, metric_results = metric.evaluate(
        dataset_hyp=gec_dataset_hyp,
        dataset_ref=gec_dataset_ref,
        persist_path=output_evaluation,
    )
    return scorer_results


def match_edits(
    sample_hyp: XSample, sample_ref: XSample
) -> List[BaseExplanationMetricResult]:
    """Match edits of sample_hyp and sample_ref.

    For each hyp sample, find the most possible matching ref sample.

    Args:
        sample_hyp (XSample): Reference sample.
        sample_ref (XSample): Hypothesis sample.

    Returns:
        List[BaseExplanationMetricResult]: _description_
    """
    results = []
    for edit_hyp in sample_hyp.edits:
        src_pos_hyp = set(range(edit_hyp.src_interval[0], edit_hyp.src_interval[1]+1))
        best_edit_ref, max_overlap = None, 0
        for edit_ref in sample_ref.edits:
            src_pos_ref = set(range(edit_ref.src_interval[0], edit_ref.src_interval[1]+1))
            curr_overlap = len(src_pos_hyp & src_pos_ref)
            if curr_overlap > max_overlap:
                best_edit_ref = edit_ref
                max_overlap = curr_overlap

        result = BaseExplanationMetricResult(edit_hyp=edit_hyp, edit_ref=best_edit_ref)
        results.append(result)
        LOGGER.debug(f"Match Edit: {result}")

    return results


def evaluate_exp(
    dataset_hyp: XDataset, dataset_ref: XDataset, verbose: bool = False
) -> Dict[str, Any]:
    """Evaluate performance of evaluation.

    To evaluate explanations, we must match edits of hyp and ref.

    Args:
        dataset_ref (XDataset): Reference dataset.
        dataset_hyp (XDataset): Hypothesis dataset.

    Returns:
        Dict[str, Any]: Evaluation results.
    """

    num_pred, num_true, hit = 0, 0, 0
    # Match edits
    sample_results = []
    for sample_hyp, sample_ref in zip(dataset_hyp, dataset_ref):
        edit_results = match_edits(sample_hyp=sample_hyp, sample_ref=sample_ref)
        sample_result = SampleExplanationMetricResult(bases=edit_results)
        sample_results.append(sample_result)

        num_pred += len(sample_hyp.edits)
        num_true += len(sample_ref.edits)
        hit += len(list(filter(lambda x: x.edit_ref, edit_results)))

    # Evaluate different parts of explanations
    eval_error_type = evaluate_exp_error_type(sample_results, verbose=verbose)
    eval_error_severity = evaluate_exp_error_severity(sample_results)
    eval_error_descrption = evaluate_exp_error_description(
        sample_results, verbose=verbose
    )
    
    """Visulize results as a table."""
    error_type_data = {}
    for key, value in eval_error_type.items():
        error_type_data[key] = value

    error_severity_data = {}
    for key, value in eval_error_severity.items():
        error_severity_data[key] = value

    error_description_data = {}
    for key, value in eval_error_descrption.items():
        error_description_data[key] = value

    print("Error Type:")
    print(tabulate(error_type_data.items(), headers=["Error Type", "Value"], tablefmt="grid"))
    print()

    # 打印 error_severity 的表格
    print("Error Severity:")
    print(tabulate(error_severity_data.items(), headers=["Error Severity", "Value"], tablefmt="grid"))
    print()

    # 打印 error_description 的表格
    print("Error Description:")
    print(tabulate(error_description_data.items(), headers=["Error Description", "Value"], tablefmt="grid"))
    
    return {
        "num_pred": num_pred,
        "num_true": num_true,
        "hit": hit,
        "hit_ratio": round(hit / num_pred, 4),
        "error_type": eval_error_type,
        "error_severity": eval_error_severity,
        "error_description": eval_error_descrption,
    }


def evaluate_exp_error_type(
    # sample_hyp: XSample,
    # sample_ref: XSample,
    sample_results: List[SampleExplanationMetricResult],
    verbose: bool = True,
) -> Dict[str, Any]:
    """Evaluate performance of error classification.

    各类别Precision: [0.996, 0.9797, 0.9424, 0.9236, 0.9348, 0.9446, 0.8964, 0.9473, 0.9111, 0.9106]
    各类别Recall: [0.992, 0.965, 0.835, 0.907, 0.874, 0.971, 0.926, 0.97, 0.963, 0.978]
    各类别F1: [0.994, 0.9723, 0.8855, 0.9152, 0.9034, 0.9576, 0.911, 0.9585, 0.9363, 0.9431]
    整体微平均Precision: 0.9381
    整体微平均Recall: 0.9381
    整体微平均F1: 0.9381

    Args:
        sample_hyp (XSample): Reference sample.
        sample_ref (XSample): Hypothesis sample.
        verbose (bool, optional): Whether to print details. Defaults to False.

    Returns:
        Dict[str, Any]: Evaluation results for error type.
    """

    y_true, y_pred = [], []
    for sample_result in sample_results:
        for edit_result in sample_result.bases:
            if edit_result.edit_ref is not None:
                y_true.append(edit_result.edit_ref.error_type)
                y_pred.append(edit_result.edit_hyp.error_type)

    acc = accuracy_score(y_true=y_true, y_pred=y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true=y_true, y_pred=y_pred, average=None, labels=VALID_ERROR_TYPES
    )
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        y_true=y_true, y_pred=y_pred, average="micro", labels=VALID_ERROR_TYPES
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true=y_true, y_pred=y_pred, average="macro", labels=VALID_ERROR_TYPES
    )
    if verbose:
        print("整体微平均Precision:", float("{:.4f}".format(micro_precision)))
        print("整体微平均Recall:", float("{:.4f}".format(micro_recall)))
        print("整体微平均F1:", float("{:.4f}".format(micro_f1)))
        print("整体宏平均Precision:", float("{:.4f}".format(macro_precision)))
        print("整体宏平均Recall:", float("{:.4f}".format(macro_recall)))
        print("整体宏平均F1:", float("{:.4f}".format(macro_f1)))
        print("各类别PRF")
        for idx, label in enumerate(VALID_ERROR_TYPES):
            print(
                f"{label} P / R / F1:",
                "{:.4f} {:.4f} {:.4f}".format(precision[idx], recall[idx], f1[idx]),
            )

    return {
        "acc": acc,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "labels_precision": precision.tolist(),
        "labels_recall": recall.tolist(),
        "labels_f1": f1.tolist(),
    }


def evaluate_exp_error_severity(
    sample_results: List[SampleExplanationMetricResult],
) -> Dict[str, Any]:
    y_true, y_pred = [], []
    for sample_result in sample_results:
        for edit_result in sample_result.bases:
            if edit_result.edit_ref is not None:
                y_true.append(edit_result.edit_ref.error_severity)
                y_pred.append(edit_result.edit_hyp.error_severity)

    mae = mean_absolute_error(y_true=y_true, y_pred=y_pred)
    return {"mae": mae}


def evaluate_exp_error_description(
    sample_results: List[SampleExplanationMetricResult], verbose: bool = True
) -> Dict[str, Any]:
    y_true, y_pred = [], []
    for sample_result in sample_results:
        for edit_result in sample_result.bases:
            if edit_result.edit_ref is not None:
                y_true.append(edit_result.edit_ref.error_description)
                y_pred.append(edit_result.edit_hyp.error_description)

    rouge = Rouge()
    bleu_reults, meteor_results = [], []
    rouge1_results, rouge2_results, rouge_long_results = [], [], []
    for hyp, ref in zip(y_pred, y_true):
        hyp_tokens = tokenize(hyp)
        ref_tokens = tokenize(ref)

        # BLEU
        bleu = sentence_bleu(references=[ref_tokens], hypothesis=hyp_tokens)
        bleu_reults.append(bleu)

        # METEOR
        meteor = meteor_score(references=[ref_tokens], hypothesis=hyp_tokens)
        meteor_results.append(meteor)

        # ROUGE
        rouge_tmp = rouge.get_scores(
            hyps=[" ".join(hyp_tokens)], refs=[" ".join(ref_tokens)]
        )
        rouge1_results.append(rouge_tmp[0]["rouge-1"]["f"])
        rouge2_results.append(rouge_tmp[0]["rouge-2"]["f"])
        rouge_long_results.append(rouge_tmp[0]["rouge-l"]["f"])

    return {
        "bleu": round(np.average(bleu_reults), 4),
        "meteor": round(np.average(meteor_results), 4),
        "rouge-1": round(np.average(rouge1_results), 4),
        "rouge-2": round(np.average(rouge2_results), 4),
        "rouge-L": round(np.average(rouge_long_results), 4),
    }


def tokenize(content: str) -> List[str]:
    return [x for x in remove_space(content.strip())]
