import hashlib
from enum import Enum, unique
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

from datasets import concatenate_datasets, interleave_datasets

from ..extras.logging import get_logger


if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from transformers import Seq2SeqTrainingArguments

    from llmtuner.hparams import DataArguments


logger = get_logger(__name__)


@unique
class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"
    OBSERVATION = "observation"


def checksum(data_files: List[str], file_sha1: Optional[str] = None) -> None:
    if file_sha1 is None:
        logger.warning(
            "Checksum failed: missing SHA-1 hash value in dataset_info.json."
        )
        return

    if len(data_files) != 1:
        logger.warning("Checksum failed: too many files.")
        return

    with open(data_files[0], "rb") as f:
        sha1 = hashlib.sha1(f.read()).hexdigest()
        if sha1 != file_sha1:
            logger.warning(
                "Checksum failed: mismatched SHA-1 hash value at {}.".format(
                    data_files[0]
                )
            )


def infer_max_len(
    source_len: int, target_len: int, max_len: int, reserved_label_len: int
) -> Tuple[int, int]:
    max_target_len = int(max_len * (target_len / (source_len + target_len)))
    max_target_len = max(max_target_len, reserved_label_len)
    max_source_len = max_len - min(max_target_len, target_len)
    return max_source_len, max_target_len


def merge_dataset(
    all_datasets: List[Union["Dataset", "IterableDataset"]],
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
) -> Union["Dataset", "IterableDataset"]:
    if len(all_datasets) == 1:
        return all_datasets[0]
    elif data_args.mix_strategy == "concat":
        if data_args.streaming:
            logger.warning(
                "The samples between different datasets will not be mixed in streaming mode."
            )
        return concatenate_datasets(all_datasets)
    elif data_args.mix_strategy.startswith("interleave"):
        if not data_args.streaming:
            logger.warning(
                "We recommend using `mix_strategy=concat` in non-streaming mode."
            )
        return interleave_datasets(
            datasets=all_datasets,
            probabilities=data_args.interleave_probs,
            seed=training_args.seed,
            stopping_strategy=(
                "first_exhausted"
                if data_args.mix_strategy.endswith("under")
                else "all_exhausted"
            ),
        )
    else:
        raise ValueError("Unknown mixing strategy.")


def split_dataset(
    dataset: Union["Dataset", "IterableDataset"],
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
) -> Dict[str, "Dataset"]:
    if training_args.do_train:
        if data_args.val_size > 1e-6:  # Split the dataset
            if data_args.streaming:
                dataset = dataset.shuffle(
                    buffer_size=data_args.buffer_size, seed=training_args.seed
                )
                val_set = dataset.take(int(data_args.val_size))
                train_set = dataset.skip(int(data_args.val_size))
                return {"train_dataset": train_set, "eval_dataset": val_set}
            else:
                val_size = (
                    int(data_args.val_size)
                    if data_args.val_size > 1
                    else data_args.val_size
                )
                dataset = dataset.train_test_split(
                    test_size=val_size, seed=training_args.seed, shuffle=False
                )

                return {
                    "train_dataset": dataset["train"],
                    "eval_dataset": dataset["test"],
                }
        else:
            if data_args.streaming:
                dataset = dataset.shuffle(
                    buffer_size=data_args.buffer_size, seed=training_args.seed
                )

            return {"train_dataset": dataset}
    else:  # do_eval or do_predict
        return {"eval_dataset": dataset}
