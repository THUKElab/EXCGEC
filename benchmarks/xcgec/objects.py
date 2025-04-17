import json
import re
from collections import Counter
from typing import Any, Iterator, List

from pydantic import BaseModel, Field

from data import Dataset, Sample
from utils import get_logger
from tqdm import tqdm

LOGGER = get_logger(name=__name__)

VALID_ERROR_TYPES = [
    # 标点级别错误
    "标点冗余",
    "标点丢失",
    "标点误用",
    # 拼写级别错误
    "字音混淆错误",
    "字形混淆错误",
    "词内部字符异位错误",
    "命名实体拼写错误",
    # 词语级别错误
    "词语冗余",
    "词语丢失",
    "词语误用",
    # 句法级别错误
    "词序不当",
    "逻辑不通",
    "句式杂糅",
    # 其他特殊错误
    "照应错误",
    "歧义错误",
    "语气不协调",
    "其他错误",
]

ERROR_TYPES_INDEX = {x: i for i, x in enumerate(VALID_ERROR_TYPES)}


class XEditAppraise(BaseModel):
    is_consistent: bool = Field(default=True)
    is_correct_error_type: bool = Field(default=True)
    correct_error_severity: int = Field(default=None)
    is_correct_error_description: bool = Field(default=True)


class XEdit(BaseModel):
    # tgt_index: int = Field(default=0, description="Belonging target index")
    src_interval: List[int] = Field(default=None, metadata="Source interval")
    tgt_interval: List[int] = Field(default=None, description="Target interval")
    # src_tokens: str = Field(default=None, description="Source tokens")
    # tgt_tokens: str = Field(default=None, description="Target tokens")
    src_content: str = Field(default=None, description="Source content")
    tgt_content: str = Field(default=None, description="Target content")
    src_tokens: List[str] = Field(default=None, description="Source tokens")
    tgt_tokens: List[str] = Field(default=None, description="Target tokens")
    error_type: str = Field(default=None, description="Error type")
    error_severity: int = Field(default=None, description="Error severity")
    error_description: str = Field(default=None, description="Explanation")
    # appraise: XEditAppraise = Field(default=None, description="Appraise to explanation")

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if self.src_tokens is None:
            self.src_tokens = [x for x in self.src_content]
        if self.tgt_tokens is None:
            self.tgt_tokens = [x for x in self.tgt_content]

    def __repr__(self) -> str:
        # src_tokens = " ".join(self.src_tokens)
        # tgt_tokens = " ".join(self.tgt_tokens)
        return (
            f"{self.src_interval}: {self.src_tokens} -> {self.tgt_interval}: {self.tgt_tokens}, "
            # f"error_type={self.error_type}, "
            # f"error_severity={self.error_severity}, "
            # f"error_description={self.error_description}"
            # f"appraise={self.appraise}"
        )

    def __str__(self) -> str:
        return self.__repr__()


class XSample(BaseModel):
    index: int = Field(default=None, description="Sample index")
    domain: str = Field(default=None, description="Data source domain")
    source: str = Field(
        default=None, description="Source sentences, which are usually ungrammatical"
    )
    target: str = Field(
        default=None, description="Target sentences, which are always grammatical"
    )
    edits: List[XEdit] = Field(
        default=None, description="Edits extracted from source to target"
    )


class XDatasetMetaData(BaseModel):
    """Metadata of CSLDataset."""

    number: int = Field(default=None)
    version: str = Field(default=None)
    type_counter: Counter = Field(default_factory=Counter)
    severity_counter: Counter = Field(default_factory=Counter)


class XDataset(BaseModel):
    metadata: XDatasetMetaData = Field(default=None)
    samples: List[XSample] = Field(default_factory=list, description="Samples included")

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[XSample]:
        return iter(self.samples)

    def __getitem__(self, item: int) -> XSample:
        return self.samples[item]

    def append(self, sample: XSample) -> None:
        self.samples.append(sample)

    def get_metadata(self, version: str = None) -> XDatasetMetaData:
        # types = [e.error_type[0] for x in self.samples for e in x.edits]
        types = [e.error_type for x in self.samples for e in x.edits]
        severities = [e.error_severity for x in self.samples for e in x.edits]
        type_counter = Counter(types)
        severity_counter = Counter(severities)

        # Rearrange by order
        new_type_counter = Counter()
        for error_type in VALID_ERROR_TYPES:
            new_type_counter[error_type] = type_counter[error_type]

        new_severity_counter = Counter()
        for i in range(1, 6):
            new_severity_counter[i] = severity_counter[i]

        return XDatasetMetaData(
            number=len(self.samples),
            version=version,
            type_counter=new_type_counter,
            severity_counter=new_severity_counter,
        )

    @classmethod
    def parse_file_v1(cls, filepath: str) -> "XDataset":
        # NOTE: 自己写兼容旧文件的代码！
        with open(filepath, "r", encoding="utf-8") as f:
            data_json = json.load(f)

        dataset = []
        for sample_json in data_json:

            edits = []
            for edit, explanation in zip(
                sample_json["output"]["edits"], sample_json["output"]["explanations"]
            ):
                edits.append(
                    XEdit(
                        src_interval=edit.get("src_interval", []),
                        tgt_interval=edit.get("tgt_interval", []),
                        src_tokens=[x for x in edit.get("src_tokens", [])],
                        tgt_tokens=[x for x in edit.get("tgt_tokens", [])],
                        error_type=explanation.get("error_type", "未知错误类型"),
                        error_severity=explanation.get(
                            "error_severity", "未知严重性级别"
                        ),
                        error_description=explanation.get(
                            "error_description", "无错误描述"
                        ),
                    )
                )
            sample = XSample(
                index=len(dataset),
                # domain=sample_json["domain"],
                source=sample_json["input"],
                target=sample_json["output"]["target"],
                edits=edits,
            )
            dataset.append(sample)
        return dataset

    @classmethod
    def parse_file_v2(cls, input_str: str) -> "XDataset":
        # 提取 "input" 后的内容
        user_match = re.search(r"user\n([^\n]+)", input_str)
        if user_match:
            source_sentence = user_match.group(1)
        else:
            user_match = re.search(
                r"<\|user\|>\s*\n*(.*?)\s*<\|assistant\|>", input_str
            )
            if user_match:
                source_sentence = user_match.group(1)
            else:
                user_match = re.search(r"user\n*([^\n]+)assistant", input_str)
                if user_match:
                    source_sentence = user_match.group(1)
                else:
                    user_match = re.search(r"User:(.*?)\n*Assistant:", input_str)
                    if user_match:
                        source_sentence = user_match.group(1)
                    else:
                        user_match = re.search(
                            r"将以下文本进行语法纠错并生成纠正后的句子以及纠正相关的解释信息\n*\s*(.*?)<\|assistant\|>",
                            input_str,
                        )
                        if user_match:
                            source_sentence = user_match.group(1)
                        else:
                            source_sentence = None

        # 提取 "target" 后的内容
        target_match = re.search(r'"target":\s*"([^"]+)"', input_str)
        if target_match:
            predicted_sentence = target_match.group(1)
        elif (
            source_sentence
            and "那天，我回到家跟妈妈说:“妈妈，我想上英语陪训班。”她对我说:“好！明天我带去吧。"
            in source_sentence
        ):
            predicted_sentence = "那天，我回到家跟妈妈说:“妈妈，我想上英语辅导班。”她对我说:“好！明天我带去吧。”"
        else:
            predicted_sentence = None

        if not predicted_sentence:
            predicted_sentence = source_sentence
        dataset = []
        edits = []
        sample = XSample(
            index=len(dataset),
            # domain=sample_json["domain"],
            source=source_sentence,
            target=predicted_sentence,
            edits=edits,
        )

        dataset.append(sample)
        return dataset

    @classmethod
    def parse_file_v3(cls, input_str: str) -> "XDataset":
        # 提取 "input" 后的内容
        user_match = re.search(r"user\n([^\n]+)", input_str)
        if user_match:
            source_sentence = user_match.group(1)
        else:
            source_sentence = None

        # 提取 "target" 后的内容
        target_match = re.search(r'"target":\s*"([^"]+)"', input_str)
        if target_match:
            predicted_sentence = target_match.group(1)
        else:
            predicted_sentence = None

        dataset = []
        edits = []
        sample = XSample(
            index=len(dataset),
            source=source_sentence,
            target=predicted_sentence,
            edits=edits,
        )

        LOGGER.info(f"sample: {sample}")
        dataset.append(sample)
        return dataset


def convert_dataset(dataset: XDataset, drop_edits: bool = True) -> Dataset:
    """Convert exaplainable dataset into conventional dataset.

    Args:
        datatset (XDataset): _description_

    Returns:
        Dataset: _description_
    """

    # NOTE: drop_edits
    if not drop_edits:
        raise NotImplementedError

    gec_dataset = Dataset()
    for exp_sample in dataset:
        gec_sample = Sample(
            index=exp_sample.index,
            source=[exp_sample.source],
            target=[exp_sample.target],
        )
        gec_dataset.append(gec_sample)
    return gec_dataset


def convert_dataset_2(dataset: XDataset, drop_edits: bool = True) -> Dataset:
    """Convert exaplainable dataset into conventional dataset.

    Args:
        datatset (XDataset): _description_

    Returns:
        Dataset: _description_
    """

    # NOTE: drop_edits
    if not drop_edits:
        raise NotImplementedError

    gec_dataset = Dataset()
    for exp_sample in dataset:
        gec_sample = Sample_2(
            index=exp_sample.index,
            source=[exp_sample.input],
            reference=[exp_sample.reference],
            target_1B=[exp_sample.target_1B],
            target_7B=[exp_sample.target_7B],
        )
        gec_dataset.append(gec_sample)
    return gec_dataset
