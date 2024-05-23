import json
from collections import Counter
from typing import Any, Iterator, List

from pydantic import BaseModel, Field

from data import Dataset, Sample
from utils import get_logger

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
            f"{self.src_interval}: {self.src_content} -> {self.tgt_interval}: {self.tgt_content}, "
            f"error_type={self.error_type}, "
            f"error_severity={self.error_severity}, "
            f"error_description={self.error_description}"
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
            for edit_json in sample_json["output"]["edits"]:
                edits.append(
                    XEdit(
                        # src_interval=edit_json["src_interval"],
                        # tgt_interval=edit_json["tgt_interval"],
                        # src_tokens=[x for x in edit_json["src_tokens"]],
                        # tgt_tokens=[x for x in edit_json["tgt_tokens"]],
                        src_content=edit_json["src_tokens"],
                        tgt_content=edit_json["tgt_tokens"],
                        error_type=edit_json["error_type"],
                        error_severity=edit_json["error_severity"],
                        error_description=edit_json["error_description"],
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

    # @classmethod
    # def parse_file_old(cls, filepath: str) -> "XDataset":
    #     with open(filepath, "r", encoding="utf-8") as f:
    #         data_json = json.load(f)

    #     samples = []
    #     for sample_json in data_json:
    #         edits = []
    #         for edit_json in sample_json["edits"]:
    #             llm_explanation = edit_json["llm_explanation"]
    #             error_type = llm_explanation["error_type"]

    #             if error_type in ERROR_TYPE_ALIAS.keys():
    #                 LOGGER.warning(
    #                     f"错误类型别名：{error_type} -> {ERROR_TYPE_ALIAS[error_type]}"
    #                 )
    #                 error_type = ERROR_TYPE_ALIAS[error_type]

    #             appraise_content = (
    #                 "请给出正确的错误类型、错误程度，并判断纠正解释是否正确。"
    #                 # "如果纠正解释不正确，则无需给出正确的错误类型、错误程度。"
    #             )
    #             if error_type not in VALID_ERROR_TYPES or error_type == "其他错误":
    #                 appraise_content = (
    #                     "未登录错误类型！！！请给出正确的错误类型、错误程度，并判断纠正解释是否正确。"
    #                     # "如果纠正解释不正确，则无需给出正确的错误类型、错误程度。"
    #                 )
    #                 LOGGER.warning(f"未登录错误类型：{error_type}")
    #             edits.append(
    #                 XEdit(
    #                     src_interval=edit_json["src_interval"],
    #                     tgt_interval=edit_json["tgt_interval"],
    #                     # src_tokens=[x for x in edit_json["src_tokens"]],
    #                     # tgt_tokens=[x for x in edit_json["tgt_tokens"]],
    #                     src_tokens=edit_json["src_tokens"],
    #                     tgt_tokens=edit_json["tgt_tokens"],
    #                     error_type=error_type,
    #                     error_severity=llm_explanation["severity"],
    #                     error_description=llm_explanation["description"],
    #                     appraise=XEditAppraise(content=appraise_content),
    #                 )
    #             )
    #         samples.append(
    #             XSample(
    #                 index=sample_json["id"],
    #                 domain=sample_json["domain"],
    #                 source=sample_json["source"],
    #                 target=sample_json["target"],
    #                 edits=edits,
    #             )
    #         )
    #     dataset = cls(samples=samples)
    #     dataset.metadata = dataset.get_metadata(version="20240319")
    #     return dataset


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
