from enum import Enum
from typing import Any, Dict, Type

from utils import get_logger

from .weigher_base import BaseWeigher
from .weigher_length import LengthWeigher
from .weigher_llm import LLMWeigher
from .weigher_similarity import SimilarityWeigher

LOGGER = get_logger(__name__)

WEIGHERS = {
    "base": BaseWeigher,
    "sigmoid": LengthWeigher,
}


class WeigherType(str, Enum):
    NONE = "none"
    LENGTH = "length"
    SIMILARITY = "similarity"
    LLM = "llm"


WEIGHER_CLASS: Dict[WeigherType, Type[BaseWeigher]] = {
    WeigherType.NONE: BaseWeigher,
    WeigherType.LENGTH: LengthWeigher,
    WeigherType.SIMILARITY: SimilarityWeigher,
    WeigherType.LLM: LLMWeigher,
}


def get_weigher(weigher_type: WeigherType, **kwargs: Any) -> BaseWeigher:
    LOGGER.info(f"Build Weigher: {weigher_type}")
    return WEIGHER_CLASS[weigher_type](**kwargs)


__all__ = [
    BaseWeigher,
    LengthWeigher,
    LLMWeigher,
    SimilarityWeigher,
    WeigherType,
    WEIGHER_CLASS,
    get_weigher,
]
