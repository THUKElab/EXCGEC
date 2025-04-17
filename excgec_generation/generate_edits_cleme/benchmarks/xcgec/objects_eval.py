from typing import List

from pydantic import BaseModel, Field

from .objects import XEdit


class BaseExplanationMetricResult(BaseModel):
    edit_hyp: XEdit = Field(default=None)
    edit_ref: XEdit = Field(default=None)
    error_description_score: float = Field(default=None)


class SampleExplanationMetricResult(BaseModel):
    bases: List[BaseExplanationMetricResult] = Field(default_factory=list)


class ExplanationScore(BaseModel):
    hit: int = Field(default=None)

    error_type_tp: int = Field(default=None)
    error_type_tp: int = Field(default=None)
    error_type_tp: int = Field(default=None)
    error_type_f1: float = Field(default=None)
    error_type_acc: float = Field(default=None)
    error_severity_mse: float = Field(default=None)
    error_description_score: float = Field(default=None)
