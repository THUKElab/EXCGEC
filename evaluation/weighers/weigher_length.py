from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from data import Edit

from ..schema import BaseEditMetricResult, SampleMetricResult
from .weigher_base import BaseWeigher

# @dataclass
# class SigmoidWeigher(BaseWeigher):
#     def signature(self) -> str:
#         return "sigmoid"

#     alpha: float = field(default=3.0, metadata={"help": "Scale factor"})
#     bias: float = field(
#         default=0.0, metadata={"help": "Bias factor (default not used)"}
#     )
#     min_value: float = field(
#         default=1.0, metadata={"help": "Clamp factor of minimal value"}
#     )
#     max_value: float = field(
#         default=1.0, metadata={"help": "Clamp factor of maximal value"}
#     )
#     reverse: bool = field(default=False, metadata={"help": "Reverse the value or not"})

#     def __call__(self, edit_len: int) -> float:
#         if self.reverse:
#             weight = self.alpha * (
#                 1 / (1 + (self.alpha - 1) * np.exp(edit_len - self.bias))
#             )
#         else:
#             weight = self.alpha * (
#                 1 / (1 + (self.alpha - 1) * np.exp(-edit_len + self.bias))
#             )
#         return np.clip(weight, self.min_value, self.max_value).item()

#     def __str__(self):
#         return (
#             f"SigmoidWeigher(alpha={self.alpha}, bias={self.bias}, "
#             f"min_value={self.min_value}, max_value={self.max_value}, reverse={self.reverse})"
#         )


class LengthWeigher(BaseWeigher, BaseModel):
    """Weigh edits by length."""

    tp_alpha: float = Field(default=2.0, description="Scale factor of tp_edits")
    tp_bias: float = Field(default=0.0, description="Bias factor of tp_edits")
    tp_min_value: float = Field(default=1.0, description="Minimum weight of tp_edits")
    tp_max_value: float = Field(default=1.0, description="Maximum weight of tp_edits")

    fp_alpha: float = Field(default=2.0, description="Scale factor of fp_edits")
    fp_bias: float = Field(default=0.0, description="Bias factor of fp_edits")
    fp_min_value: float = Field(default=1.0, description="Minimum weight of fp_edits")
    fp_max_value: float = Field(default=1.0, description="Maximum weight of fp_edits")

    fn_alpha: float = Field(default=2.0, description="Scale factor of fn_edits")
    fn_bias: float = Field(default=0.0, description="Bias factor of fn_edits")
    fn_min_value: float = Field(default=1.0, description="Minimum weight of fn_edits")
    fn_max_value: float = Field(default=1.0, description="Maximum weight of fn_edits")

    tn_alpha: float = Field(default=2.0, description="Scale factor of tn_edits")
    tn_bias: float = Field(default=0.0, description="Bias factor of tn_edits")
    tn_min_value: float = Field(default=1.0, description="Minimum weight of tn_edits")
    tn_max_value: float = Field(default=1.0, description="Maximum weight of tn_edits")

    @staticmethod
    def weigh_edit(
        edit: Edit,
        alpha: float,
        bias: float,
        min_value: float,
        max_value: float,
        reverse: bool = False,
    ) -> float:
        edit_len = max(len(edit.src_tokens), len(edit.tgt_tokens))
        if reverse:
            weight = alpha * (1 / (1 + (alpha - 1) * np.exp(edit_len - bias)))
        else:
            weight = alpha * (1 / (1 + (alpha - 1) * np.exp(-edit_len + bias)))
        return np.clip(weight, min_value, max_value).item()

    def __call__(self, metric_result: SampleMetricResult, **kwargs: Any) -> None:
        """Generate weights for MetricSampleResult."""
        for ref_result in metric_result.ref_results:
            assert isinstance(ref_result, BaseEditMetricResult)

            for tp_edit in ref_result.tp_edits:
                tp_edit.weight = self.weigh_edit(
                    edit=tp_edit,
                    alpha=self.tp_alpha,
                    bias=self.tp_bias,
                    min_value=self.tp_min_value,
                    max_value=self.tp_max_value,
                    reverse=False,
                )
            for fp_edit in ref_result.fp_edits:
                fp_edit.weight = self.weigh_edit(
                    edit=fp_edit,
                    alpha=self.fp_alpha,
                    bias=self.fp_bias,
                    min_value=self.fp_min_value,
                    max_value=self.fp_max_value,
                    reverse=True,
                )
            for fn_edit in ref_result.fn_edits:
                fn_edit.weight = self.weigh_edit(
                    edit=fn_edit,
                    alpha=self.fn_alpha,
                    bias=self.fn_bias,
                    min_value=self.fn_min_value,
                    max_value=self.fn_max_value,
                    reverse=False,
                )
            for tn_edit in ref_result.tn_edits:
                tn_edit.weight = self.weigh_edit(
                    edit=tn_edit,
                    alpha=self.tn_alpha,
                    bias=self.tn_bias,
                    min_value=self.tn_min_value,
                    max_value=self.tn_max_value,
                    reverse=False,
                )
