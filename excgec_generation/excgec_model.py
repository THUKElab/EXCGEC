from transformers import AutoModelForCausalLM
from .excgec_generation_mixin import ExcgecGenerationMixin


class ExcgecModel(ExcgecGenerationMixin, AutoModelForCausalLM):
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        model = super().from_pretrained(*args, **kwargs)

        model.generate = ExcgecGenerationMixin.generate.__get__(model, ExcgecModel)
        model.excgec_beam_search = ExcgecGenerationMixin.excgec_beam_search.__get__(
            model, ExcgecModel
        )
        model.excgec_sample = ExcgecGenerationMixin.excgec_sample.__get__(
            model, ExcgecModel
        )
        return model


