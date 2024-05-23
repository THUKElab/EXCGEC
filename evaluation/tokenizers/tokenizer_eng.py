from functools import lru_cache
from typing import Any, List

import spacy
from spacy.tokens import Doc

from .tokenizer_base import BaseTokenizer


class TokenizerEng(BaseTokenizer):
    DEFAULT_SPACY_MODEL = "en_core_web_sm"

    def __init__(
        self, seg_by_spacy: str = False, spacy_model: str = DEFAULT_SPACY_MODEL
    ) -> None:
        self.seg_by_spacy = seg_by_spacy
        self.spacy_model = spacy.load(spacy_model, disable=["ner"])

    def signature(self) -> str:
        return "eng"

    @lru_cache(maxsize=2**16)
    def __call__(self, line: str, to_list: bool = False) -> Any:
        if self.seg_by_spacy:
            # Segment the line using spacy
            text = self.spacy_model(line)
        else:
            # Segment the line by spaces
            text = Doc(self.spacy_model.vocab, self.segment(line))
            self.spacy_model.tagger(text)
            self.spacy_model.parser(text)

        if to_list:
            return [token.text for token in text]
        return text

    @staticmethod
    def segment(line: str) -> List[str]:
        return line.split()

    def detokenize(self, tokens: Doc) -> str:
        return tokens.text

    # def tokenize(self, text: str) -> List[Any]:
    #     doc = self.spacy_model(text.strip(), disable=["parser", "tagger", "ner"])
    #     tokens = [str(token) for token in doc]
    #     return tokens

    # def tokenize_batch(
    #     self, text_list: List[str], batch_size: int = 1024
    # ) -> List[List[Any]]:
    #     docs = self.spacy_model.pipe(
    #         text_list, batch_size=batch_size, disable=["parser", "tagger", "ner"]
    #     )
    #     docs = [[x.text for x in line] for line in docs]
    #     return docs

    def destroy(self):
        del self._tokenizer
        self._tokenizer = None
