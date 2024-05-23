"""
`Scorer` is an abstract class that enforces the implementation of a set
of abstract methods. This way, a correctly implemented metric will work
seamlessly with the rest of the codebase.

Scorer                                        # Abstract Scorer Class
  ├── SystemScorer                            # Corpus-level Scorer
  ├── SentenceScorer                          # Sentence-level Scorer
  ├── SentenceScorerForGLEU                   # Sentence-level Scorer for GLEU
  └── SentenceScorerForGLEU                   # Sentence-level Scorer for GLEU
"""

import sys
from abc import ABC, abstractmethod
from typing import Any, TextIO

from pydantic import BaseModel, Field

from ..schema import BaseScorerResult, OverallScorerResult


class BaseScorer(ABC, BaseModel):
    table_print: bool = Field(default=True)

    def __call__(self, **kwargs: Any) -> OverallScorerResult:
        """Score evaluation results with the scorer."""
        return self.score(**kwargs)

    @abstractmethod
    def score(self, **kwargs: Any) -> OverallScorerResult:
        raise NotImplementedError

    @abstractmethod
    def score_corpus(self, **kwargs: Any) -> BaseScorerResult:
        raise NotImplementedError

    @abstractmethod
    def score_sentence(self, **kwargs: Any) -> BaseScorerResult:
        raise NotImplementedError

    @abstractmethod
    def print_result_table(
        self, result: OverallScorerResult, sout: TextIO = sys.stdout, **kwargs: Any
    ) -> None:
        """Visulize results as a table."""
        raise NotImplementedError
