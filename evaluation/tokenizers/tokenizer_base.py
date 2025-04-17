from abc import ABC, abstractmethod
from typing import Any, List, Sequence


class BaseTokenizer(ABC):
    """A base dummy tokenizer to derive from."""

    def signature(self) -> str:
        """Returns a signature for the tokenizer."""
        raise NotImplementedError()

    @abstractmethod
    def __call__(self, line: str, to_list: bool = False) -> Sequence[Any]:
        """Tokenize an input line with the tokenizer.

        Typically, self.detokenize(self.__call__(line)) == line

        Args:
            line (str): Text to tokenize.
            to_list (str): Whether to return list.

        Returns:
            Sequence[Any]: Tokenized tokens.
        """
        raise NotImplementedError()

    @abstractmethod
    def segment(self, line: str) -> List[str]:
        """Segment an input line into a list of string.

        Args:
            line (str): Text to segment.

        Returns:
            List[str]: Segmented tokens.
        """
        raise NotImplementedError()

    @abstractmethod
    def detokenize(self, tokens: Sequence[Any]) -> str:
        """Detokenize into a string.

        Args:
            tokens (Sequence[Any]): Tokenized input.

        Returns:
            str: Detokenized string.
        """
        raise NotImplementedError()

    # @abstractmethod
    # def tokenize(self, line: str) -> List[str]:
    #     raise NotImplementedError()

    # @abstractmethod
    # def convert_tokens(self, tokens: Any) -> List[str]:
    #     raise NotImplementedError

    def destroy(self):
        pass
