from functools import lru_cache
from typing import Any, List

from ltp import LTP
from pypinyin import Style, lazy_pinyin, pinyin

from utils import get_logger, remove_space

from .tokenizer_base import BaseTokenizer

LOGGER = get_logger(__name__)


class TokenizerZho(BaseTokenizer):
    def __init__(self, granularity: str = "char", **kwargs: Any) -> None:
        self.granularity = granularity
        self._tokenizer = None
        if granularity == "word":
            model_name_or_path = kwargs.get(
                "pretrained_model_name_or_path", "LTP/small"
            )
            self._tokenizer = LTP(pretrained_model_name_or_path=model_name_or_path)
            self._tokenizer.add_words(words=["[缺失成分]"])
            LOGGER.info(
                f"{self.__class__.__name__} initialize LTP model: {model_name_or_path}"
            )

    def signature(self) -> str:
        return "zho"

    @lru_cache(maxsize=2**16)
    def __call__(self, line: str, add_extra_info=True, to_list: bool = False) -> Any:
        """The tokenization of Chinese text in this script contains two options:

        1) Tokenize at char-level
        2) Tokenize at word-level

        Args:
            line (str): A segment to tokenize.

        Raises:
            ValueError: Invalid granularity if self.granularity is unsupported.

        Returns:
            Any: Tokenized tokens.
        """
        if self.granularity == "char":
            tokens = self.tokenize_char(line, add_extra_info=add_extra_info)
        elif self.granularity == "word":
            tokens = self.tokenize_word(line, add_extra_info=add_extra_info)
        else:
            raise ValueError(f"Invalid granularity: {self.granularity}")

        if to_list:
            return [token[0] for token in tokens]
        return tokens

    @staticmethod
    def segment(line: str) -> List[str]:
        return [char for char in remove_space(line.strip())]

    @classmethod
    def tokenize_char(cls, line: str, add_extra_info=False) -> List[Any]:
        """Tokenize Chinese text by characters.

        Args:
            line (str): Input Chinese text.
            add_extra_info (bool, optional): Whether to acquire extra info. Defaults to False.

        Raises:
            RuntimeError: Input char is empty.

        Returns:
            List[Any]: Tokenized results.
        """
        line_tok = " ".join(cls.segment(line))
        # [缺失成分] is a single special token
        line_tok = line_tok.replace("[ 缺 失 成 分 ]", "[缺失成分]").split()
        if not add_extra_info:
            return line_tok

        result = []
        for char in line_tok:
            py = pinyin(char, style=Style.NORMAL, heteronym=True)
            if not len(py):  # Raise Error if char is empty
                raise RuntimeError(f"Unknown pinyin: `{char}` from {line_tok}")
            result.append((char, "unk", py[0]))
        return result

    def tokenize_word(self, line: str, add_extra_info: bool = False) -> List[Any]:
        """Tokenize Chinese text by words.

        Args:
            line (str): Input Chinese text.
            add_extra_info (bool, optional): Whether to acquire extra info. Defaults to False.

        Returns:
            List[Any]: Tokenized results.
        """
        line = remove_space(line)
        if not add_extra_info:
            (words,) = self._tokenizer.pipeline([line], tasks=["cws"]).to_tuple()
            return words[0]
        words, pos = self._tokenizer.pipeline([line], tasks=["cws", "pos"]).to_tuple()

        result = []
        for s, p in zip(words, pos):
            py = [lazy_pinyin(word) for word in s]
            result.append(list(zip(s, p, py)))
        return result

    def detokenize(self, tokens: List[Any]) -> str:
        return "".join([x[0] for x in tokens])
