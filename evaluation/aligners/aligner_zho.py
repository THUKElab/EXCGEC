import os
from typing import Tuple

from data import PUNCTUATION

from ..langs.zho.zho import GranularityType
from .aligner_base import BaseAligner

DEFAULT_DIR_RESOURCE = os.path.join(os.path.dirname(__file__), "../langs/zho")


class AlignerZho(BaseAligner):
    _open_pos = {}

    def signature(self) -> str:
        return "zho"

    def __init__(
        self,
        del_cost: float = 1.0,
        ins_cost: float = 1.0,
        standard: bool = False,
        brute_force: bool = False,
        verbose: bool = False,
        granularity: str = GranularityType.CHAR,
    ) -> None:
        super().__init__(
            standard=standard,
            del_cost=del_cost,
            ins_cost=ins_cost,
            brute_force=brute_force,
            verbose=verbose,
        )
        self.granularity = granularity

        # Load resource: Chinese confustion set
        self.confusion_dict = {}
        path_confustion = os.path.join(DEFAULT_DIR_RESOURCE, "confusion_dict.txt")
        with open(path_confustion, "r", encoding="utf-8") as f:
            for line in f:
                li = line.strip().split(" ")
                self.confusion_dict[li[0]] = li[1:]

        # Load resource: Chinese cilin
        self.semantic_dict = {}
        path_cilin = os.path.join(DEFAULT_DIR_RESOURCE, "cilin.txt")
        with open(path_cilin, "r", encoding="gbk") as f:
            for line in f:
                code, *words = line.strip().split(" ")
                for word in words:
                    self.semantic_dict[word] = code

    def get_sub_cost(self, src_token: Tuple, tgt_token: Tuple) -> float:
        """Cost of Linguistic Damerau-Levenshtein.

        Args:
            src_token (Tuple): A source Token (text, POS, pinyin)
            tgt_token (Tuple): A target Token (text, POS, pinyin)

        Returns:
            cost (float): A linguistic cost between 0 < x < 2
        """
        if src_token[0] == tgt_token[0]:
            return 0
        if self.granularity == GranularityType.WORD:
            # 词级别可以额外利用词性信息
            semantic_cost = self._get_semantic_cost(src_token[0], tgt_token[0]) / 6.0
            pos_cost = self._get_pos_cost(src_token[1], tgt_token[1])
            char_cost = self._get_char_cost(
                src_token[0], tgt_token[0], src_token[2], tgt_token[2]
            )
            return semantic_cost + pos_cost + char_cost
        else:
            # 字级别只能利用字义信息（从大词林中获取）和字面相似度信息
            semantic_cost = self._get_semantic_cost(src_token[0], tgt_token[0]) / 6.0
            if src_token[0] in PUNCTUATION and tgt_token[0] in PUNCTUATION:
                pos_cost = 0.0
            elif src_token[0] not in PUNCTUATION and tgt_token[0] not in PUNCTUATION:
                pos_cost = 0.25
            else:
                pos_cost = 0.499
            char_cost = self._get_char_cost(
                src_token[0], tgt_token[0], src_token[2], tgt_token[2]
            )
            return semantic_cost + char_cost + pos_cost

    def _get_semantic_class(self, word: str) -> Tuple:
        """Acquire the semantic class of `word`.

        Based on the paper: Improved-Edit-Distance Kernel for Chinese Relation Extraction
        """
        if word in self.semantic_dict:
            code = self.semantic_dict[word]
            high, mid, low = code[0], code[1], code[2:4]
            return high, mid, low
        return None

    def _get_semantic_cost(self, a: str, b: str) -> int:
        """计算基于语义信息的替换操作cost
        :param a: 单词a的语义类别
        :param b: 单词b的语义类别
        :return: 替换编辑代价
        """
        a_class = self._get_semantic_class(a)
        b_class = self._get_semantic_class(b)
        # unknown class, default to 1
        if a_class is None or b_class is None:
            return 4
        elif a_class == b_class:
            return 0
        else:
            return 2 * (3 - self._get_class_diff(a_class, b_class))

    @staticmethod
    def _get_class_diff(a_class: Tuple[str], b_class: Tuple[str]) -> int:
        """根据大词林的信息，计算两个词的语义类别的差距
        d == 3 for equivalent semantics
        d == 0 for completely different semantics
        """
        d = sum([a == b for a, b in zip(a_class, b_class)])
        return d

    def _get_pos_cost(self, a_pos: str, b_pos: str) -> float:
        """计算基于词性信息的编辑距离cost
        :param a_pos: 单词a的词性
        :param b_pos: 单词b的词性
        :return: 替换编辑代价
        """
        if a_pos == b_pos:
            return 0
        elif a_pos in self._open_pos and b_pos in self._open_pos:
            return 0.25
        else:
            return 0.499

    def _get_char_cost(self, a: str, b: str, pinyin_a, pinyin_b) -> float:
        """
        NOTE: This is a replacement of ERRANTS lemma cost for Chinese
        计算基于字符相似度的编辑距离cost
        """
        if not (check_all_chinese(a) and check_all_chinese(b)):
            return 0.5
        if len(a) > len(b):
            a, b = b, a
            pinyin_a, pinyin_b = pinyin_b, pinyin_a
        if a == b:
            return 0
        else:
            return self._get_spell_cost(a, b, pinyin_a, pinyin_b)

    def _get_spell_cost(self, a: str, b: str, pinyin_a, pinyin_b) -> float:
        """计算两个单词拼写相似度，分别由字形相似度和字音相似度组成
        :@param a: word
        :@param b: word，且单词a的长度小于等于b
        :@param pinyin_a: 单词a的拼音
        :@param pinyin_b: 单词b的拼音
        :@return: sub_cost
        """
        count = 0
        for i in range(len(a)):
            for j in range(len(b)):
                if (
                    a[i] == b[j]
                    or (set(pinyin_a) & set(pinyin_b))
                    or (
                        b[j] in self.confusion_dict.keys()
                        and a[i] in self.confusion_dict[b[j]]
                    )
                    or (
                        a[i] in self.confusion_dict.keys()
                        and b[j] in self.confusion_dict[a[i]]
                    )
                ):
                    count += 1
                    break
        return (len(a) - count) / (len(a) * 2)


def check_all_chinese(word):
    """判断一个单词是否全部由中文组成"""
    return all(["\u4e00" <= ch <= "\u9fff" for ch in word])
