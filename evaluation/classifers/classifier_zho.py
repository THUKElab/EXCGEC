from typing import List, Tuple

from pypinyin import Style, pinyin

from data import Edit

from .classifier_base import BaseClassifier

# file_path = os.path.dirname(os.path.abspath(__file__))
# char_smi = CharFuncs(os.path.join(file_path.replace("modules", ""), 'data/char_meta.txt'))
char_smi = None


class ClassifierZho(BaseClassifier):
    # 后缀词比例太少，暂且分入其它
    POS2TYPE = {
        "n": "NOUN",
        "nd": "NOUN",
        "nh": "NOUN-NE",
        "ni": "NOUN-NE",
        "nl": "NOUN-NE",
        "ns": "NOUN-NE",
        "nt": "NOUN-NE",
        "nz": "NOUN-NE",
        "v": "VERB",
        "a": "ADJ",
        "b": "ADJ",
        "c": "CONJ",
        "r": "PRON",
        "d": "ADV",
        "u": "AUX",
        "m": "NUM",
        "p": "PERP",
        "q": "QUAN",
        "wp": "PUNCT",
    }

    def signature(self) -> str:
        return "zho"

    def __init__(self, granularity: str = "char"):
        super().__init__()
        self.granularity = granularity
        if granularity == "word":
            raise NotImplementedError

    def __call__(self, source: List[Tuple], target: List[Tuple], edit: Edit) -> Edit:
        """为编辑操作划分错误类型
        @param source: 错误句子信息
        @param target: 正确句子信息
        @param edit: 编辑操作
        @return: 划分完错误类型后的编辑操作
        """
        error_type = edit.types[0]
        src_span = " ".join(edit.src_tokens)
        tgt_span = " ".join(edit.tgt_tokens)

        if error_type[0] == "T":
            edit.types = ["T"]
        elif error_type[0] == "D":
            if self.granularity == "word":
                if len(src_span) > 1:
                    # 词组冗余暂时分为 OTHER
                    edit.types = ["R:OTHER"]
                else:
                    pos = self.POS2TYPE.get(source[edit.src_interval[0]][1])
                    pos = "NOUN" if pos == "NOUN-NE" else pos
                    pos = "MC" if tgt_span == "[缺失成分]" else pos
                    edit.types = ["U:{:s}".format(pos)]
            else:
                edit.types = ["U"]
        elif error_type[0] == "I":
            if self.granularity == "word":
                if len(tgt_span) > 1:
                    # 词组丢失暂时分为 OTHER
                    edit.types = ["M:OTHER"]
                else:
                    pos = self.POS2TYPE.get(target[edit.tgt_interval[0]][1])
                    pos = "NOUN" if pos == "NOUN-NE" else pos
                    pos = "MC" if tgt_span == "[缺失成分]" else pos
                    edit.types = ["M:{:s}".format(pos)]
            else:
                edit.types = ["M"]
        elif error_type[0] == "S":
            if self.granularity == "word":
                if check_spell_error(
                    src_span.replace(" ", ""), tgt_span.replace(" ", "")
                ):
                    edit.types = ["S:SPELL"]
                    # Todo 暂且不单独区分命名实体拼写错误
                    # if edit[4] - edit[3] > 1:
                    #     cor = Correction("S:SPELL:COMMON", tgt_span, (edit[1], edit[2]))
                    # else:
                    #     pos = self.get_pos_type(tgt[edit[3]][1])
                    #     if pos == "NOUN-NE":  # 命名实体拼写有误
                    #         cor = Correction("S:SPELL:NE", tgt_span, (edit[1], edit[2]))
                    #     else:  # 普通词语拼写有误
                    #         cor = Correction("S:SPELL:COMMON", tgt_span, (edit[1], edit[2]))
                else:
                    if len(tgt_span) > 1:
                        # 词组被替换暂时分为 OTHER
                        edit.types = ["S:OTHER"]
                    else:
                        pos = self.POS2TYPE.get(target[edit.tgt_interval[0]][1])
                        pos = "NOUN" if pos == "NOUN-NE" else pos
                        pos = "MC" if tgt_span == "[缺失成分]" else pos
                        edit.types = ["S:{:s}".format(pos)]
            else:
                edit.types = ["S"]
        return edit


def check_spell_error(src_span: str, tgt_span: str, threshold: float = 0.8) -> bool:
    if len(src_span) != len(tgt_span):
        return False
    src_chars = [ch for ch in src_span]
    tgt_chars = [ch for ch in tgt_span]
    if sorted(src_chars) == sorted(tgt_chars):  # 词内部字符异位
        return True
    for src_char, tgt_char in zip(src_chars, tgt_chars):
        if src_char != tgt_char:
            if src_char not in char_smi.data or tgt_char not in char_smi.data:
                return False
            v_sim = char_smi.shape_similarity(src_char, tgt_char)
            p_sim = char_smi.pronunciation_similarity(src_char, tgt_char)
            if v_sim + p_sim < threshold and not (
                set(pinyin(src_char, style=Style.NORMAL, heteronym=True)[0])
                & set(pinyin(tgt_char, style=Style.NORMAL, heteronym=True)[0])
            ):
                return False
    return True
