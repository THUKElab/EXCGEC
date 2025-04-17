import sys

from utils.string_utils import split_sentence

DEFAULT_SENTENCE_LENGTH = 64


def main():
    input_sentences = sys.argv[1]
    output_sentences = sys.argv[2]
    output_ids = sys.argv[3]

    with open(input_sentences, "r", encoding="utf-8") as f_in:
        with open(output_sentences, "w", encoding="utf-8") as f_out:
            with open(output_ids, "w", encoding="utf-8") as f_out_id:
                for idx, line in enumerate(f_in):
                    line = line.rstrip("\n")
                    if len(line) < DEFAULT_SENTENCE_LENGTH:
                        f_out.write(line + "\n")
                        f_out_id.write(str(idx) + "\n")
                        continue
                    sents = split_sentence(line, flag="zh")
                    for sent in sents:
                        f_out.write(sent + "\n")
                        f_out_id.write(str(idx) + "\n")


if __name__ == "__main__":
    main()
