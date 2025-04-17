import json
import argparse

VALID_ERROR_TYPES = [
    # 标点级别错误
    "标点冗余",
    "标点丢失",
    "标点误用",
    # 拼写级别错误
    "字音混淆错误",
    "字形混淆错误",
    "词内部字符异位错误",
    "命名实体拼写错误",
    # 词语级别错误
    "词语冗余",
    "词语丢失",
    "词语误用",
    # 句法级别错误
    "词序不当",
    "逻辑不通",
    "句式杂糅",
    # 其他特殊错误
    "照应错误",
    "歧义错误",
    "语气不协调",
    "其他错误",
]


def process_json_file(input_file, output_file):
    # 读取文件内容
    with open(input_file, "r", encoding="utf-8") as file:
        content = file.read()

    # 去掉 <TGT> 标记
    content = content.replace("<TGT>", "")
    updated_content = (
        content.replace("<|im_end|>", "")
        .replace("<|endoftext|>", "")
        .replace("<|begin_of_text|>", "")
        .replace("<|eot_id|>", "")
    )
    updated_content = (
        updated_content.replace("<｜begin▁of▁sentence｜>", "")
        .replace("<｜end▁of▁sentence｜>", "")
        .replace("<|im_start|>", "")
        .replace("[gMASK] sop ", "")
    )
    updated_content = updated_content.replace("<s>", "").replace("</s>", "")
    # 输出去掉 <TGT> 标记后的内容

    # 将去掉 <TGT> 标记后的内容写回文件（如果需要）
    with open(input_file, "w", encoding="utf-8") as file:
        file.write(updated_content)

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    id = 0
    output_list = []
    for sample in data:
        try:
            # 尝试解析字符串为 JSON
            json_object = json.loads(sample["output"])

            # 处理 explanations 中的每个元素
            if "explanations" in json_object:
                for explanation in json_object["explanations"]:
                    # 检查 error_type 是否是合法的
                    if "error_type" in explanation:
                        if explanation["error_type"] not in VALID_ERROR_TYPES:
                            explanation["error_type"] = "其他错误"

                    # 检查 error_severity 是否是合法的
                    if "error_severity" in explanation:
                        if not isinstance(
                            explanation["error_severity"], int
                        ) or explanation["error_severity"] not in [1, 2, 3, 4, 5]:
                            explanation["error_severity"] = 1

            sample["output"] = json_object
            output_list.append(sample)
        except json.JSONDecodeError as e:
            json_object = {
                "target": sample["input"],
                "edits": [],
                "explanations": [],
            }
            sample["output"] = json_object
            output_list.append(sample)
            id += 1

    print("empty: ", id, "all: ", len(output_list))
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_list, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a JSON file.")

    parser.add_argument(
        "--input_file", type=str, required=True, help="Path to the input JSON file."
    )
    parser.add_argument(
        "--output_file", type=str, required=True, help="Path to the output JSON file."
    )

    args = parser.parse_args()

    process_json_file(args.input_file, args.output_file)
