import os
import sys
import json
import argparse
from transformers import AutoTokenizer
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from excgec_generation.excgec_model import ExcgecModel

MODEL_TO_TERMINATORS = {
    "llama3": "<|eot_id|>",
    "deepseek": "<｜end▁of▁sentence｜>",
    "qwen": "<|im_end|>",
}


def predict(input_file, output_file, model, tokenizer, model_dir, is_first):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    output_list = []
    for sample in tqdm(data, desc="Processing samples", unit="sample"):
        prompt = sample["input"]
        messages = [
            {
                "role": "system",
                "content": "将以下文本进行语法纠错并生成纠正后的句子以及纠正相关的解释信息",
            },
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=3072,
            Model_p=model_dir,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            max_length=2048,
            top_p=0.9,
        )
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]

        output_sample = {"input": prompt, "output": response}
        output_list.append(output_sample)

        if is_first:
            with open(output_file, "w", encoding="utf-8") as f1:
                json.dump([output_sample], f1, indent=4, ensure_ascii=False)
            is_first = False
        else:
            with open(output_file, "r", encoding="utf-8") as f2:
                context = json.load(f2)
                if isinstance(context, list):
                    context.append(output_sample)
                else:
                    raise ValueError(
                        "Error: The content of the JSON file is not a list."
                    )

            with open(output_file, "w", encoding="utf-8") as f3:
                json.dump(context, f3, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EXGEC model.")

    parser.add_argument(
        "--input_file", type=str, required=True, help="Path to the input JSON file."
    )
    parser.add_argument(
        "--output_file", type=str, required=True, help="Path to the output JSON file."
    )
    parser.add_argument(
        "--model_dir", type=str, required=True, help="Path to the model directory."
    )

    args = parser.parse_args()

    device = "cuda"
    is_first = True

    model = ExcgecModel.from_pretrained(
        args.model_dir, torch_dtype="auto", device_map="auto", trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

    model_name = None
    if "llama3" in os.path.basename(args.model_dir).lower():
        model_name = "llama3"
    elif "deepseek" in os.path.basename(args.model_dir).lower():
        model_name = "deepseek"
    elif "qwen" in os.path.basename(args.model_dir).lower():
        model_name = "qwen"

    if model_name:
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids(MODEL_TO_TERMINATORS[model_name]),
        ]
    else:
        raise ValueError(
            f"Unsupported model in {args.model_dir}. Unable to determine model name."
        )

    predict(args.input_file, args.output_file, model, tokenizer, args.model_dir, is_first)
