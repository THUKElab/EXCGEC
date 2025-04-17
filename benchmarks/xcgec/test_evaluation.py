import sys
import os
import json
import re
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from benchmarks.xcgec.evaluate import evaluate, get_chunked_dataset
from benchmarks.xcgec.objects import XDataset, XEdit, XSample

def string_to_json(input_chunk):
    """
    将字符串转换成包含 src_interval, tgt_interval, src_tokens 和 tgt_tokens 字段的 JSON 字典
    
    Args:
        input_str (str): 包含字段和值的字符串
        
    Returns:
        dict: 转换后的 JSON 格式字典
    """
    # 构建 JSON 格式的字典
    data = {
        "src_interval": input_chunk.src_interval,
        "tgt_interval": input_chunk.tgt_interval,
        "src_tokens": input_chunk.src_tokens,
        "tgt_tokens": input_chunk.tgt_tokens
    }
    
    return data

def test_extract_edits(input_str: str) -> None:
    dataset_ref = XDataset.parse_file_v2(input_str)
    
    gec_dataset_ref = get_chunked_dataset(
        dataset=dataset_ref, merge_distance=1, output_visualize=sys.stdout
    )
    
    edits_strs = []
    # Treat chunks as extracted edits.
    for sample in gec_dataset_ref:
        chunks = list(filter(lambda x: x.types, sample.chunks[0][0]))
        for i in chunks:
            
            edits_str = string_to_json(i)
            edits_strs.append(edits_str)
    edits = {
        "edits": edits_strs
    }
    
    return json.dumps(edits, ensure_ascii=False).strip('{}') + ','

def test_extract_edits_only_exp(input_str: str) -> None:
    dataset_ref = XDataset.parse_file_v2(input_str)
    
    gec_dataset_ref = get_chunked_dataset(
        dataset=dataset_ref, merge_distance=1, output_visualize=sys.stdout
    )
    
    edits_strs = []
    # Treat chunks as extracted edits.
    for sample in gec_dataset_ref:
        chunks = list(filter(lambda x: x.types, sample.chunks[0][0]))
        for i in chunks:
            edits_str = string_to_json(i)
            edits_strs.append(edits_str)
    edits = {
        "edits": edits_strs
    }
    
    
    return edits_strs


def test_evaluation(filepath_hyp,filepath_ref) -> None:
    #filepath_ref = "./data/demo/ref.json"
    #filepath_hyp = "./data/demo/hyp.json"

    dataset_ref = XDataset.parse_file_v1(filepath_ref)
    dataset_hyp = XDataset.parse_file_v1(filepath_hyp)

    results = evaluate(dataset_ref=dataset_ref, dataset_hyp=dataset_hyp)
    print(results)
    print("--------------")
    print(filepath_hyp)


def test_evaluation2() -> None:
    dataset_ref = XDataset()
    dataset_hyp = XDataset()
    sample_ref = XSample(
        index=0,
        domain="test",
        source="第一；病者所得的病是否无药可救？",
        target="第一：病人所得的病是否无药可救？",
        edits=[
            XEdit(
                src_interval=[2, 5],
                tgt_interval=[2, 5],
                src_content="；病者",
                tgt_content="：病人",
                error_type="标点误用",
                error_severity=4,
                error_description="【；】通常用于表示句子内部并列关系的稍微弱于句号的停顿，而【：】则用于引出解释、说明或詳述的内容，此处列表的起始更适合使用【：】而非【；】，因此应将【；】替换为{：}，使句子语气更加准确。同时，【病者】指病人的古汉语说法，现代汉语中更常使用【病人】，所以需要将【病者】替换为{病人}，使其更加符合现代汉语习惯。",
            )
        ],
    )
    dataset_ref.append(sample_ref)

    sample_hyp = XSample(
        index=0,
        domain="test",
        source="第一；病者所得的病是否无药可救？",
        target="第一：病者所得的病是否为无药可用的？",
        edits=[
            XEdit(
                src_interval=[2, 3],
                tgt_interval=[2, 3],
                src_content="；",
                tgt_content="：",
                error_type="标点误用",
                error_severity=4,
                error_description="在中文语境中，应使用冒号【：】来提示下文开始。因此应将【；】替换为{：}。",
            )
        ],
    )
    dataset_hyp.append(sample_hyp)

    results = evaluate(dataset_ref=dataset_ref, dataset_hyp=dataset_hyp)
    print(results)

def extract_before_explanations(output_string):
    # 找到 "explanations" 的起始位置
    index = output_string.find("\"explanations\"")
    # 截取从开头到 "explanations" 开始之前的内容
    if index != -1:
        return output_string[:index]
    return output_string
