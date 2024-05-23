import sys

from benchmarks.xcgec.evaluate import evaluate, get_chunked_dataset
from benchmarks.xcgec.objects import XDataset, XEdit, XSample


def test_extract_edits() -> None:
    filepath_ref = "/data/yejh/nlp/EXCGEC/exp-cgec/data/test1.json"
    dataset_ref = XDataset.parse_file_v1(filepath_ref)

    gec_dataset_ref = get_chunked_dataset(
        dataset=dataset_ref, merge_distance=1, output_visualize=sys.stdout
    )

    # Treat chunks as extracted edits.
    for sample in gec_dataset_ref:
        # print(sample)
        chunks = list(filter(lambda x: x.types, sample.chunks[0][0]))
        print(f"Source: {sample.source[0]}")
        print(f"Target: {sample.target[0]}")
        print("Chunks: " + "\n".join(map(str, chunks)))
        print()


def test_evaluation() -> None:
    filepath_ref = "benchmarks/xcgec/data/demo/ref.json"
    filepath_hyp = "benchmarks/xcgec/data/demo/hyp.json"

    dataset_ref = XDataset.parse_file_v1(filepath_ref)
    dataset_hyp = XDataset.parse_file_v1(filepath_hyp)

    results = evaluate(dataset_ref=dataset_ref, dataset_hyp=dataset_hyp)
    print(results)


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


if __name__ == "__main__":
    # test_extract_edits()
    test_evaluation2()
