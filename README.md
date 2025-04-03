# EXCGEC: A Benchmark for Edit-Wise Explainable Chinese Grammatical Error Correction

</div>

-----

The repository contains the codes and data for our AAAI 2025 Main Paper: [EXCGEC: A Benchmark for Edit-Wise Explainable Chinese Grammatical Error Correction](https://arxiv.org/abs/2407.00924)

This paper introduces the EXGEC task, establishes the EXCGEC benchmark, and provides a comprehensive evaluation suite to advance the study of explainable Grammatical Error Correction.

## Features
* We propose the EXGEC task and establish the EXCGEC benchmark with a Chinese dataset and comprehensive metrics.
* We develop EXGEC baseline models and investigate the performance of various LLMs using the proposed benchmark.
* We perform detailed analyses and human evaluation experiments to assess the effectiveness of automatic metrics for error descriptions.

## Requirements and Installation
Python version >= 3.10

```bash
git clone https://https://github.com/THUKElab/EXCGEC.git
conda create -n excgec-eval python=3.10.14
conda activate excgec-eval
pip install  -r eval_requirements.txt
conda deactivate

conda create -n excgec python=3.10.14
conda activate excgec
cd LLaMA-Factory
pip install -e .[metrics]
```

## Usage
----
```perl
excgec
├── benchmarks/            # Contains scripts and tools for data processing and benchmark evaluation.
├── evaluation/            # Includes tools and scripts for model evaluation and performance metrics.
├── excgec_generation/     # Holds decoding strategies and related components for EXGEC (Explainable Grammatical Error Correction).
├── exp-cgec/              # Main directory containing executable files for training, fine-tuning, and evaluation.
├── LLaMA-Factory/         # Contains components related to the LLaMA model factory and setup.
├── util/                  # Miscellaneous utility scripts and functions for various tasks.
└── LLM/                   # Stores fine-tuned models and associated code.

```
EXCGEC finetuning shell file
```bash
cd exp-cgec
bash excgec_finetuning_lora.sh
```

