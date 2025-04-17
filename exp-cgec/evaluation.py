import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.xcgec.test_evaluation import test_evaluation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation for EXCGEC predictions.")

    parser.add_argument("--filepath_hyp", type=str, required=True, help="Path to the hypothesis (predicted) JSON file.")
    parser.add_argument("--filepath_ref", type=str, required=True, help="Path to the reference JSON file.")

    args = parser.parse_args()

    test_evaluation(args.filepath_hyp, args.filepath_ref)
    