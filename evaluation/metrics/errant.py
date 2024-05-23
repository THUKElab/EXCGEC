from typing import List

from data import Edit, Sample

from ..aligners import AlignerType, get_aligner
from ..classifers import ClassifierType, get_classifier
from ..mergers import MergerType, MergeStrategy, get_merger
from ..schema import BaseEditMetricResult, SampleMetricResult
from ..scorers import ScorerType
from ..tokenizers import TokenizerType
from .base import BaseEditMetric


class Errant(BaseEditMetric):
    """Computes the ERRANT score given hypotheses and references.

    Args:
        lang (str): _description_
        scorer (ScorerType, optional): _description_. Defaults to ScorerType.PRF.
        tokenizer (TokenizerType, optional): _description_. Defaults to None.
        aligner (AlignerType, optional): _description_. Defaults to None.
        merger (MergerType, optional): _description_. Defaults to None.
        classifier (ClassifierType, optional): _description_. Defaults to None.
        enable_tqdm (bool, optional): _description_. Defaults to True.
        aligner_standard (bool, optional): Align parallel texts using standard Levenshtein.
            Defaults to False.
        merger_strategy (str, optional): Merging strategy for automatic alignment.
            Defaults to None.

    Args for English (Eng):
        use_spacy (bool): Tokenize the text using spacy (default: False)
        spacy_model (str): Align parallel texts using standard Levenshtein (default: False)

    Args for Chinese (Zho):
        granularity (str):
    """

    def __init__(
        self,
        lang: str,
        scorer_type: ScorerType = ScorerType.PRF,
        tokenizer_type: TokenizerType = None,
        aligner_type: AlignerType = None,
        aligner_standard: bool = False,
        merger_type: MergerType = None,
        merger_strategy: MergeStrategy = None,
        classifier_type: ClassifierType = None,
        enable_tqdm: bool = True,
    ) -> None:
        tokenizer_type = tokenizer_type or lang
        aligner_type = aligner_type or lang
        merger_type = merger_type or lang
        classifier_type = classifier_type or lang

        super().__init__(
            lang=lang,
            tokenizer_type=tokenizer_type,
            scorer_type=scorer_type,
            enable_tqdm=enable_tqdm,
        )
        self.aligner = get_aligner(aligner_type, standard=aligner_standard)
        self.merger = get_merger(merger_type, strategy=merger_strategy)
        self.classifier = get_classifier(classifier_type)

    def pickable_edits(
        self, sample_edits: List[List[List[Edit]]]
    ) -> List[List[List[Edit]]]:
        """Make edits pickable."""
        for edits in sample_edits[0]:
            for edit in edits:
                edit.src_tokens_tok = None
                edit.tgt_tokens_tok = None
        return sample_edits

    def parallel_to_edits(self, sample: Sample) -> List[List[List[Edit]]]:
        """Generate edits of the given sample.

        Args:
            sample (Sample): Sample with sources and targets.

        Returns:
            List[List[List[Edit]]]: Generated edits.
        """
        # source, target = source.strip(), target.strip()
        if not sample.target:
            return [[[]]]
        src_tok = self.tokenizer(sample.source[0])
        src_detok = self.tokenizer.detokenize(src_tok).strip()

        sample_edits = [[]]
        for idx, target in enumerate(sample.target):
            if src_detok != target:
                # Tokenize target
                tgt_tok = self.tokenizer(sample.target[idx])
                # Align source and target
                align_seq = self.aligner(src_tok, tgt_tok)
                # Merge alignment
                edits = self.merger(src_tok, tgt_tok, align_seq, idx)
                # Update Edit object with an updated error type
                for edit in edits:
                    self.classifier(src_tok, tgt_tok, edit)
                sample_edits[0].append(edits)
            else:
                sample_edits[0].append([])
        return self.pickable_edits(sample_edits)

    def evaluate_sample_correction(
        self, sample_hyp: Sample, sample_ref: Sample
    ) -> SampleMetricResult:
        """Acquire TP, FP and FN correction counts.

        Args:
            sample_hyp (Sample): Hypothesis Sample.
            sample_ref (Sample): References Sample.

        Returns:
            MetricSampleResult: Correction result of ERRANT.
        """
        ref_results: List[BaseEditMetricResult] = []
        for ref_edits in sample_ref.edits[0]:
            tp_edits, fp_edits, fn_edits = [], [], []
            # TODO: On occasion, multiple tokens at same span
            for hyp_edit in sample_hyp.edits[0][0]:
                if hyp_edit in ref_edits:
                    tp_edits.append(hyp_edit)
                else:
                    fp_edits.append(hyp_edit)
            for ref_edit in ref_edits:
                if ref_edit not in sample_hyp.edits[0][0]:
                    fn_edits.append(ref_edit)
            ref_result = BaseEditMetricResult(
                tp_edits=tp_edits.copy(),
                fp_edits=fp_edits.copy(),
                fn_edits=fn_edits.copy(),
            )
            ref_results.append(ref_result)
        return SampleMetricResult(ref_results=ref_results)

    def evaluate_sample_detection(
        self, sample_hyp: Sample, sample_ref: Sample
    ) -> SampleMetricResult:
        """Acquire TP, FP and FN detection counts.

        Args:
            sample_hyp (Sample): Hypothesis Samples.
            sample_ref (Sample): References Samples.

        Returns:
            MetricSampleResult: Detection result of ERRANT.
        """
        pass
