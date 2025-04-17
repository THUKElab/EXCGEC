from abc import ABC, abstractmethod


class BaseClassifier(ABC):
    """A base dummy Classifier to derive from."""

    @abstractmethod
    def signature(self) -> str:
        """Returns a signature for the tokenizer."""
        raise NotImplementedError()

    @abstractmethod
    def __call__(self, source, target, edit):
        """Classifies grammatical errors with the classifier."""
        raise NotImplementedError()
