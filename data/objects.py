import copy
from typing import Any, Iterator, List, Optional

from pydantic import BaseModel, Field

EDIT_TEMPLATE = "A {i1} {i2}|||{type}|||{target}|||REQUIRED|||-NONE-|||{target_index}"


class Edit(BaseModel):
    src_interval: List[int] = Field(default=None, description="")
    tgt_interval: List[int] = Field(default=None, description="")
    src_tokens: List[str] = Field(default=None, description="")
    tgt_tokens: List[str] = Field(default=None, description="")
    src_tokens_tok: Optional[List[Any]] = Field(default=None, description="")
    tgt_tokens_tok: Optional[List[Any]] = Field(default=None, description="")
    tgt_index: int = Field(default=None, description="")
    types: List[str] = Field(default_factory=list, description="")
    weight: Optional[float] = Field(default=None, description="")

    @property
    def source(self) -> str:
        return " ".join(self.src_tokens)

    @property
    def target(self) -> str:
        return " ".join(self.tgt_tokens)

    @property
    def m2(self) -> str:
        return EDIT_TEMPLATE.format(
            self.src_interval[0],
            self.src_interval[1],
            self.types[0],
            self.target,
            self.tgt_index,
        )

    def is_valid(self) -> bool:
        tgt_beg_idx = self.tgt_interval[0]
        tgt_end_idx = self.tgt_interval[1]
        return self.tgt_tokens[tgt_beg_idx:tgt_end_idx] == self.tgt_tokens

    def __eq__(self, other: "Edit") -> bool:
        if self.src_interval == other.src_interval:
            if self.src_tokens != other.src_tokens:
                raise ValueError(
                    f"Invalid Edit comparison:\nEdit 1:{self}\nEdit 2:{other}"
                )
            elif self.tgt_tokens == other.tgt_tokens:
                return True
        return False

    def __hash__(self) -> int:
        return (
            hash(tuple(self.src_interval))
            + hash(tuple(self.tgt_tokens))
            + hash(tuple(self.src_tokens))
            + hash(tuple(self.types))
        )

    def __deepcopy__(self, memodict={}) -> "Edit":
        return Edit(
            src_interval=self.src_interval.copy(),
            tgt_interval=self.tgt_interval.copy(),
            src_tokens=self.src_tokens.copy(),
            tgt_tokens=self.tgt_tokens.copy(),
            src_tokens_tok=copy.deepcopy(self.src_tokens_tok),
            tgt_tokens_tok=copy.deepcopy(self.tgt_tokens_tok),
            types=self.types.copy(),
            weight=self.weight,
        )


class Chunk(Edit):
    chunk_index: int = Field(default=None, description="Chunk index")

    def __eq__(self, other: "Chunk") -> bool:
        if self.chunk_index == other.chunk_index:
            return super().__eq__(other)
        return False

    def __hash__(self) -> int:
        return super().__hash__() + hash(self.chunk_index)

    def __deepcopy__(self, memodict={}) -> "Chunk":
        return Chunk(
            src_interval=self.src_interval.copy(),
            tgt_interval=self.tgt_interval.copy(),
            src_tokens=self.src_tokens.copy(),
            tgt_tokens=self.tgt_tokens.copy(),
            src_tokens_tok=copy.deepcopy(self.src_tokens_tok),
            tgt_tokens_tok=copy.deepcopy(self.tgt_tokens_tok),
            types=self.types.copy(),
            weight=self.weight,
            chunk_index=self.chunk_index,
        )


class Sample(BaseModel):
    index: int = Field(default=None, description="")
    source: List[str] = Field(default=None, description="")
    target: List[str] = Field(default=None, description="")
    edits: List[List[List[Edit]]] = Field(default=None, description="")
    chunks: List[List[List[Chunk]]] = Field(default=None, description="")

    def contains_empty(self) -> bool:
        return any([not x for x in self.source + self.target])

    def __deepcopy__(self, memodict={}) -> "Sample":
        
        return Sample(
            index=self.index,
            source=self.source.copy(),
            target=self.target.copy(),
            edits=copy.deepcopy(self.edits),
            chunks=copy.deepcopy(self.chunks),
        )

    def has_replica(self) -> bool:
        return len(self.target) != len(set(self.target))

    def has_unchanged(self) -> bool:
        return any([x == self.source[0] for x in self.target])


class Dataset(BaseModel):
    samples: List[Sample] = Field(default_factory=list)

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[Sample]:
        return iter(self.samples)

    def __getitem__(self, item: int) -> Sample:
        return self.samples[item]

    def append(self, sample: Sample) -> None:
        self.samples.append(sample)

    def extend(self, dataset: "Dataset") -> None:
        orig_len = len(self)
        self.samples.extend(dataset.samples)
        for sample_idx in range(orig_len, len(self.samples)):
            self.samples[sample_idx].index = sample_idx

    def merge(self, dataset: "Dataset"):
        if len(self) != len(dataset):
            raise ValueError(
                f"Both datasets must contain equal samples: {len(self)} != {len(dataset)}."
            )

        for sample1, sample2 in zip(self, dataset):
            if sample1.source != sample2.source:
                raise ValueError(
                    f"Both samples must contain same sources:\n"
                    f"Source 1: {sample1.source}\n"
                    f"Source 2: {sample2.source}"
                )
            sample1.target.extend(sample2.target)
            sample1.edits[0].extend(sample2.edits[0])

    def reorder(self) -> None:
        for idx, sample in enumerate(self.samples):
            sample.index = idx

    def flatten(self) -> "Dataset":
        new_dataset = Dataset()
        for sample in self.samples:
            for sid, src in enumerate(sample.source):
                for tid, tgt in enumerate(sample.target):
                    new_edits = (
                        [[copy.deepcopy(sample.edits[sid][tid])]]
                        if sample.edits is not None
                        else None
                    )
                    new_chunks = (
                        [[copy.deepcopy(sample.chunks[sid][tid])]]
                        if sample.chunks is not None
                        else None
                    )
                    new_sample = Sample(
                        index=len(new_dataset),
                        source=[src],
                        target=[tgt],
                        edits=new_edits,
                        chunks=new_chunks,
                    )
                    new_dataset.append(new_sample)
        return new_dataset


def apply_edits(src_tokens: List[str], edits: List[Edit]) -> List[str]:
    """Generate target tokens by applying edits to src_tokens.

    Args:
        src_tokens (List[str]): Source tokens.
        edits (List[Edit]): Edits.

    Returns:
        List[str]: Target tokens.
    """
    tgt_offset, tgt_tokens = 0, src_tokens.copy()
    for edit in edits:
        src_beg_idx, src_end_idx = edit.src_interval[0], edit.src_interval[1]
        tgt_beg_idx = src_beg_idx + tgt_offset
        tgt_end_idx = tgt_beg_idx + len(edit.tgt_tokens)

        tgt_tokens[tgt_beg_idx : src_end_idx + tgt_offset] = edit.tgt_tokens
        tgt_offset += len(edit.tgt_tokens) - len(edit.src_tokens)

        # Sanity Check
        if edit.src_tokens != src_tokens[src_beg_idx:src_end_idx]:
            raise ValueError(
                f"Inconsistent Source Tokens: {edit} != {src_tokens[src_beg_idx: src_end_idx]}"
            )
        if edit.tgt_tokens != tgt_tokens[tgt_beg_idx:tgt_end_idx]:
            raise ValueError(
                f"Inconsistent Target Tokens: {edit} != {tgt_tokens[src_beg_idx: src_end_idx]}"
            )
    return tgt_tokens
