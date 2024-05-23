from abc import ABC, abstractmethod
from typing import Any, List, Tuple

import numpy as np


class BaseAligner(ABC):
    """Align the source and target sentences.

    Args:
        del_cost (float): Cost of deletion.
        ins_cost (float): Cost of insertion.
        standard (bool): Whether use standard Standard Levenshtein. Default to False.
        brute_force (bool): Whether brute-force search all possible alignments.
            Defaults to False. Setting to Ture may introduce heavy computation.
    """

    def __init__(
        self,
        del_cost: float = 1.0,
        ins_cost: float = 1.0,
        standard: bool = False,
        brute_force: bool = False,
        verbose: bool = False,
    ) -> None:
        self.standard = standard
        self.del_cost = del_cost
        self.ins_cost = ins_cost
        self.brute_force = brute_force
        self.verbose = verbose
        self.align_seqs = None

    def signature(self) -> str:
        """Returns a signature for the tokenizer."""
        raise NotImplementedError()

    def __call__(self, source: List[str], target: List[str]) -> List[Tuple]:
        """Aligns parallel source-target sentences with the aligner.

        Args:
            source (List[str]): Source sentence.
            target (List[str]): Target sentence.

        Returns:
            List[Tuple]: Alignment sequence [(op, o_start, o_end, c_start, c_end), ...]
        """
        # Align orig and cor and get the cost and op matrices
        cost_matrix, oper_matrix = self.align(source, target)
        # Get the cheapest align sequence from the op matrix
        align_seq = self.get_cheapest_align_seq(oper_matrix)

        if self.verbose:
            print(f"Source: {source}")
            print(f"Target: {target}")
            print(f"Cost Matrix: {cost_matrix}")
            print(f"Oper Matrix: {oper_matrix}")
            print(f"Alignment: {align_seq}")
            for a in align_seq:
                print(a[0], source[a[1] : a[2]], target[a[3] : a[4]])
        return align_seq

    def align(self, source: List[str], target: List[str]) -> Tuple:
        """基于改进的动态规划算法，为原句子的每个字打上编辑标签，以便使它能够成功转换为目标句子。
        编辑操作类别：
        1) M: Match (KEEP)，即当前字保持不变
        2) D: Delete，删除，即当前字需要被删除
        3) I: Insert，插入，即当前字需要被插入
        4) T: Transposition，移位操作，即涉及到词序问题
        """
        # Create the cost_matrix and the oper_matrix
        cost_matrix = np.zeros((len(source) + 1, len(target) + 1))
        oper_matrix = np.full((len(source) + 1, len(target) + 1), "O", dtype=object)

        # Fill in the edges
        for i in range(1, len(source) + 1):
            cost_matrix[i][0] = cost_matrix[i - 1][0] + 1
            oper_matrix[i][0] = ["D"] if self.brute_force else "D"
        for j in range(1, len(target) + 1):
            cost_matrix[0][j] = cost_matrix[0][j - 1] + 1
            oper_matrix[0][j] = ["I"] if self.brute_force else "I"

        # Loop through the cost matrix
        for i in range(len(source)):
            for j in range(len(target)):
                if self.signature() == "eng":
                    src_token, tgt_token = source[i].orth, target[j].orth
                elif self.signature() == "zho":
                    src_token, tgt_token = source[i][0], target[j][0]
                else:
                    raise NotImplementedError(f"signature: {self.signature()}")

                if src_token == tgt_token:  # Match
                    cost_matrix[i + 1][j + 1] = cost_matrix[i][j]
                    oper_matrix[i + 1][j + 1] = ["M"] if self.brute_force else "M"
                else:  # Non-match
                    del_cost = cost_matrix[i][j + 1] + self.del_cost
                    ins_cost = cost_matrix[i + 1][j] + self.ins_cost
                    trans_cost = float("inf")
                    if self.standard:  # Standard Levenshtein (S = 1)
                        sub_cost = cost_matrix[i][j] + 1
                    else:
                        # Custom substitution
                        sub_cost = cost_matrix[i][j] + self.get_sub_cost(
                            source[i], target[j]
                        )
                        # Transpositions require >=2 tokens
                        # Traverse the diagonal while there is not a Match.
                        k = 1
                        while (
                            i - k >= 0
                            and j - k >= 0
                            and cost_matrix[i - k + 1][j - k + 1]
                            != cost_matrix[i - k][j - k]
                        ):
                            if self.signature() == "eng":
                                p1 = sorted([a.lower for a in source[i - k : i + 1]])
                                p2 = sorted([b.lower for b in target[j - k : j + 1]])
                            elif self.signature() == "zho":
                                p1 = sorted([a[0] for a in source[i - k : i + 1]])
                                p2 = sorted([b[0] for b in target[j - k : j + 1]])
                            else:
                                raise NotImplementedError(
                                    f"signature: {self.signature()}"
                                )
                            if p1 == p2:
                                trans_cost = cost_matrix[i - k][j - k] + k
                                break
                            k += 1
                    costs = [trans_cost, sub_cost, ins_cost, del_cost]
                    # Get the index of the cheapest (first cheapest if tied)
                    ind = costs.index(min(costs))
                    # Save the cost and the op in the matrices
                    cost_matrix[i + 1][j + 1] = costs[ind]

                    if not self.brute_force:
                        if ind == 0:
                            oper_matrix[i + 1][j + 1] = "T" + str(k + 1)
                        elif ind == 1:
                            oper_matrix[i + 1][j + 1] = "S"
                        elif ind == 2:
                            oper_matrix[i + 1][j + 1] = "I"
                        else:
                            oper_matrix[i + 1][j + 1] = "D"
                    else:
                        for idx, cost in enumerate(costs):
                            if cost == costs[ind]:
                                if idx == 0:
                                    if oper_matrix[i + 1][j + 1] == "O":
                                        oper_matrix[i + 1][j + 1] = ["T" + str(k + 1)]
                                    else:
                                        oper_matrix[i + 1][j + 1].append(
                                            "T" + str(k + 1)
                                        )
                                elif idx == 1:
                                    if oper_matrix[i + 1][j + 1] == "O":
                                        oper_matrix[i + 1][j + 1] = ["S"]
                                    else:
                                        oper_matrix[i + 1][j + 1].append("S")
                                elif idx == 2:
                                    if oper_matrix[i + 1][j + 1] == "O":
                                        oper_matrix[i + 1][j + 1] = ["I"]
                                    else:
                                        oper_matrix[i + 1][j + 1].append("I")
                                else:
                                    if oper_matrix[i + 1][j + 1] == "O":
                                        oper_matrix[i + 1][j + 1] = ["D"]
                                    else:
                                        oper_matrix[i + 1][j + 1].append("D")
        return cost_matrix, oper_matrix

    def get_cheapest_align_seq(self, oper_matrix: np.ndarray) -> List[Tuple]:
        """Retrieve the editing sequence with the smallest cost through backtracking.

        Args:
            oper_matrix (np.ndarray): Two-dimension operation matrix

        Returns:
            List[Tuple]: [(op, o_start, o_end, c_start, c_end), ...]
        """
        if self.brute_force:  # BF search
            return self.get_cheapest_align_seq_bf(oper_matrix)

        align_seq = []
        i = oper_matrix.shape[0] - 1
        j = oper_matrix.shape[1] - 1
        # Work backwards from bottom right until we hit top left
        while i + j != 0:
            # Get the edit operation in the current cell
            op = oper_matrix[i][j]
            if op in {"M", "S"}:  # Matches and substitutions
                align_seq.append((op, i - 1, i, j - 1, j))
                i -= 1
                j -= 1
            elif op == "D":  # Deletions
                align_seq.append((op, i - 1, i, j, j))
                i -= 1
            elif op == "I":  # Insertions
                align_seq.append((op, i, i, j - 1, j))
                j -= 1
            else:  # Transpositions
                # Get the size of the transposition
                k = int(op[1:])
                align_seq.append((op, i - k, i, j - k, j))
                i -= k
                j -= k
        # Reverse the list to go from left to right and return
        align_seq.reverse()
        return align_seq

    def get_cheapest_align_seq_bf(self, oper_matrix: np.ndarray) -> List[Tuple]:
        self.align_seqs = []
        i = oper_matrix.shape[0] - 1
        j = oper_matrix.shape[1] - 1
        if abs(i - j) > 10:
            self._dfs(i, j, [], oper_matrix, "first")
        else:
            self._dfs(i, j, [], oper_matrix, "all")
        final_align_seqs = [seq[::-1] for seq in self.align_seqs]
        return final_align_seqs

    def _dfs(self, i, j, align_seq_now, oper_matrix, strategy: str = "all") -> None:
        """DFS to obtain all sequences with the same minimum cost."""
        if i + j == 0:
            self.align_seqs.append(align_seq_now)
        else:  # 可以类比成搜索一棵树从根结点到叶子结点的所有路径
            ops = oper_matrix[i][j]
            if strategy != "all":
                ops = ops[:1]
            for op in ops:
                if op in {"M", "S"}:
                    self._dfs(
                        i - 1,
                        j - 1,
                        align_seq_now + [(op, i - 1, i, j - 1, j)],
                        oper_matrix,
                        strategy,
                    )
                elif op == "D":
                    self._dfs(
                        i - 1,
                        j,
                        align_seq_now + [(op, i - 1, i, j, j)],
                        oper_matrix,
                        strategy,
                    )
                elif op == "I":
                    self._dfs(
                        i,
                        j - 1,
                        align_seq_now + [(op, i, i, j - 1, j)],
                        oper_matrix,
                        strategy,
                    )
                else:
                    k = int(op[1:])
                    self._dfs(
                        i - k,
                        j - k,
                        align_seq_now + [(op, i - k, i, j - k, j)],
                        oper_matrix,
                        strategy,
                    )

    @abstractmethod
    def get_sub_cost(self, src_token: Any, tgt_token: Any) -> float:
        raise NotImplementedError()
