import sys
from copy import deepcopy
from typing import Dict, List

from data import Sample

from ..constants import KEY_FN, KEY_FP, KEY_TN, KEY_TP
from .base import BaseEditMetric


class MaxMatch(BaseEditMetric):
    def __init__(
        self,
        lang: str,
        tokenizer: str = None,
        scorer: str = "corpus",
        enable_tqdm: bool = False,
        **kwargs,
    ):
        super(MaxMatch, self).__init__(scorer=scorer, enable_tqdm=enable_tqdm)
        self.tokenizer = self.build_tokenizer(
            name=tokenizer if tokenizer else lang,
            **kwargs,
        )
        self.candidate_max_unchanged_words = kwargs.get(
            "candidate_max_unchanged_words", 2
        )
        self.reference_max_unchanged_words = kwargs.get(
            "reference_max_unchanged_words", 0
        )
        self.ignore_whitespace_casing = kwargs.get("ignore_whitespace_casing", False)

    def evaluate_sample(
        self,
        sample_hyp: Sample,
        sample_ref: Sample,
        *args,
        **kwargs,
    ) -> List[Dict]:
        assert (
            len(sample_hyp.source)
            == len(sample_hyp.target)
            == len(sample_ref.source)
            == 1
        )
        assert (
            sample_hyp.source[0] == sample_ref.source[0]
        ), f"Source Not Equal: {sample_hyp.source[0]} || {sample_ref.source[0]}"
        source, candidate, targets = (
            sample_hyp.source[0],
            sample_hyp.target[0],
            sample_ref.target,
        )
        target_edits = sample_ref.edits[0] if sample_ref.edits else None
        source, candidate, targets = (
            source.strip(),
            candidate.strip(),
            list(map(lambda x: x.strip(), targets)),
        )
        if sample_hyp.source_tokens and sample_hyp.target_tokens:
            source_tok = self.tokenizer.convert_tokens(sample_hyp.source_tokens[0])
            candidate_tok = self.tokenizer.convert_tokens(sample_hyp.target_tokens[0])
        else:
            source_tok, candidate_tok = self.tokenizer.tokenize(
                source
            ), self.tokenizer.tokenize(candidate)
        # print(f"{source=}\n{candidate=}\n{targets=}")
        V, E, dist, edits = self.get_graph(
            source_tok, candidate_tok, self.candidate_max_unchanged_words
        )
        result = []
        for index, target in enumerate(targets):
            if target_edits:
                target_edit_seq = []
                for edit in target_edits[index]:
                    target_edit_seq.append(
                        (
                            edit.src_interval[0],
                            edit.src_interval[1],
                            " ".join(edit.src_tokens),
                            [" ".join(edit.tgt_tokens)],
                        )
                    )
            else:
                if sample_ref.target_tokens:
                    target_tok = self.tokenizer.convert_tokens(
                        sample_ref.target_tokens[index]
                    )
                else:
                    target_tok = self.tokenizer.tokenize(target)
                target_edit_seq = self.get_graph_edit_seq(
                    source_tok, target_tok, self.reference_max_unchanged_words
                )
                target_edit_seq = [
                    (*item[:-1], [item[-1]]) for item in reversed(target_edit_seq)
                ]
            candidate_edit_seq = self.get_edit_seq(V, E, dist, edits, target_edit_seq)
            correct = self.matchSeq(candidate_edit_seq, target_edit_seq)
            # print(f"{candidate_edit_seq=}, {len(candidate_edit_seq)=}\n{target_edit_seq=},
            # {len(target_edit_seq)=}\n{correct=}, {len(correct)=}")
            result.append(
                {
                    KEY_TP: len(correct),
                    KEY_FP: len(candidate_edit_seq) - len(correct),
                    KEY_FN: len(target_edit_seq) - len(correct),
                    KEY_TN: 0,
                }
            )
        return result

    def get_graph_edit_seq(self, source_tok, target_tok, max_unchanged_words):
        return self.get_edit_seq(
            *self.get_graph(source_tok, target_tok, max_unchanged_words)
        )

    def get_graph(self, source_tok, target_tok, max_unchanged_words):
        lmatrix1, backpointers1 = self.levenshtein_matrix(
            source_tok, target_tok, 1, 1, 1
        )
        lmatrix2, backpointers2 = self.levenshtein_matrix(
            source_tok, target_tok, 1, 1, 2
        )
        V1, E1, dist1, edits1 = self.edit_graph(lmatrix1, backpointers1)
        V2, E2, dist2, edits2 = self.edit_graph(lmatrix2, backpointers2)
        V, E, dist, edits = self.merge_graph(
            V1, V2, E1, E2, dist1, dist2, edits1, edits2
        )
        V, E, dist, edits = self.transitive_arcs(V, E, dist, edits, max_unchanged_words)
        return V, E, dist, edits

    def get_edit_seq(self, V, E, dist, edits, gold=[]):
        equals_ignore_whitespace_casing = (
            lambda a, b: a.replace(" ", "").lower() == b.replace(" ", "").lower()
        )
        dist = self.set_weights(E, dist, edits, gold)
        edit_seq = self.best_edit_seq_bf(V, E, dist, edits)
        if self.ignore_whitespace_casing:
            edit_seq = [
                x for x in edit_seq if not equals_ignore_whitespace_casing(x[2], x[3])
            ]
        return edit_seq

    def levenshtein_matrix(self, first, second, cost_ins=1, cost_del=1, cost_sub=2):
        first_length = len(first) + 1
        second_length = len(second) + 1

        # init
        distance_matrix = [[None] * second_length for x in range(first_length)]
        backpointers = {}
        distance_matrix[0][0] = 0
        for i in range(1, first_length):
            distance_matrix[i][0] = i
            edit = ("del", i - 1, i, first[i - 1], "", 0)
            backpointers[(i, 0)] = [((i - 1, 0), edit)]
        for j in range(1, second_length):
            distance_matrix[0][j] = j
            edit = (
                "ins",
                0,
                0,
                "",
                second[j - 1],
                0,
            )  # always insert from the beginning
            # edit = ("ins", j-1, j-1, '', second[j-1], 0)
            backpointers[(0, j)] = [((0, j - 1), edit)]

        # fill the matrix
        for i in range(1, first_length):
            for j in range(1, second_length):
                deletion = distance_matrix[i - 1][j] + cost_del
                insertion = distance_matrix[i][j - 1] + cost_ins
                if first[i - 1] == second[j - 1]:
                    substitution = distance_matrix[i - 1][j - 1]
                else:
                    substitution = distance_matrix[i - 1][j - 1] + cost_sub
                if substitution == min(substitution, deletion, insertion):
                    distance_matrix[i][j] = substitution
                    if first[i - 1] != second[j - 1]:
                        edit = ("sub", i - 1, i, first[i - 1], second[j - 1], 0)
                    else:
                        edit = ("noop", i - 1, i, first[i - 1], second[j - 1], 1)
                    try:
                        backpointers[(i, j)].append(((i - 1, j - 1), edit))
                    except KeyError:
                        backpointers[(i, j)] = [((i - 1, j - 1), edit)]
                if deletion == min(substitution, deletion, insertion):
                    distance_matrix[i][j] = deletion
                    edit = ("del", i - 1, i, first[i - 1], "", 0)
                    try:
                        backpointers[(i, j)].append(((i - 1, j), edit))
                    except KeyError:
                        backpointers[(i, j)] = [((i - 1, j), edit)]
                if insertion == min(substitution, deletion, insertion):
                    distance_matrix[i][j] = insertion
                    edit = ("ins", i, i, "", second[j - 1], 0)
                    try:
                        backpointers[(i, j)].append(((i, j - 1), edit))
                    except KeyError:
                        backpointers[(i, j)] = [((i, j - 1), edit)]
        return (distance_matrix, backpointers)

    def edit_graph(self, levi_matrix, backpointers):
        V = []
        E = []
        dist = {}
        edits = {}
        # breath-first search through the matrix
        v_start = (len(levi_matrix) - 1, len(levi_matrix[0]) - 1)
        queue = [v_start]
        while len(queue) > 0:
            v = queue[0]
            queue = queue[1:]
            if v in V:
                continue
            V.append(v)
            try:
                for vnext_edits in backpointers[v]:
                    vnext = vnext_edits[0]
                    edit_next = vnext_edits[1]
                    E.append((vnext, v))
                    dist[(vnext, v)] = 1
                    edits[(vnext, v)] = edit_next
                    if vnext not in queue:
                        queue.append(vnext)
            except KeyError:
                pass
        return (V, E, dist, edits)

    def transitive_arcs(self, V, E, dist, edits, max_unchanged_words):
        def get_distance(dist, v1, v2):
            return dist.get((v1, v2), float("inf"))

        for k in range(len(V)):
            vk = V[k]
            for i in range(len(V)):
                vi = V[i]
                try:
                    eik = edits[(vi, vk)]
                except KeyError:
                    continue
                for j in range(len(V)):
                    vj = V[j]
                    try:
                        ekj = edits[(vk, vj)]
                    except KeyError:
                        continue
                    dik = get_distance(dist, vi, vk)
                    dkj = get_distance(dist, vk, vj)
                    if dik + dkj < get_distance(dist, vi, vj):
                        eij = self.merge_edits(eik, ekj)
                        if eij[-1] <= max_unchanged_words:
                            E.append((vi, vj))
                            dist[(vi, vj)] = dik + dkj
                            edits[(vi, vj)] = eij
        # remove noop transitive arcs
        for edge in E:
            e = edits[edge]
            if e[0] == "noop" and dist[edge] > 1:
                E.remove(edge)
                dist[edge] = float("inf")
                del edits[edge]
        return (V, E, dist, edits)

    def merge_edits(self, e1, e2, joiner=" "):
        e = None
        if e1[0] == "ins":
            if e2[0] == "ins":
                e = ("ins", e1[1], e2[2], "", e1[4] + joiner + e2[4], e1[5] + e2[5])
            elif e2[0] == "del":
                e = ("sub", e1[1], e2[2], e2[3], e1[4], e1[5] + e2[5])
            elif e2[0] == "sub":
                e = ("sub", e1[1], e2[2], e2[3], e1[4] + joiner + e2[4], e1[5] + e2[5])
            elif e2[0] == "noop":
                e = ("sub", e1[1], e2[2], e2[3], e1[4] + joiner + e2[4], e1[5] + e2[5])
        elif e1[0] == "del":
            if e2[0] == "ins":
                e = ("sub", e1[1], e2[2], e1[3], e2[4], e1[5] + e2[5])
            elif e2[0] == "del":
                e = ("del", e1[1], e2[2], e1[3] + joiner + e2[3], "", e1[5] + e2[5])
            elif e2[0] == "sub":
                e = ("sub", e1[1], e2[2], e1[3] + joiner + e2[3], e2[4], e1[5] + e2[5])
            elif e2[0] == "noop":
                e = ("sub", e1[1], e2[2], e1[3] + joiner + e2[3], e2[4], e1[5] + e2[5])
        elif e1[0] == "sub":
            if e2[0] == "ins":
                e = ("sub", e1[1], e2[2], e1[3], e1[4] + joiner + e2[4], e1[5] + e2[5])
            elif e2[0] == "del":
                e = ("sub", e1[1], e2[2], e1[3] + joiner + e2[3], e1[4], e1[5] + e2[5])
            elif e2[0] == "sub":
                e = (
                    "sub",
                    e1[1],
                    e2[2],
                    e1[3] + joiner + e2[3],
                    e1[4] + joiner + e2[4],
                    e1[5] + e2[5],
                )
            elif e2[0] == "noop":
                e = (
                    "sub",
                    e1[1],
                    e2[2],
                    e1[3] + joiner + e2[3],
                    e1[4] + joiner + e2[4],
                    e1[5] + e2[5],
                )
        elif e1[0] == "noop":
            if e2[0] == "ins":
                e = ("sub", e1[1], e2[2], e1[3], e1[4] + joiner + e2[4], e1[5] + e2[5])
            elif e2[0] == "del":
                e = ("sub", e1[1], e2[2], e1[3] + joiner + e2[3], e1[4], e1[5] + e2[5])
            elif e2[0] == "sub":
                e = (
                    "sub",
                    e1[1],
                    e2[2],
                    e1[3] + joiner + e2[3],
                    e1[4] + joiner + e2[4],
                    e1[5] + e2[5],
                )
            elif e2[0] == "noop":
                e = (
                    "noop",
                    e1[1],
                    e2[2],
                    e1[3] + joiner + e2[3],
                    e1[4] + joiner + e2[4],
                    e1[5] + e2[5],
                )
        else:
            raise ValueError
        return e

    def set_weights(self, E, dist, edits, gold_edits):
        EPSILON = 0.001

        gold_set = deepcopy(gold_edits)
        retdist = deepcopy(dist)

        M = {}
        G = {}
        for edge in E:
            tE = edits[edge]
            s, e = tE[1], tE[2]
            if (s, e) not in M:
                M[(s, e)] = []
            M[(s, e)].append(edge)
            if (s, e) not in G:
                G[(s, e)] = []

        for gold in gold_set:
            s, e = gold[0], gold[1]
            if (s, e) not in G:
                G[(s, e)] = []
            G[(s, e)].append(gold)

        for k in sorted(M.keys()):
            M[k] = sorted(M[k])

            if k[0] == k[1]:  # insertion case
                lptr = 0
                rptr = len(M[k]) - 1
                cur = lptr

                g_lptr = 0
                g_rptr = len(G[k]) - 1

                while lptr <= rptr:
                    hasGoldMatch = False
                    edge = M[k][cur]
                    thisEdit = edits[edge]

                    if cur == lptr:
                        cur_gold = list(range(g_lptr, g_rptr + 1))
                    else:
                        cur_gold = reversed(list(range(g_lptr, g_rptr + 1)))

                    for i in cur_gold:
                        gold = G[k][i]
                        if (
                            thisEdit[1] == gold[0]
                            and thisEdit[2] == gold[1]
                            and thisEdit[3] == gold[2]
                            and thisEdit[4] in gold[3]
                        ):
                            hasGoldMatch = True
                            retdist[edge] = -len(E)
                            if cur == lptr:
                                # g_lptr += 1 # why?
                                g_lptr = i + 1
                            else:
                                # g_rptr -= 1 # why?
                                g_rptr = i - 1
                            break

                    if not hasGoldMatch and thisEdit[0] != "noop":
                        retdist[edge] += EPSILON
                    if hasGoldMatch:
                        if cur == lptr:
                            lptr += 1
                            while lptr < len(M[k]) and M[k][lptr][0] != M[k][cur][1]:
                                if edits[M[k][lptr]] != "noop":
                                    retdist[M[k][lptr]] += EPSILON
                                lptr += 1
                            cur = lptr
                        else:
                            rptr -= 1
                            while rptr >= 0 and M[k][rptr][1] != M[k][cur][0]:
                                if edits[M[k][rptr]] != "noop":
                                    retdist[M[k][rptr]] += EPSILON
                                rptr -= 1
                            cur = rptr
                    else:
                        if cur == lptr:
                            lptr += 1
                            cur = rptr
                        else:
                            rptr -= 1
                            cur = lptr
            else:
                # deletion or substitution, don't care about order,
                # no harm if setting parallel edges weight < 0
                for edge in M[k]:
                    hasGoldMatch = False
                    thisEdit = edits[edge]
                    for gold in G[k]:
                        if (
                            thisEdit[1] == gold[0]
                            and thisEdit[2] == gold[1]
                            and thisEdit[3] == gold[2]
                            and thisEdit[4] in gold[3]
                        ):
                            hasGoldMatch = True
                            retdist[edge] = -len(E)
                            break
                    if not hasGoldMatch and thisEdit[0] != "noop":
                        retdist[edge] += EPSILON
        return retdist

    def best_edit_seq_bf(self, V, E, dist, edits):
        thisdist = {}
        path = {}
        for v in V:
            thisdist[v] = float("inf")
        thisdist[(0, 0)] = 0
        for i in range(len(V) - 1):
            for edge in E:
                v = edge[0]
                w = edge[1]
                if thisdist[v] + dist[edge] < thisdist[w]:
                    thisdist[w] = thisdist[v] + dist[edge]
                    path[w] = v
        # backtrack
        v = sorted(V)[-1]
        editSeq = []
        while True:
            try:
                w = path[v]
            except KeyError:
                break
            edit = edits[(w, v)]
            if edit[0] != "noop":
                editSeq.append((edit[1], edit[2], edit[3], edit[4]))
            v = w
        return editSeq

    def merge_graph(self, V1, V2, E1, E2, dist1, dist2, edits1, edits2):
        # vertices
        V = deepcopy(V1)
        for v in V2:
            if v not in V:
                V.append(v)
        V = sorted(V)

        # edges
        E = E1
        for e in E2:
            if e not in V:
                E.append(e)
        E = sorted(E)

        # distances
        dist = deepcopy(dist1)
        for k in list(dist2.keys()):
            if k not in list(dist.keys()):
                dist[k] = dist2[k]
            else:
                if dist[k] != dist2[k]:
                    print(
                        "WARNING: merge_graph: distance does not match!",
                        file=sys.stderr,
                    )
                    dist[k] = min(dist[k], dist2[k])

        # edit contents
        edits = deepcopy(edits1)
        for e in list(edits2.keys()):
            if e not in list(edits.keys()):
                edits[e] = edits2[e]
            else:
                if edits[e] != edits2[e]:
                    print("WARNING: merge_graph: edit does not match!", file=sys.stderr)
        return (V, E, dist, edits)

    def matchSeq(self, editSeq, gold_edits):
        m = []
        goldSeq = deepcopy(gold_edits)
        last_index = 0
        for e in reversed(editSeq):
            for i in range(last_index, len(goldSeq)):
                g = goldSeq[i]
                if self.matchEdit(e, g):
                    m.append(e)
                    last_index = i + 1
        return m

    def matchEdit(self, e, g):
        # start offset
        if e[0] != g[0]:
            return False
        # end offset
        if e[1] != g[1]:
            return False
        # original string
        if e[2] != g[2]:
            return False
        # correction string
        if not e[3] in g[3]:
            return False
        # all matches
        return True
