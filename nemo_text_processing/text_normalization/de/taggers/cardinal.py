# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict

from nemo_text_processing.text_normalization.de.utils import get_abs_path, load_labels
from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_DIGIT,
    NEMO_SIGMA,
    GraphFst,
    delete_space,
    insert_space,
)

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False

AND = "und"


def get_ties_digit(digit_path: str, tie_path: str) -> 'pynini.FstLike':
    """
    getting all inverse normalizations for numbers between 21 - 100

    Args:
        digit_path: file to digit tsv
        tie_path: file to tie tsv, e.g. 20, 30, etc.
    Returns:
        res: fst that converts numbers to their verbalization
    """

    digits = defaultdict(list)
    ties = defaultdict(list)
    for k, v in load_labels(digit_path):
        digits[v].append(k)
    digits["1"] = ["ein"]

    for k, v in load_labels(tie_path):
        ties[v].append(k)

    d = []
    for i in range(21, 100):
        s = str(i)
        if s[1] == "0":
            continue

        for di in digits[s[1]]:
            for ti in ties[s[0]]:
                word = di + f" {AND} " + ti
                d.append((word, s))

    res = pynini.string_map(d)
    return res


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals, e.g. 
        "101" ->  cardinal { integer: "ein hundert und zehn" }

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = False):
        super().__init__(name="cardinal", kind="classify", deterministic=deterministic)

        graph_zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv")).invert()
        graph_digit_no_one = pynini.string_file(get_abs_path("data/numbers/digit.tsv")).invert()
        graph_one = pynini.string_file(get_abs_path("data/numbers/ones.tsv")).invert()
        graph_digit = graph_digit_no_one | graph_one
        self.digit = (graph_digit | graph_zero).optimize()
        graph_teen = pynini.string_file(get_abs_path("data/numbers/teen.tsv")).invert()

        graph_ties = pynini.string_file(get_abs_path("data/numbers/ties.tsv")).invert()
        # separator = "."

        def tens_no_zero():
            return (
                pynutil.delete("0") + graph_digit
                | get_ties_digit(
                    get_abs_path("data/numbers/digit.tsv"), get_abs_path("data/numbers/ties.tsv")
                ).invert()
                | graph_teen
                | (graph_ties + pynutil.delete("0"))
            )

        def hundred_non_zero():
            return (graph_digit_no_one + insert_space | pynini.cross("1", "ein ")) + pynutil.insert("hundert") + (
                pynini.closure(insert_space + pynutil.insert(AND), 0, 1) + insert_space + tens_no_zero()
                | pynutil.delete("00")
            ) | pynutil.delete("0") + tens_no_zero()

        def thousand():
            return (hundred_non_zero() + insert_space + pynutil.insert("tausend") | pynutil.delete("000")) + (
                insert_space + hundred_non_zero() | pynutil.delete("000")
            )

        graph_million = pynini.union(
            hundred_non_zero() + insert_space + pynutil.insert("million") + pynini.closure(pynutil.insert("en"), 0, 1),
            pynutil.delete("000"),
        )

        graph_billion = pynini.union(
            hundred_non_zero()
            + insert_space
            + pynutil.insert("milliarde")
            + pynini.closure(pynutil.insert("n"), 0, 1),
            pynutil.delete("000"),
        )

        graph_trillion = pynini.union(
            hundred_non_zero() + insert_space + pynutil.insert("billion") + pynini.closure(pynutil.insert("en"), 0, 1),
            pynutil.delete("000"),
        )

        graph_quadrillion = pynini.union(
            hundred_non_zero()
            + insert_space
            + pynutil.insert("billiarde")
            + pynini.closure(pynutil.insert("n"), 0, 1),
            pynutil.delete("000"),
        )

        graph_quintillion = pynini.union(
            hundred_non_zero()
            + insert_space
            + pynutil.insert("trillion")
            + pynini.closure(pynutil.insert("en"), 0, 1),
            pynutil.delete("000"),
        )

        graph_sextillion = pynini.union(
            hundred_non_zero()
            + insert_space
            + pynutil.insert("trilliarde")
            + pynini.closure(pynutil.insert("n"), 0, 1),
            pynutil.delete("000"),
        )
        graph = pynini.union(
            graph_sextillion
            + insert_space
            + graph_quintillion
            + insert_space
            + graph_quadrillion
            + insert_space
            + graph_trillion
            + insert_space
            + graph_billion
            + insert_space
            + graph_million
            + insert_space
            + thousand()
        )

        self.graph = (
            ((NEMO_DIGIT - "0" + pynini.closure(NEMO_DIGIT, 0)) - "0" - "1")
            @ pynini.cdrewrite(pynini.closure(pynutil.insert("0")), "[BOS]", "", NEMO_SIGMA)
            @ NEMO_DIGIT ** 24
            @ graph
            @ pynini.cdrewrite(delete_space, "[BOS]", "", NEMO_SIGMA)
            @ pynini.cdrewrite(delete_space, "", "[EOS]", NEMO_SIGMA)
            @ pynini.cdrewrite(pynini.cross("  ", " "), "", "", NEMO_SIGMA)
        )
        self.graph |= graph_zero | pynini.cross("1", "eins")

        # self.graph = pynini.cdrewrite(pynutil.delete(separator), "", "", NEMO_SIGMA) @ self.graph
        self.graph = self.graph.optimize()

        self.graph_hundred_component_at_least_one_none_zero_digit = (
            ((NEMO_DIGIT - "0" + pynini.closure(NEMO_DIGIT, 0)) - "0" - "1")
            @ pynini.cdrewrite(pynini.closure(pynutil.insert("0")), "[BOS]", "", NEMO_SIGMA)
            @ NEMO_DIGIT ** 3
            @ hundred_non_zero()
        ) | pynini.cross("1", "eins")

        self.graph_hundred_component_at_least_one_none_zero_digit = (
            self.graph_hundred_component_at_least_one_none_zero_digit.optimize()
        )

        self.two_digit_non_zero = (
            pynini.closure(NEMO_DIGIT, 1, 2) @ self.graph_hundred_component_at_least_one_none_zero_digit
        )

        optional_minus_graph = pynini.closure(pynutil.insert("negative: ") + pynini.cross("-", "\"true\" "), 0, 1)

        final_graph = optional_minus_graph + pynutil.insert("integer: \"") + self.graph + pynutil.insert("\"")
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
