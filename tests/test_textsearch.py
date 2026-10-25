import math

import pytest

from cbir.textsearch import (TextIndex, analyze, auto_fuzziness, edit_distance,
                             quantize_length)

DOCS = ["sky blue sky", "blue sea", "red car", "sky mountain snow", ""]


def bm25(tf, df, n_docs, dl, avgdl, k1=1.2, b=0.75):
    """Textbook Lucene BM25, written independently of the module."""
    idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
    return idf * tf / (tf + k1 * (1 - b + b * dl / avgdl))


@pytest.fixture
def idx():
    return TextIndex(DOCS)


def test_analyzer_lowercases_and_splits():
    assert analyze("Sky, BLUE-sea; it's 3.5") == ["sky", "blue", "sea", "it's", "3.5"]


def test_match_or_scores_are_bm25(idx):
    ids, sc = idx.match("sky")
    avgdl = (3 + 2 + 2 + 3) / 4          # the empty document is not counted
    assert ids.tolist() == [0, 3]       # tf=2 beats tf=1
    assert sc[0] == pytest.approx(bm25(2, 2, 4, 3, avgdl))
    assert sc[1] == pytest.approx(bm25(1, 2, 4, 3, avgdl))


def test_match_or_adds_terms_and_and_requires_all(idx):
    ids, sc = idx.match("blue sky", "or")
    assert ids.tolist()[0] == 0 and set(ids.tolist()) == {0, 1, 3}
    assert idx.match("blue sky", "and")[0].tolist() == [0]
    assert idx.match("blue unknownword", "and")[0].tolist() == []
    assert idx.match("")[0].tolist() == []


def test_match_is_case_insensitive(idx):
    assert idx.match("SKY")[0].tolist() == [0, 3]


def test_edit_distance_with_transposition():
    assert edit_distance("blue", "bule", 2) == 1          # one transposition = one edit
    assert edit_distance("kitten", "sitting", 5) == 3
    assert edit_distance("abcdef", "uvwxyz", 2) == 3      # above the limit -> max + 1


def test_auto_fuzziness():
    assert [auto_fuzziness(w) for w in ("ab", "sky", "table", "mountain")] == [0, 1, 1, 2]


def test_fuzzy_tolerates_typos_but_is_a_single_case_sensitive_term(idx):
    assert idx.fuzzy("sky")[0].tolist() == [0, 3]
    assert idx.fuzzy("skey")[0].tolist() == [0, 3]        # 1 edit, len 4
    assert idx.fuzzy("bule")[0].tolist() == [1, 0]        # transposition; shorter doc ranks first
    assert idx.fuzzy("mountian")[0].tolist() == [3]       # len 8 -> up to 2 edits
    assert idx.fuzzy("sk")[0].tolist() == []              # len 2 -> exact only
    assert idx.fuzzy("red car")[0].tolist() == []         # not analysed: no such term
    # exact term outranks a 1-edit neighbour
    ids, sc = idx.fuzzy("Sky")                             # 'S' != 's': 1 edit, boosted by 2/3
    assert ids.tolist() == [0, 3] and sc[0] < idx.fuzzy("sky")[1][0]


def test_length_norm_quantisation_matches_lucene_small_floats():
    assert [quantize_length(n) for n in (0, 5, 23, 24)] == [0, 5, 23, 24]
    assert all(quantize_length(n) <= n for n in range(0, 500))
    assert quantize_length(100) == 96
