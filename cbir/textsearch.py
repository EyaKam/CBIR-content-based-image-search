"""Keyword search over image tags, modelled on Elasticsearch's `match` and `fuzzy` queries.

This reproduces what the reference project does with Elasticsearch on its `tags` field
(a `text` field, standard analyzer, default BM25), without needing an Elasticsearch server:

* analysis  : lower-case, split on non-word characters (approximation of the standard analyzer)
* `match`   : analysed query, OR of the terms by default (AND on request), BM25 ranking
* `fuzzy`   : the query is NOT analysed (one term, case-sensitive, exactly like ES term-level
              queries), edit distance "AUTO" (0 for 1-2 chars, 1 for 3-5, 2 for 6+), adjacent
              transpositions count as one edit, at most 50 expanded terms, each weighted by its
              similarity to the query term

BM25 follows Lucene (k1 = 1.2, b = 0.75, idf = ln(1 + (N - n + 0.5) / (n + 0.5)), document length
quantised like Lucene's norm byte). Scores should rank like Elasticsearch's; they are not
guaranteed to be bit-identical.
"""
import math
import re
from collections import defaultdict

import numpy as np

K1 = 1.2
B = 0.75
MAX_EXPANSIONS = 50
MAX_TRACK_TOTAL = 10000          # ES `track_total_hits` default

_TOKEN = re.compile(r"\w+(?:['’.]\w+)*", re.UNICODE)


def analyze(text):
    """Lower-cased tokens, like ES's standard analyzer (no stop words)."""
    return [t.lower() for t in _TOKEN.findall(text or "")]


# ---- Lucene's SmallFloat length-norm quantisation (lengths < 24 are exact) ----------------
def _long_to_int4(i):
    nbits = i.bit_length()
    if nbits < 4:
        return i
    shift = nbits - 4
    return ((i >> shift) & 0x07) | ((shift + 1) << 3)


def _int4_to_long(i):
    bits = i & 0x07
    shift = (i >> 3) - 1
    return bits if shift == -1 else (bits | 0x08) << shift


_FREE = 255 - _long_to_int4(2 ** 31 - 1)


def quantize_length(n):
    if n < _FREE:
        return n
    return _FREE + _int4_to_long(_long_to_int4(n - _FREE))


# ---- edit distance -------------------------------------------------------------------------
def edit_distance(a, b, max_d):
    """Damerau-Levenshtein (adjacent transposition = 1 edit); returns max_d + 1 if above max_d."""
    la, lb = len(a), len(b)
    if abs(la - lb) > max_d:
        return max_d + 1
    prev2 = None
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        best = cur[0]
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            v = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if prev2 is not None and i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                v = min(v, prev2[j - 2] + 1)
            cur[j] = v
            best = min(best, v)
        if best > max_d:
            return max_d + 1
        prev2, prev = prev, cur
    return prev[lb]


def auto_fuzziness(term):
    n = len(term)
    return 0 if n < 3 else 1 if n < 6 else 2


class TextIndex:
    """Inverted index over one text field; documents are identified by their position."""

    def __init__(self, docs):
        self.n = len(docs)
        self.postings = defaultdict(dict)            # term -> {doc: tf}
        lengths = np.zeros(self.n)
        for i, text in enumerate(docs):
            toks = analyze(text)
            lengths[i] = len(toks)
            for t in toks:
                self.postings[t][i] = self.postings[t].get(i, 0) + 1
        has_text = lengths > 0
        self.doc_count = int(has_text.sum())
        self.avgdl = float(lengths[has_text].mean()) if self.doc_count else 1.0
        dl = np.array([quantize_length(int(x)) for x in lengths], dtype=np.float64)
        self.norm = K1 * (1.0 - B + B * dl / self.avgdl)
        self.vocab = list(self.postings)

    # ------------------------------------------------------------------ scoring
    def _idf(self, df):
        return math.log(1.0 + (self.doc_count - df + 0.5) / (df + 0.5))

    def _add_term(self, scores, term, boost=1.0, df=None):
        post = self.postings.get(term)
        if not post:
            return
        idf = self._idf(len(post) if df is None else df)
        for doc, tf in post.items():
            scores[doc] = scores.get(doc, 0.0) + boost * idf * tf / (tf + self.norm[doc])

    @staticmethod
    def _rank(scores):
        """-> (ids, scores) sorted by score desc, then by document position."""
        if not scores:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
        ids = np.fromiter(scores.keys(), dtype=np.int64, count=len(scores))
        sc = np.fromiter(scores.values(), dtype=np.float64, count=len(scores))
        order = np.lexsort((ids, -sc))
        return ids[order], sc[order]

    # ------------------------------------------------------------------ queries
    def match(self, query, operator="or"):
        terms = analyze(query)
        if not terms:
            return self._rank({})
        if operator == "and":
            if any(t not in self.postings for t in set(terms)):
                return self._rank({})
            docs = set.intersection(*(set(self.postings[t]) for t in set(terms)))
            scores = {}
            for t in terms:
                self._add_term(scores, t)
            return self._rank({d: s for d, s in scores.items() if d in docs})
        scores = {}
        for t in terms:
            self._add_term(scores, t)
        return self._rank(scores)

    def fuzzy_terms(self, term):
        """Vocabulary terms within the AUTO edit distance, as [(boost, term)] (best 50)."""
        max_d = auto_fuzziness(term)
        found = []
        for w in self.vocab:
            if abs(len(w) - len(term)) > max_d:
                continue
            d = edit_distance(term, w, max_d)
            if d > max_d:
                continue
            boost = 1.0 if d == 0 else 1.0 - d / min(len(w), len(term))
            if boost > 0:
                found.append((boost, w))
        found.sort(key=lambda x: (-x[0], x[1]))
        return found[:MAX_EXPANSIONS]

    def fuzzy(self, term):
        if not term:
            return self._rank({})
        cands = self.fuzzy_terms(term)
        if not cands:
            return self._rank({})
        df_blend = max(len(self.postings[w]) for _, w in cands)   # blended doc frequencies
        scores = {}
        for boost, w in cands:
            self._add_term(scores, w, boost=boost, df=df_blend)
        return self._rank(scores)

    def search(self, query, kind="match", operator="or"):
        if kind == "fuzzy":
            return self.fuzzy(query)
        return self.match(query, operator)
