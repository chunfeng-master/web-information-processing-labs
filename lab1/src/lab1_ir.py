from __future__ import annotations

import json
import math
import re
import struct
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import jieba


TOKEN_RE = re.compile(r"[\u4e00-\u9fff]+|[a-z]+(?:'[a-z]+)?|\d+(?:\.\d+)?", re.I)
DEFAULT_STOPWORDS = {
    "的", "了", "和", "是", "在", "我", "也", "有", "就", "都", "而", "及", "与", "着",
    "或", "一个", "没有", "我们", "你", "你们", "他们", "她们", "它们", "这", "那", "很",
    "还", "被", "把", "给", "到", "从", "为", "对", "中", "上", "下", "吗", "呢", "吧",
    "啊", "呀", "哦", "嗯", "之", "其", "以及", "并", "并且", "但是", "如果", "因为",
    "所以", "然后", "可以", "能够", "进行", "这个", "那个", "自己", "什么", "怎么", "如何",
}


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def normalize_text(text: str) -> str:
    """NFKC, lower-case Latin, normalize whitespace; punctuation is removed at token stage."""
    text = unicodedata.normalize("NFKC", text or "").lower()
    return re.sub(r"\s+", " ", text).strip()


def clean_tokens(tokens: Iterable[str], remove_stopwords: bool, stopwords=DEFAULT_STOPWORDS) -> list[str]:
    out: list[str] = []
    for raw in tokens:
        token = normalize_text(raw).strip()
        if not token or not TOKEN_RE.fullmatch(token):
            continue
        if token.isdigit():
            token = "<num>"
        if remove_stopwords and token in stopwords:
            continue
        out.append(token)
    return out


class BidirectionalMaximumMatcher:
    """Dictionary-based BM tokenizer (FMM + RMM), as taught in lecture 2."""

    def __init__(self, words: Iterable[str], max_len: int = 8):
        self.words = {w.strip().lower() for w in words if w.strip()}
        self.max_len = max_len

    @classmethod
    def from_jieba_dictionary(cls, max_len: int = 8) -> "BidirectionalMaximumMatcher":
        jieba.initialize()
        return cls(jieba.dt.FREQ.keys(), max_len=max_len)

    def _fmm(self, text: str) -> list[str]:
        result, i = [], 0
        while i < len(text):
            if not text[i].isalnum() and not ('\u4e00' <= text[i] <= '\u9fff'):
                i += 1
                continue
            if text[i].isascii() and text[i].isalnum():
                m = re.match(r"[a-z0-9.]+", text[i:], re.I)
                result.append(m.group(0)); i += len(m.group(0)); continue
            length = min(self.max_len, len(text) - i)
            while length > 1 and text[i:i+length] not in self.words:
                length -= 1
            result.append(text[i:i+length]); i += length
        return result

    def _rmm(self, text: str) -> list[str]:
        result, i = [], len(text)
        while i > 0:
            if not text[i-1].isalnum() and not ('\u4e00' <= text[i-1] <= '\u9fff'):
                i -= 1
                continue
            if text[i-1].isascii() and text[i-1].isalnum():
                j = i - 1
                while j > 0 and text[j-1].isascii() and (text[j-1].isalnum() or text[j-1] == '.'):
                    j -= 1
                result.append(text[j:i]); i = j; continue
            length = min(self.max_len, i)
            while length > 1 and text[i-length:i] not in self.words:
                length -= 1
            result.append(text[i-length:i]); i -= length
        return result[::-1]

    def cut(self, text: str) -> list[str]:
        text = normalize_text(text)
        fwd, rev = self._fmm(text), self._rmm(text)
        if len(fwd) != len(rev):
            return fwd if len(fwd) < len(rev) else rev
        f_single = sum(len(x) == 1 for x in fwd)
        r_single = sum(len(x) == 1 for x in rev)
        return fwd if f_single <= r_single else rev


def statistical_cut(text: str) -> list[str]:
    return list(jieba.cut(normalize_text(text), HMM=True))


@dataclass
class Posting:
    doc_id: int
    tf: int
    positions: list[int]


class InvertedIndex:
    def __init__(self):
        self.postings: dict[str, list[Posting]] = defaultdict(list)
        self.doc_lengths: dict[int, int] = {}
        self.doc_count = 0

    def add(self, doc_id: int, tokens: Sequence[str]) -> None:
        positions: dict[str, list[int]] = defaultdict(list)
        for pos, token in enumerate(tokens):
            positions[token].append(pos)
        for token, pos_list in positions.items():
            self.postings[token].append(Posting(doc_id, len(pos_list), pos_list))
        self.doc_lengths[doc_id] = len(tokens)
        self.doc_count += 1

    def finalize(self) -> None:
        for ps in self.postings.values():
            ps.sort(key=lambda p: p.doc_id)

    def ids(self, term: str) -> list[int]:
        return [p.doc_id for p in self.postings.get(term, [])]

    def phrase(self, terms: Sequence[str]) -> list[int]:
        if not terms:
            return []
        candidates = set(self.ids(terms[0]))
        by_term = []
        for term in terms:
            mapping = {p.doc_id: set(p.positions) for p in self.postings.get(term, [])}
            candidates &= mapping.keys(); by_term.append(mapping)
        answer = []
        for doc_id in sorted(candidates):
            starts = by_term[0][doc_id]
            if any(all(start + offset in by_term[offset][doc_id] for offset in range(1, len(terms))) for start in starts):
                answer.append(doc_id)
        return answer


def intersect(a: Sequence[int], b: Sequence[int]) -> tuple[list[int], int]:
    i = j = comparisons = 0; out = []
    while i < len(a) and j < len(b):
        comparisons += 1
        if a[i] == b[j]: out.append(a[i]); i += 1; j += 1
        elif a[i] < b[j]: i += 1
        else: j += 1
    return out, comparisons


def intersect_with_skips(a: Sequence[int], b: Sequence[int], step_a: int, step_b: int) -> tuple[list[int], dict]:
    i = j = comparisons = jumps = 0; out = []
    while i < len(a) and j < len(b):
        comparisons += 1
        if a[i] == b[j]: out.append(a[i]); i += 1; j += 1
        elif a[i] < b[j]:
            nxt = i + step_a
            if step_a > 1 and nxt < len(a) and a[nxt] <= b[j]: i = nxt; jumps += 1
            else: i += 1
        else:
            nxt = j + step_b
            if step_b > 1 and nxt < len(b) and b[nxt] <= a[i]: j = nxt; jumps += 1
            else: j += 1
    return out, {"comparisons": comparisons, "jumps": jumps}


def union(a: Sequence[int], b: Sequence[int]) -> list[int]:
    return sorted(set(a) | set(b))


def difference(a: Sequence[int], b: Sequence[int]) -> list[int]:
    return sorted(set(a) - set(b))


def gap_encode(ids: Sequence[int]) -> list[int]:
    prev = 0; gaps = []
    for value in ids: gaps.append(value - prev); prev = value
    return gaps


def gap_decode(gaps: Sequence[int]) -> list[int]:
    value = 0; out = []
    for gap in gaps: value += gap; out.append(value)
    return out


def vb_encode_number(number: int) -> bytes:
    if number < 0: raise ValueError("VB only supports non-negative integers")
    parts = [number & 0x7F]; number >>= 7
    while number: parts.append(number & 0x7F); number >>= 7
    parts.reverse(); parts[-1] |= 0x80
    return bytes(parts)


def vb_encode(numbers: Iterable[int]) -> bytes:
    return b"".join(vb_encode_number(n) for n in numbers)


def vb_decode(data: bytes) -> list[int]:
    out, value = [], 0
    for byte in data:
        value = 128 * value + (byte & 0x7F)
        if byte & 0x80: out.append(value); value = 0
    if value: raise ValueError("truncated VB stream")
    return out


def compress_dictionary_block_front(terms: Sequence[str], block_size: int = 8) -> bytes:
    """Blocking plus front coding. Each block stores its first term and suffixes."""
    ordered = sorted(terms); out = bytearray()
    for start in range(0, len(ordered), block_size):
        block = ordered[start:start+block_size]
        first = block[0].encode("utf-8")
        out.extend(struct.pack(">H", len(first))); out.extend(first)
        previous = block[0]
        for term in block[1:]:
            prefix = 0
            while prefix < min(len(previous), len(term)) and previous[prefix] == term[prefix]: prefix += 1
            suffix = term[prefix:].encode("utf-8")
            out.extend(struct.pack(">HH", prefix, len(suffix))); out.extend(suffix)
            previous = term
    return bytes(out)


def baseline_index_bytes(index: InvertedIndex) -> int:
    dictionary = sum(len(t.encode("utf-8")) + 1 + 8 for t in index.postings)
    postings = sum(8 + 4 + 4 * len(p.positions) for ps in index.postings.values() for p in ps)
    return dictionary + postings


def compressed_index_bytes(index: InvertedIndex, block_size: int = 8) -> tuple[int, int, int]:
    dictionary = len(compress_dictionary_block_front(list(index.postings), block_size))
    posting_bytes = 0
    for ps in index.postings.values():
        posting_bytes += len(vb_encode(gap_encode([p.doc_id for p in ps])))
        posting_bytes += len(vb_encode(p.tf for p in ps))
        for p in ps: posting_bytes += len(vb_encode(gap_encode(p.positions)))
    return dictionary + posting_bytes, dictionary, posting_bytes


def cosine_tfidf_search(index: InvertedIndex, query_tokens: Sequence[str], top_k: int = 10) -> list[tuple[int, float]]:
    qtf = Counter(query_tokens); qweights = {}; scores = defaultdict(float); norms = defaultdict(float)
    for term, count in qtf.items():
        ps = index.postings.get(term, [])
        if not ps: continue
        idf = math.log((index.doc_count + 1) / (len(ps) + 1)) + 1
        qweights[term] = (1 + math.log(count)) * idf
        for p in ps:
            weight = (1 + math.log(p.tf)) * idf
            scores[p.doc_id] += qweights[term] * weight
    for term in qweights:
        ps = index.postings.get(term, []); idf = math.log((index.doc_count + 1) / (len(ps) + 1)) + 1
        for p in ps: norms[p.doc_id] += ((1 + math.log(p.tf)) * idf) ** 2
    qnorm = math.sqrt(sum(v*v for v in qweights.values())) or 1.0
    ranked = [(doc, score / (qnorm * math.sqrt(norms[doc]))) for doc, score in scores.items() if norms[doc]]
    return sorted(ranked, key=lambda x: (-x[1], x[0]))[:top_k]


def timed(fn, repeats: int = 5):
    values, result = [], None
    for _ in range(repeats):
        start = time.perf_counter_ns(); result = fn(); values.append(time.perf_counter_ns() - start)
    return result, sum(values) / len(values) / 1e6
