"""从压缩 RDF 流中按电影抽样，生成课程框架读取的三个集合。"""
import gzip
import hashlib
import heapq
import json
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 2025
MOVIES = 1600
NS = rb"<http://rdf\.freebase\.com/ns/"
MID = rb"([mg]\.[A-Za-z0-9_]+)>"
FILM_EDGE = re.compile(b"^" + NS + MID + rb"\t" + NS + rb"(film\.film\.[^>]+)>\t" + NS + MID + rb"\t\.[\r]?$", re.M)


def sample_movies(path):
    # 保留哈希值最小的电影；阈值只会下降，最终入选电影的早期边也被完整保留。
    selected, heap = {}, []
    with gzip.open(path, "rb") as stream:
        while chunk := stream.read(8 << 20):
            chunk += stream.readline()
            for match in FILM_EDGE.finditer(chunk):
                head, relation, tail = match.groups()
                if head not in selected:
                    digest = hashlib.blake2b(head, key=str(SEED).encode(), digest_size=16).digest()
                    priority = int.from_bytes(digest, "big")
                    if len(heap) == MOVIES:
                        if priority >= -heap[0][0]:
                            continue
                        del selected[heapq.heappop(heap)[1]]
                    heapq.heappush(heap, (-priority, head))
                    selected[head] = set()
                selected[head].add((relation, tail))
    return sorted((h.decode(), r.decode(), t.decode())
                  for h, edges in selected.items() for r, t in edges)


def main():
    triples = sample_movies(ROOT / "data/raw/freebase_douban.gz")
    entities = sorted({v for h, _, t in triples for v in (h, t)})
    relations = sorted({r for _, r, _ in triples})
    if len(entities) < 3000:
        raise ValueError("电影子图实体不足 3000")
    output = ROOT / "data/freebase"
    output.mkdir(parents=True, exist_ok=True)
    metadata = ROOT / "data/metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    for filename, values in [("entities", entities), ("relations", relations)]:
        (metadata / f"{filename}.txt").write_text(
            "".join(f"{i} {v}\n" for i, v in enumerate(values)), encoding="utf-8")
    eid = {v: i for i, v in enumerate(entities)}
    rid = {v: i for i, v in enumerate(relations)}
    rows = [(eid[h], rid[r], eid[t]) for h, r, t in triples]
    random.Random(SEED).shuffle(rows)
    # 仅在训练集仍覆盖每个实体和关系时留出该边。
    counts = Counter(v for h, _, t in rows for v in {h, t})
    rel_counts = Counter(r for _, r, _ in rows)
    train, held = [], []
    per_split = len(rows) // 10
    for h, r, t in rows:
        endpoints = {h, t}
        if len(held) < 2 * per_split and min(counts[v] for v in endpoints) > 1 and rel_counts[r] > 1:
            held.append((h, r, t))
            counts.subtract(endpoints)
            rel_counts[r] -= 1
        else:
            train.append((h, r, t))
    if len(held) != 2 * per_split:
        raise ValueError("无法在保持训练集覆盖率时完成划分")
    random.Random(SEED + 1).shuffle(held)
    splits = dict(train=train, valid=held[:per_split], test=held[per_split:])
    for name, split in splits.items():
        (output / f"{name}.txt").write_text(
            "".join(f"{h} {r} {t}\n" for h, r, t in split), encoding="utf-8")
    stats = dict(seed=SEED, films=len({h for h, _, _ in triples}), entities=len(eid),
                 relations=len(rid), triples=len(rows), splits={k: len(v) for k, v in splits.items()})
    print(json.dumps(stats))
    (metadata / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
