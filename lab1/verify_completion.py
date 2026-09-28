import json
import sqlite3
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"outputs"
required=[
    OUT/"inverted_index_full.sqlite", OUT/"compressed_index_full.sqlite",
    OUT/"normalized_tokens_full.jsonl", OUT/"matching_tokens_full.jsonl",
    OUT/"results.json", OUT/"README.md", ROOT/"实验报告.md",
]
missing=[str(x) for x in required if not x.exists()]
assert not missing, f"缺少文件: {missing}"
d=json.loads((OUT/"results.json").read_text(encoding="utf-8"))
assert "_README" in d
assert d["index"]["doc_count"]==50000
assert d["index"]["matching_full_corpus"]["doc_count"]==50000
assert d["boolean"]["phrase"]["and_count"]>d["boolean"]["phrase"]["positional_count"]
assert all(x["same_results"] for x in d["compression"]["end_to_end_query"])
assert d["ranked"]["semantic"]["aggregation"]=="mean of separately encoded term embeddings"
with sqlite3.connect(OUT/"inverted_index_full.sqlite") as con:
    assert con.execute("SELECT COUNT(*) FROM docs").fetchone()[0]==50000
with sqlite3.connect(OUT/"compressed_index_full.sqlite") as con:
    assert con.execute("SELECT COUNT(*) FROM postings").fetchone()[0]==d["index"]["vocabulary_size"]
print("PASS：实验1全部必做结果文件与关键一致性检查通过")
