from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import sqlite3
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.lab1_ir import (  # noqa: E402
    BidirectionalMaximumMatcher, DEFAULT_STOPWORDS, clean_tokens,
    compress_dictionary_block_front, difference, gap_decode, gap_encode,
    intersect, intersect_with_skips, normalize_text, read_jsonl,
    statistical_cut, union, vb_decode, vb_encode,
)

DATA = ROOT / "Qilin-IR-50K"
OUT = ROOT / "outputs"
SEED = 2026

RESULTS_GUIDE = {
    "purpose": "实验1全部必做项的机器可读结果；本对象置于文件开头，便于助教理解。",
    "how_to_read": [
        "index：语料规模、两类分词统计及规范化样例。",
        "boolean：复杂布尔查询、执行顺序、跳表步长和位置短语实验。",
        "compression：词典压缩、倒排表压缩和压缩前后端到端查询性能。",
        "ranked：TF-IDF、文档表征、停用词对照、Top10重合率和点击代理指标。",
        "timing_ms 均为毫秒；bytes 均为字节；ratio 为压缩后/压缩前，越小越省空间。",
    ],
    "important_limits": [
        "点击只能作为相关性代理，未点击不等于不相关。",
        "稠密语义模型在完整50K库的TF-IDF Top200候选上重排，属于两阶段检索。",
        "预分词版本严格逐词编码，再对词向量取均值；并非把空格分词文本整体编码。",
    ],
}


def connect_index(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT)")
    con.execute("CREATE TABLE IF NOT EXISTS docs(doc_id INTEGER PRIMARY KEY, title TEXT, content TEXT, length INTEGER)")
    con.execute("CREATE TABLE IF NOT EXISTS postings(term TEXT, doc_id INTEGER, tf INTEGER, positions BLOB, PRIMARY KEY(term,doc_id)) WITHOUT ROWID")
    con.execute("CREATE INDEX IF NOT EXISTS postings_doc ON postings(doc_id)")
    return con


def build_index(max_docs: int | None) -> tuple[Path, dict]:
    OUT.mkdir(exist_ok=True)
    suffix = f"_{max_docs}" if max_docs else "_full"
    db_path = OUT / f"inverted_index{suffix}.sqlite"
    tokens_path = OUT / f"normalized_tokens{suffix}.jsonl"
    if db_path.exists() and tokens_path.exists():
        with sqlite3.connect(db_path) as con:
            count = int(con.execute("SELECT value FROM meta WHERE key='doc_count'").fetchone()[0])
            vocab = con.execute("SELECT COUNT(DISTINCT term) FROM postings").fetchone()[0]
            token_count = int(con.execute("SELECT value FROM meta WHERE key='token_count'").fetchone()[0])
        results_path = OUT / "results.json"
        if results_path.exists():
            prior = json.loads(results_path.read_text(encoding="utf-8")).get("index", {})
            if prior.get("doc_count") == count:
                prior["reused"] = True
                return db_path, prior
        return db_path, {"doc_count": count, "vocabulary_size": vocab, "token_count": token_count,
                         "method_comparison": {"bm_tokens": 0, "hmm_tokens": 0, "same_boundaries": 0, "sample_docs": 0},
                         "samples": [], "reused": True}

    if db_path.exists(): db_path.unlink()
    matcher = BidirectionalMaximumMatcher.from_jieba_dictionary()
    con = connect_index(db_path)
    samples, method_stats = [], {"bm_tokens": 0, "hmm_tokens": 0, "same_boundaries": 0, "sample_docs": 0}
    start = time.perf_counter(); doc_count = 0; token_count = 0
    with tokens_path.open("w", encoding="utf-8") as token_file:
        for row in read_jsonl(DATA / "documents.jsonl"):
            if max_docs and doc_count >= max_docs: break
            doc_id = int(row["doc_id"]); title = row.get("title") or ""; content = row.get("content") or ""
            text = f"{title} {content}"  # title and body are merged; title appears once.
            hmm_raw = statistical_cut(text)
            keep = clean_tokens(hmm_raw, False); removed = clean_tokens(hmm_raw, True)
            positions = defaultdict(list)
            for pos, token in enumerate(removed): positions[token].append(pos)
            con.execute("INSERT INTO docs VALUES(?,?,?,?)", (doc_id, title, content, len(removed)))
            con.executemany("INSERT INTO postings VALUES(?,?,?,?)", [
                (term, doc_id, len(pos), vb_encode(gap_encode(pos))) for term, pos in positions.items()
            ])
            token_file.write(json.dumps({"doc_id": doc_id, "keep": keep, "remove": removed}, ensure_ascii=False) + "\n")
            if doc_count < 200:
                bm = clean_tokens(matcher.cut(text), False)
                method_stats["bm_tokens"] += len(bm); method_stats["hmm_tokens"] += len(keep)
                method_stats["same_boundaries"] += int(bm == keep); method_stats["sample_docs"] += 1
                if len(samples) < 8 and bm != keep:
                    samples.append({"doc_id": doc_id, "text": text[:120], "BM": bm[:30], "HMM": keep[:30]})
            doc_count += 1; token_count += len(removed)
            if doc_count % 1000 == 0: con.commit(); print(f"indexed {doc_count:,} documents", flush=True)
    con.executemany("INSERT OR REPLACE INTO meta VALUES(?,?)", [
        ("doc_count", str(doc_count)), ("token_count", str(token_count)),
        ("tokenizer", "jieba HMM"), ("stopwords", "built-in documented list"),
    ])
    con.commit()
    vocab = con.execute("SELECT COUNT(DISTINCT term) FROM postings").fetchone()[0]
    con.close()
    stats = {"doc_count": doc_count, "vocabulary_size": vocab, "token_count": token_count,
             "seconds": time.perf_counter() - start, "method_comparison": method_stats, "samples": samples}
    return db_path, stats


def build_matching_token_file(max_docs: int | None) -> tuple[Path, dict]:
    """Retain BM results for every document so both required tokenizers are reproducible."""
    suffix = f"_{max_docs}" if max_docs else "_full"
    path = OUT / f"matching_tokens{suffix}.jsonl"
    if path.exists():
        return path, {"reused": True, "bytes": path.stat().st_size}
    matcher = BidirectionalMaximumMatcher.from_jieba_dictionary(); count = tokens = 0
    start = time.perf_counter()
    with path.open("w", encoding="utf-8") as f:
        for row in read_jsonl(DATA / "documents.jsonl"):
            if max_docs and count >= max_docs: break
            text = f"{row.get('title') or ''} {row.get('content') or ''}"
            keep = clean_tokens(matcher.cut(text), False); removed = clean_tokens(keep, True)
            f.write(json.dumps({"doc_id": int(row["doc_id"]), "keep": keep, "remove": removed}, ensure_ascii=False)+"\n")
            count += 1; tokens += len(removed)
            if count % 5000 == 0: print(f"BM tokenized {count:,} documents", flush=True)
    return path, {"doc_count": count, "token_count": tokens, "seconds": time.perf_counter()-start, "bytes": path.stat().st_size}


class SQLiteIndex:
    def __init__(self, path: Path):
        self.con = sqlite3.connect(path)
        self.doc_count = int(self.con.execute("SELECT value FROM meta WHERE key='doc_count'").fetchone()[0])

    def postings(self, term: str) -> list[tuple[int, int, list[int]]]:
        return [(doc, tf, gap_decode(vb_decode(pos))) for doc, tf, pos in self.con.execute(
            "SELECT doc_id,tf,positions FROM postings WHERE term=? ORDER BY doc_id", (term,))]

    def ids(self, term: str) -> list[int]: return [x[0] for x in self.postings(term)]
    def df(self, term: str) -> int: return self.con.execute("SELECT COUNT(*) FROM postings WHERE term=?", (term,)).fetchone()[0]
    def document(self, doc_id: int) -> dict:
        row = self.con.execute("SELECT title,content FROM docs WHERE doc_id=?", (doc_id,)).fetchone()
        return {"doc_id": doc_id, "title": row[0], "content": row[1]}

    def phrase(self, terms: list[str]) -> list[int]:
        maps = [{doc: set(pos) for doc, _, pos in self.postings(t)} for t in terms]
        if not maps: return []
        candidates = set(maps[0])
        for m in maps[1:]: candidates &= m.keys()
        return [d for d in sorted(candidates) if any(all(p+i in maps[i][d] for i in range(1,len(terms))) for p in maps[0][d])]

    def tfidf(self, query: str, top_k=10):
        terms = clean_tokens(statistical_cut(query), True); qtf = Counter(terms)
        scores, norms, qweights = defaultdict(float), defaultdict(float), {}
        for term, count in qtf.items():
            ps = self.postings(term)
            if not ps: continue
            idf = math.log((self.doc_count + 1) / (len(ps) + 1)) + 1
            qweights[term] = (1 + math.log(count)) * idf
            for doc, tf, _ in ps:
                w = (1 + math.log(tf)) * idf; scores[doc] += qweights[term] * w; norms[doc] += w*w
        qnorm = math.sqrt(sum(x*x for x in qweights.values())) or 1
        return sorted(((d, s/(qnorm*math.sqrt(norms[d]))) for d,s in scores.items()), key=lambda x:(-x[1],x[0]))[:top_k]


def build_compressed_index(idx: SQLiteIndex, path: Path) -> dict:
    """Persist gap+VB compressed docIDs, tf values and positional gaps per term."""
    if path.exists(): path.unlink()
    con=sqlite3.connect(path)
    con.execute("CREATE TABLE postings(term TEXT PRIMARY KEY, df INTEGER, doc_gaps BLOB, tfs BLOB, pos_counts BLOB, position_gaps BLOB) WITHOUT ROWID")
    start=time.perf_counter(); terms=0
    for (term,) in idx.con.execute("SELECT DISTINCT term FROM postings ORDER BY term"):
        ps=idx.postings(term); doc_ids=[d for d,_,_ in ps]
        counts=[]; all_pos=[]
        for _,_,pos in ps: counts.append(len(pos)); all_pos.extend(gap_encode(pos))
        con.execute("INSERT INTO postings VALUES(?,?,?,?,?,?)", (
            term,len(ps),vb_encode(gap_encode(doc_ids)),vb_encode(tf for _,tf,_ in ps),
            vb_encode(counts),vb_encode(all_pos)))
        terms+=1
        if terms%10000==0: con.commit()
    con.commit(); con.execute("VACUUM"); con.close()
    return {"path":path.name,"bytes":path.stat().st_size,"terms":terms,"build_seconds":time.perf_counter()-start}


class CompressedSQLiteIndex:
    def __init__(self,path:Path): self.con=sqlite3.connect(path)
    def ids(self,term:str)->list[int]:
        row=self.con.execute("SELECT doc_gaps FROM postings WHERE term=?",(term,)).fetchone()
        return [] if row is None else gap_decode(vb_decode(row[0]))


def boolean_experiments(idx: SQLiteIndex) -> dict:
    expressions = [
        {"expression": "(旅游 OR 攻略) AND (北京 OR 上海)", "groups": [["旅游","攻略"],["北京","上海"]]},
        {"expression": "(减肥 OR 健身) AND (饮食 OR 运动)", "groups": [["减肥","健身"],["饮食","运动"]]},
        {"expression": "(大学 OR 考研) AND (专业 OR 复试)", "groups": [["大学","考研"],["专业","复试"]]},
    ]
    results = []
    for item in expressions:
        lists = [union(idx.ids(a), idx.ids(b)) for a,b in item["groups"]]
        records = []
        for order in ([0,1],[1,0]):
            times=[]; comparisons=0; answer=[]
            for _ in range(30):
                st=time.perf_counter_ns(); answer, comps=intersect(lists[order[0]],lists[order[1]]); times.append(time.perf_counter_ns()-st); comparisons=comps
            records.append({"order": order, "result_count":len(answer), "comparisons":comparisons, "mean_ms":statistics.mean(times)/1e6})
        results.append({**item,"df_groups":[len(x) for x in lists],"orders":records})

    # A three-way AND makes order effects more visible.
    triple_terms = ["旅游", "攻略", "酒店"]
    term_lists = {t: idx.ids(t) for t in triple_terms}; order_runs=[]
    for order in (triple_terms, list(reversed(triple_terms)), sorted(triple_terms,key=lambda t:len(term_lists[t]))):
        cur=term_lists[order[0]]; comps=0; st=time.perf_counter_ns()
        for t in order[1:]: cur,c=intersect(cur,term_lists[t]); comps+=c
        order_runs.append({"order":order,"result_count":len(cur),"comparisons":comps,"elapsed_ms":(time.perf_counter_ns()-st)/1e6})

    a,b=idx.ids("旅游"),idx.ids("攻略"); skip=[]
    for step in [1,2,4,8,16,32,max(1,int(math.sqrt(max(len(a),1))))]:
        times=[]; meta={}
        for _ in range(50):
            st=time.perf_counter_ns(); ans,meta=intersect_with_skips(a,b,step,step); times.append(time.perf_counter_ns()-st)
        pointer_count=(len(a)//step+len(b)//step) if step>1 else 0
        skip.append({"step":step,"result_count":len(ans),**meta,"pointer_count":pointer_count,"mean_ms":statistics.mean(times)/1e6})
    phrase_candidates=[["北京","旅游"],["旅游","攻略"],["减肥","方法"],["大学","专业"],["酒店","推荐"],["电子","登机牌"]]
    phrase_options=[]
    for terms in phrase_candidates:
        positional=idx.phrase(terms); plain=idx.ids(terms[0])
        for t in terms[1:]: plain,_=intersect(plain,idx.ids(t))
        phrase_options.append((len(plain)-len(positional),terms,plain,positional))
    _,phrase_terms,plain,phrase=max(phrase_options,key=lambda x:(x[0],len(x[2])))
    excluded=next((d for d in plain if d not in set(phrase)),None)
    excluded_doc=idx.document(excluded) if excluded is not None else None
    return {"complex_queries":results,"three_way_order":order_runs,"skip_steps":skip,
            "phrase":{"query":"".join(phrase_terms),"terms":phrase_terms,"positional_count":len(phrase),"and_count":len(plain),
                      "sample_doc_ids":phrase[:10],"excluded_example":excluded_doc,
                      "explanation":"AND只要求同文档共现；位置检索要求相邻且顺序一致。"}}


def compression_experiment(idx: SQLiteIndex, compressed_path: Path) -> dict:
    con=idx.con
    terms=[r[0] for r in con.execute("SELECT DISTINCT term FROM postings ORDER BY term")]
    baseline_dictionary=sum(len(t.encode('utf-8'))+1+8 for t in terms)
    dictionary_results=[]
    for block in [2,4,8,16,32]:
        st=time.perf_counter_ns(); data=compress_dictionary_block_front(terms,block); elapsed=(time.perf_counter_ns()-st)/1e6
        dictionary_results.append({"block_size":block,"bytes":len(data),"ratio":len(data)/baseline_dictionary,"encode_ms":elapsed})
    baseline_postings=compressed_postings=posting_count=0
    st=time.perf_counter_ns()
    for term in terms:
        ps=idx.postings(term); posting_count+=len(ps)
        baseline_postings += sum(8+4+4*len(pos) for _,_,pos in ps)
        compressed_postings += len(vb_encode(gap_encode([d for d,_,_ in ps])))
        compressed_postings += len(vb_encode(tf for _,tf,_ in ps))
        compressed_postings += sum(len(vb_encode(gap_encode(pos))) for _,_,pos in ps)
    encode_ms=(time.perf_counter_ns()-st)/1e6
    # Query-time comparison includes decoding compressed docIDs.
    terms_for_query=[t for t in ["旅游","攻略","大学","减肥"] if idx.df(t)]
    query_perf=[]
    for term in terms_for_query:
        ids=idx.ids(term); packed=vb_encode(gap_encode(ids)); raw=bytes().join(int(x).to_bytes(8,'big') for x in ids)
        raw_times=[]; packed_times=[]
        for _ in range(100):
            s=time.perf_counter_ns(); [int.from_bytes(raw[i:i+8],'big') for i in range(0,len(raw),8)]; raw_times.append(time.perf_counter_ns()-s)
            s=time.perf_counter_ns(); gap_decode(vb_decode(packed)); packed_times.append(time.perf_counter_ns()-s)
        query_perf.append({"term":term,"df":len(ids),"raw_bytes":len(raw),"compressed_bytes":len(packed),
                           "raw_decode_ms":statistics.mean(raw_times)/1e6,"compressed_decode_ms":statistics.mean(packed_times)/1e6})
    build_meta=build_compressed_index(idx,compressed_path); cidx=CompressedSQLiteIndex(compressed_path)
    end_to_end=[]
    query_terms=[["旅游","攻略"],["减肥","运动"],["大学","复试"],["北京","酒店"]]
    for terms_pair in query_terms:
        raw_times=[]; compressed_times=[]; raw_result=compressed_result=[]
        for _ in range(50):
            s=time.perf_counter_ns(); raw_result,_=intersect(idx.ids(terms_pair[0]),idx.ids(terms_pair[1])); raw_times.append(time.perf_counter_ns()-s)
            s=time.perf_counter_ns(); compressed_result,_=intersect(cidx.ids(terms_pair[0]),cidx.ids(terms_pair[1])); compressed_times.append(time.perf_counter_ns()-s)
        if raw_result!=compressed_result: raise AssertionError(f"compressed mismatch: {terms_pair}")
        end_to_end.append({"query":" AND ".join(terms_pair),"result_count":len(raw_result),
                           "uncompressed_mean_ms":statistics.mean(raw_times)/1e6,
                           "compressed_mean_ms":statistics.mean(compressed_times)/1e6,
                           "same_results":True})
    return {"baseline_dictionary_bytes":baseline_dictionary,"dictionary_block_front":dictionary_results,
            "baseline_postings_bytes":baseline_postings,"gap_vb_postings_bytes":compressed_postings,
            "postings_ratio":compressed_postings/baseline_postings,"posting_count":posting_count,"encode_ms":encode_ms,
            "compressed_index_file":build_meta,"query_decode":query_perf,"end_to_end_query":end_to_end}


def ranked_experiments(idx: SQLiteIndex, skip_semantic: bool) -> dict:
    queries=["毛巾变黄","酒酿好处","收藏完整登机牌"]
    tfidf={}
    for q in queries:
        ranked=idx.tfidf(q,10)
        tfidf[q]=[{"rank":i+1,"doc_id":d,"score":round(s,6),"title":idx.document(d)["title"]} for i,(d,s) in enumerate(ranked)]
    result={"tfidf":tfidf}
    if skip_semantic: result["semantic"]={"skipped":True}; return result
    from sentence_transformers import SentenceTransformer
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    model=SentenceTransformer(model_name)
    semantic={}
    query_rows={r["query_id"]:r for r in read_jsonl(DATA/"queries.jsonl")}
    query_ids_by_text={r["query"]:qid for qid,r in query_rows.items()}
    clicked=defaultdict(set)
    for row in read_jsonl(DATA/"search_logs.jsonl"):
        if row.get("click")==1: clicked[row["query_id"]].add(int(row["doc_id"]))

    def normalize_rows(matrix):
        norm=np.linalg.norm(matrix,axis=1,keepdims=True); norm[norm==0]=1
        return matrix/norm

    def mean_token_embeddings(token_lists):
        """Encode each distinct segmented term separately, then mean-pool per text."""
        vocabulary=sorted({token for tokens in token_lists for token in tokens})
        vectors=model.encode(vocabulary,batch_size=256,normalize_embeddings=True,show_progress_bar=False)
        lookup={token:vectors[i] for i,token in enumerate(vocabulary)}
        dim=vectors.shape[1]; pooled=[]
        for tokens in token_lists:
            pooled.append(np.mean([lookup[t] for t in tokens],axis=0) if tokens else np.zeros(dim,dtype=np.float32))
        return normalize_rows(np.asarray(pooled,dtype=np.float32)),len(vocabulary)

    def proxy_metrics(items,relevant):
        if not relevant: return {"available":False,"note":"该自拟查询没有对应点击标签"}
        ranked_ids=[x["doc_id"] for x in items]
        hits=[i+1 for i,d in enumerate(ranked_ids) if d in relevant]
        return {"available":True,"clicked_docs":len(relevant),"recall_at_10":len(hits)/len(relevant),
                "mrr_at_10":0 if not hits else 1/min(hits),
                "note":"点击是相关性代理；未点击不等于不相关"}

    for q in queries:
        candidates=[d for d,_ in idx.tfidf(q,200)]
        docs=[idx.document(d) for d in candidates]
        raw_text=[normalize_text(f"{d['title']} {d['content']}")[:1800] for d in docs]
        segmented_keep=[clean_tokens(statistical_cut(t),False) for t in raw_text]
        segmented_remove=[[x for x in tokens if x not in DEFAULT_STOPWORDS] for tokens in segmented_keep]
        q_raw=model.encode([q],normalize_embeddings=True,show_progress_bar=False)[0]
        direct_embs=model.encode(raw_text,batch_size=32,normalize_embeddings=True,show_progress_bar=False)
        keep_embs,keep_vocab=mean_token_embeddings(segmented_keep+[clean_tokens(statistical_cut(q),False)])
        remove_embs,remove_vocab=mean_token_embeddings(segmented_remove+[clean_tokens(statistical_cut(q),True)])
        variants={"direct_no_preseg":(direct_embs,q_raw),
                  "preseg_keep_stopwords_token_mean":(keep_embs[:-1],keep_embs[-1]),
                  "preseg_remove_stopwords_token_mean":(remove_embs[:-1],remove_embs[-1])}
        ranked_variants={}
        for name,(embs,qv) in variants.items():
            order=np.argsort(-(embs@qv))[:10]
            ranked_variants[name]=[{"rank":i+1,"doc_id":candidates[j],"score":round(float(embs[j]@qv),6),"title":docs[j]["title"]} for i,j in enumerate(order)]
        def overlap(a,b): return len({x['doc_id'] for x in a}&{x['doc_id'] for x in b})/10
        qid=query_ids_by_text.get(q); relevant=clicked.get(qid,set()) if qid else set()
        tfidf_items=tfidf[q]
        ranked_variants["comparisons"]={
            "direct_vs_preseg_overlap_at_10":overlap(ranked_variants["direct_no_preseg"],ranked_variants["preseg_keep_stopwords_token_mean"]),
            "keep_vs_remove_stopwords_overlap_at_10":overlap(ranked_variants["preseg_keep_stopwords_token_mean"],ranked_variants["preseg_remove_stopwords_token_mean"]),
            "tfidf_vs_direct_semantic_overlap_at_10":overlap(tfidf_items,ranked_variants["direct_no_preseg"]),
            "candidate_scope":"TF-IDF top 200 from the full corpus",
            "preseg_definition":"每个HMM词项单独输入同一模型，L2归一化后对词向量取算术平均，再归一化文本向量",
            "unique_terms_encoded_keep":keep_vocab,"unique_terms_encoded_remove":remove_vocab,
            "click_proxy":{"tfidf":proxy_metrics(tfidf_items,relevant),
                           "direct_semantic":proxy_metrics(ranked_variants["direct_no_preseg"],relevant),
                           "preseg_keep":proxy_metrics(ranked_variants["preseg_keep_stopwords_token_mean"],relevant),
                           "preseg_remove":proxy_metrics(ranked_variants["preseg_remove_stopwords_token_mean"],relevant)},
        }
        semantic[q]=ranked_variants
    result["semantic"]={"model":model_name,"aggregation":"mean of separately encoded term embeddings", "queries":semantic}; return result


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--max-docs",type=int); parser.add_argument("--skip-semantic",action="store_true")
    args=parser.parse_args(); random.seed(SEED); np.random.seed(SEED); OUT.mkdir(exist_ok=True)
    db_path,stats=build_index(args.max_docs); idx=SQLiteIndex(db_path)
    matching_path,matching_stats=build_matching_token_file(args.max_docs); stats["matching_full_corpus"]=matching_stats
    suffix=f"_{args.max_docs}" if args.max_docs else "_full"
    compressed_path=OUT/f"compressed_index{suffix}.sqlite"
    boolean=boolean_experiments(idx); compression=compression_experiment(idx,compressed_path); ranked=ranked_experiments(idx,args.skip_semantic)
    results={"_README":RESULTS_GUIDE,"seed":SEED,"index":stats,"boolean":boolean,"compression":compression,"ranked":ranked}
    (OUT/"results.json").write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT/"README.md").write_text("""# 实验结果文件说明

本目录只包含实验生成物，不包含原始数据。助教建议先看 `results.json` 顶部的 `_README`，再结合实验报告阅读。

- `results.json`：全部实验的机器可读结果、查询案例、耗时、压缩率和检索结果。
- `inverted_index_full.sqlite`：未压缩位置倒排索引，含 `docs`、`postings`、`meta` 三张表。
- `compressed_index_full.sqlite`：持久化压缩索引。docID 使用 gap+VB，tf、位置计数和位置间距使用 VB。
- `normalized_tokens_full.jsonl`：HMM 分词的保留/去除停用词两套结果，一行一篇文档，可直接用文本编辑器查看。
- `matching_tokens_full.jsonl`：双向最大匹配分词的保留/去除停用词两套结果，一行一篇文档，可直接用文本编辑器查看。

计时单位为毫秒，空间单位为字节。点击指标只作为相关性代理，不能把未点击文档直接视为不相关。
""",encoding="utf-8")
    print(f"done: {OUT/'results.json'}")


if __name__ == "__main__": main()
