"""离线 A/B：同一份库、同一批题，比较 纯向量 / 纯BM25 / 融合 的 hit@k。

不调 LLM、不写库、不占线上 qdrant（读的是 /tmp/ab_qdrant 拷贝）。
"""
import json, sqlite3, time, urllib.request
import numpy as np
import jieba
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

SET = "/workspace/amc_eval/expanded/sets/sub60.json"
DB = "/workspace/amc_eval/expanded_store/memory.db"
QD = "/tmp/ab_qdrant"
EMB = "http://localhost:8080/v1/embeddings"
VEC_TOP_K, BM25_TOP_K = 400, 400
WV, WB = 0.5, 0.5
KS = [1, 3, 5, 10, 20]


def embed(texts):
    body = json.dumps({"model": "bge-m3", "input": texts}).encode()
    req = urllib.request.Request(EMB, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.loads(r.read())
    return np.array([x["embedding"] for x in d["data"]], dtype="float32")


def tok(text):
    return [w.strip().lower() for w in jieba.cut(text or "") if w.strip()]


items = json.load(open(SET))["items"]
users = sorted({i["user_id"] for i in items})

con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
mem = {u: con.execute("select id, content from memories where user_id=?", (u,)).fetchall() for u in users}
print("sqlite memories", {u: len(v) for u, v in mem.items()})

cli = QdrantClient(path=QD)
vecs = {}
for u in users:
    off = None
    while True:
        pts, off = cli.scroll("memories", scroll_filter=qm.Filter(must=[qm.FieldCondition(key="user_id", match=qm.MatchValue(value=u))]),
                              limit=1000, offset=off, with_vectors=True, with_payload=False)
        for p in pts:
            vecs[p.id] = np.array(p.vector, dtype="float32")
        if off is None:
            break
print("qdrant vectors", {u: sum(1 for i, _ in mem[u] if i in vecs) for u in users})

jieba.cut("预热")
bm = {}
for u in users:
    docs = [c for _, c in mem[u]]
    bm[u] = (BM25Okapi([tok(d) for d in docs]), [i for i, _ in mem[u]])

qvec = embed([i["query"] for i in items])


def rank_lists(u, q, qv):
    ids = [i for i, _ in mem[u]]
    mat = np.stack([vecs[i] for i in ids if i in vecs])
    vids = [i for i in ids if i in vecs]
    cos = mat @ qv / (np.linalg.norm(mat, axis=1) * np.linalg.norm(qv) + 1e-9)
    order = np.argsort(-cos)[:VEC_TOP_K]
    vrank = [vids[k] for k in order]
    vscore = {vids[k]: float(cos[k]) for k in order}

    b, bids = bm[u]
    bs = b.get_scores(tok(q))
    bo = np.argsort(-bs)[:BM25_TOP_K]
    brank = [bids[k] for k in bo]
    bmax = max((bs[k] for k in bo), default=0.0) or 1.0
    bscore = {bids[k]: float(bs[k]) / bmax for k in bo}

    cand = {}
    for i in vrank:
        cand[i] = [vscore[i], None]
    for i in brank:
        if i in cand:
            cand[i][1] = bscore[i]
        else:
            cand[i] = [None, bscore[i]]
    fscore = {}
    for i, (vs, bs_) in cand.items():
        if vs is not None and bs_ is not None:
            fscore[i] = (WV * vs + WB * bs_) / (WV + WB)
        else:
            fscore[i] = vs if vs is not None else bs_
    frank = sorted(fscore, key=lambda x: -fscore[x])
    return vrank, brank, frank


rows = []
for n, it in enumerate(items):
    u = it["user_id"]
    gold = set(it["gold_ids"])
    vrank, brank, frank = rank_lists(u, it["query"], qvec[n])
    r = {"id": it["id"], "qtype": it["qtype"], "gold_kind": it.get("gold_kind"),
         "n_gold": len(gold),
         "vec_pool": len(gold & set(vrank)), "bm25_pool": len(gold & set(brank)),
         "bm25_only_pool": len((gold & set(brank)) - set(vrank)),
         "vec_only_pool": len((gold & set(vrank)) - set(brank))}
    for name, rk in (("vec", vrank), ("bm25", brank), ("fused", frank)):
        rank = next((j + 1 for j, x in enumerate(rk) if x in gold), None)
        r[name + "_rank"] = rank
        for k in KS:
            r[f"{name}@{k}"] = bool(rank and rank <= k)
    rows.append(r)

scored = [r for r in rows if r["gold_kind"] != "weak"]
print(f"\n{'arm':6s}" + "".join(f"{'h@'+str(k):>8s}" for k in KS) + "     MRR")
for name in ("vec", "bm25", "fused"):
    line = f"{name:6s}"
    for k in KS:
        line += f"{sum(r[name+'@'+str(k)] for r in scored)/len(scored)*100:7.1f}%"
    mrr = np.mean([1 / r[name + "_rank"] if r[name + "_rank"] else 0 for r in scored])
    print(line + f" {mrr:7.3f}")
print(f"\nn={len(scored)} (weak 剔除 {len(rows)-len(scored)})")
print("gold 覆盖：向量 top400 %.1f 条/题，BM25 top400 %.1f 条/题，BM25 独有 %.1f 条/题，向量独有 %.1f 条/题" % (
    np.mean([r["vec_pool"] for r in scored]), np.mean([r["bm25_pool"] for r in scored]),
    np.mean([r["bm25_only_pool"] for r in scored]), np.mean([r["vec_only_pool"] for r in scored])))
print("BM25 把 gold 带进池而向量没有的题数：", sum(1 for r in scored if r["bm25_only_pool"] > 0), "/", len(scored))
json.dump(rows, open("/tmp/ab_bm25_rows.json", "w"), ensure_ascii=False, indent=1)
