"""多路召回的倒数排名融合（Reciprocal Rank Fusion, RRF）。

依据：
- Cormack, Clarke & Buettcher (SIGIR 2009) 提出的 RRF：
  score(d) = Σ_r 1 / (k + rank_r(d))，k 通常取 60。它只看"在每一路里排第几"，
  不需要对齐各路分数量纲，也不需要调权重，在多路异构检索里稳定优于手工加权。
- 2026《Training-Free Lexical–Dense Fusion for Conversational-Memory
  Retrieval》在会话记忆检索任务上复验：lexical(BM25) + dense 的 rank 融合
  优于手工加权求和。

v2 用的是 0.5*cosine + 0.5*bm25_norm 的手工加权：两路量纲完全不同，
且把 BM25 的 top1 归一到 1.0，会让"只是撞了几个字面词"的弱匹配候选虚高。
这里改为按排名融合。
"""

from __future__ import annotations


def rrf_fuse(rank_lists: list[list[str]], k: int = 60) -> dict[str, float]:
    """多路排名列表 -> RRF 分数。rank_lists 中每个子列表按相关性从高到低排列。"""
    scores: dict[str, float] = {}
    kk = max(int(k), 1)
    for ranks in rank_lists:
        for rank, mid in enumerate(ranks, start=1):
            if not mid:
                continue
            scores[mid] = scores.get(mid, 0.0) + 1.0 / (kk + rank)
    return scores


def normalize_max(scores: dict[str, float]) -> dict[str, float]:
    """按最大值归一到 (0, 1]，保序；空输入返回空 dict。"""
    if not scores:
        return {}
    hi = max(scores.values())
    if hi <= 0:
        return {mid: 0.0 for mid in scores}
    return {mid: s / hi for mid, s in scores.items()}
