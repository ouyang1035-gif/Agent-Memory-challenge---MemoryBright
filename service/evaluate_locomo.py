# -*- coding: utf-8 -*-
"""LoCoMo 真实基准评测：解析对话 -> 灌入记忆 -> 检索 -> 评估证据命中。

评测逻辑：
- 把 context 按日期分割成多个 session，再按发言轮次解析成 messages
- Add 全部 messages 到对应 user_id
- 对每个 QA，Search 检索 top_k=10，检查返回的记忆文本是否包含金标答案关键词
- 指标：证据命中率（答案关键词是否出现在检索结果中）

注意：这是检索质量评估，不是最终答案评估（答案由平台 Answer 模型生成）。
"""
import json
import re
import time
import urllib.request

import pandas as pd

BASE = "http://127.0.0.1:8000"


def call(method, path, payload=None):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        BASE + path, data=data, method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def parse_context(context: str, speaker_a: str, speaker_b: str) -> list[dict]:
    """把 LoCoMo context 文本解析成 messages 列表。

    格式：DATE: ...\nCONVERSATION:\nSpeaker said, "..."\nSpeaker said, "..." ...
    """
    # 按 DATE 分割成多个 session
    date_pattern = r"DATE: (.+?)\nCONVERSATION:\n"
    parts = re.split(date_pattern, context)
    # parts: [prefix, date1, conv1, date2, conv2, ...]

    messages = []
    # 跳过 prefix（"Below is a conversation..."）
    for i in range(1, len(parts), 2):
        date_str = parts[i].strip()
        conv_text = parts[i + 1] if i + 1 < len(parts) else ""

        # 解析时间戳（"1:56 pm on 8 May, 2023" -> Unix ms）
        timestamp = parse_date_to_ms(date_str)

        # 按发言分割：Speaker said, "..."
        said_pattern = rf'({re.escape(speaker_a)}|{re.escape(speaker_b)}) said, "(.*?)"(?= and |\. |\n|$)'
        matches = re.findall(said_pattern, conv_text, re.DOTALL)

        for speaker, content in matches:
            role = "user" if speaker == speaker_a else "assistant"
            messages.append({
                "role": role,
                "content": content.strip(),
                "timestamp": timestamp,
            })

    return messages


def parse_date_to_ms(date_str: str) -> int:
    """把 "1:56 pm on 8 May, 2023" 转成 Unix 毫秒。"""
    from datetime import datetime
    try:
        # 提取 "8 May, 2023"
        m = re.search(r"on (.+)", date_str)
        if m:
            date_part = m.group(1).strip()
            dt = datetime.strptime(date_part, "%d %B, %Y")
            return int(dt.timestamp() * 1000)
    except Exception:
        pass
    # 兜底：返回固定时间
    return 1704067200000


def norm(t: str) -> str:
    return t.replace(" ", "").lower()


def evaluate_locomo(max_samples: int = 3, top_k: int = 10):
    df = pd.read_parquet("locomo.parquet")
    total_q = 0
    total_hit = 0
    total_top1_hit = 0
    cat_stats = {}

    for idx in range(min(max_samples, len(df))):
        row = df.iloc[idx]
        sample_id = row["sample_id"]
        user_id = f"locomo-{sample_id}"

        print(f"\n===== Sample {idx+1}/{max_samples}: {sample_id} =====")

        # 解析对话
        messages = parse_context(row["context"], row["speaker_a"], row["speaker_b"])
        print(f"解析出 {len(messages)} 条消息")

        # 分批 Add（每批 20 条，模拟平台分块）
        batch_size = 20
        for i in range(0, len(messages), batch_size):
            batch = messages[i:i+batch_size]
            s, b = call("POST", "/add", {
                "request_id": f"{user_id}-batch-{i//batch_size}",
                "messages": batch,
                "user_id": user_id,
                "session_id": f"{user_id}-sess-{i//batch_size}",
            })
            if s != 200:
                print(f"  [ERROR] add batch {i//batch_size}: {s}")
                break
        print(f"  Add 完成")

        # 解析 QA
        qa_list = json.loads(row["qa"])
        print(f"  共 {len(qa_list)} 个问题")

        # 抽样：每个类别最多测 per_cat 题（快速验证，全量太慢）
        per_cat = 5
        cat_count = {}
        sampled = []
        for q in qa_list:
            if q.get("is_adversarial"):
                continue
            cat = q["category"]
            if cat_count.get(cat, 0) < per_cat:
                sampled.append(q)
                cat_count[cat] = cat_count.get(cat, 0) + 1
        print(f"  抽样 {len(sampled)} 题（每类最多 {per_cat} 题）")

        for qi, q in enumerate(sampled):
            if q.get("is_adversarial"):
                continue  # 跳过对抗题（无答案）
            total_q += 1
            category = q["category"]
            query = q["question"]
            answer = q["answer"]

            s, b = call("POST", "/search", {
                "query": query,
                "user_id": user_id,
                "top_k": top_k,
            })
            data = b.get("data", [])
            joined = norm(" ".join(it["content"] for it in data))
            top1 = norm(data[0]["content"]) if data else ""

            # 证据命中：答案关键词是否出现在检索结果中
            # 简单策略：答案分词后检查是否都在结果中
            answer_keywords = [norm(w) for w in re.findall(r"\w+", answer) if len(w) > 2]
            hit = all(k in joined for k in answer_keywords) if answer_keywords else False
            top1_hit = any(k in top1 for k in answer_keywords) if answer_keywords else False

            total_hit += 1 if hit else 0
            total_top1_hit += 1 if top1_hit else 0

            d = cat_stats.setdefault(category, {"q": 0, "hit": 0, "top1": 0})
            d["q"] += 1
            d["hit"] += 1 if hit else 0
            d["top1"] += 1 if top1_hit else 0

            mark = "OK " if hit else "   "
            print(f"  [{mark}] cat{category} {query[:45]:47s} hit={'Y' if hit else 'N'}")

        print(f"  当前累计: {total_hit}/{total_q} 命中 ({total_hit/total_q:.1%})")

    print("\n===== 总体结果 =====")
    print(f"总问题数: {total_q}")
    print(f"证据命中率: {total_hit/total_q:.2%}")
    print(f"Top1 命中率: {total_top1_hit/total_q:.2%}")

    print("\n===== 分类别 =====")
    cat_names = {1: "single-hop", 2: "temporal", 3: "multi-hop", 4: "open-domain"}
    for cat, d in sorted(cat_stats.items()):
        name = cat_names.get(cat, f"cat-{cat}")
        print(f"{name:12s} | n={d['q']:3d} | recall={d['hit']/d['q']:.2%} | top1={d['top1']/d['q']:.2%}")


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    evaluate_locomo(max_samples=n)
