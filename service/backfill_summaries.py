"""v4 P0：离线回填摘要记忆层（kind='summary'）。

只写 SQLite（vectors=None），不启动服务、不碰 Qdrant，可对任意库重复执行。
用法（在 v4 目录下）：
  SQLITE_DB_PATH=/workspace/amc_eval/eval_cand_store/memory.db \
  ENABLE_SUMMARY_LAYER=true SUMMARY_MAX_CHARS=800 python3 backfill_summaries.py
"""
from __future__ import annotations

import logging

import config
import store
import summarizer


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if not config.ENABLE_SUMMARY_LAYER:
        print("ENABLE_SUMMARY_LAYER=false，拒绝回填")
        return 1

    print(f"db={config.SQLITE_DB_PATH} max_chars={config.SUMMARY_MAX_CHARS}")
    sessions = store.list_sessions()
    done = 0
    empty = 0
    for user_id, session_id in sessions:
        digest = summarizer.rebuild_session_summary(user_id, session_id)
        if digest is None:
            empty += 1
        else:
            done += 1
    print(f"sessions={len(sessions)} written={done} skipped={empty}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
