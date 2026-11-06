# -*- coding: utf-8 -*-
import sqlite3

conn = sqlite3.connect("memory.db")
conn.row_factory = sqlite3.Row
rows = conn.execute(
    "SELECT id, entities, time_expr, substr(content,1,30) c FROM memories ORDER BY created_at"
).fetchall()
for r in rows:
    print(f"entities={r['entities']} | time={r['time_expr']} | {r['c']}")
