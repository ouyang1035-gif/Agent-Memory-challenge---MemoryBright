# -*- coding: utf-8 -*-
import sqlite3

conn = sqlite3.connect("memory.db")
conn.row_factory = sqlite3.Row
rows = conn.execute(
    "SELECT type, substr(content,1,28) c, superseded, round(importance,2) imp, "
    "is_correction FROM memories ORDER BY created_at"
).fetchall()
for r in rows:
    print(f"type={r['type']:11s} sup={r['superseded']} imp={r['imp']} corr={r['is_correction']} | {r['c']}")

cols = [d[1] for d in conn.execute("PRAGMA table_info(memories)").fetchall()]
print("\nCOLUMNS:", cols)
