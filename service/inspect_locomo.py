# -*- coding: utf-8 -*-
import pandas as pd

df = pd.read_parquet("locomo.parquet")
r = df.iloc[0]
print("=== context (first 800 chars) ===")
print(repr(r["context"][:800]))
print("\n=== qa (first 800 chars) ===")
print(repr(r["qa"][:800]))
print("\n=== speakers ===", r["speaker_a"], "/", r["speaker_b"])
print("=== num_questions ===", r["num_questions"])
