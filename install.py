# scripts/download_dataset.py
from datasets import load_dataset
import json
import os

# Создаём папку, если её нет
os.makedirs("data/clean", exist_ok=True)

# Загружаем 50 примеров HotpotQA
ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation[:50]")

# Сохраняем в JSONL
with open("data/clean/v0.1.jsonl", "w", encoding="utf-8") as f:
    for item in ds:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Сохранено {len(ds)} примеров в data/clean/v0.1.jsonl")