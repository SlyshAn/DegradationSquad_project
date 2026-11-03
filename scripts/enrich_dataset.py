import json

INPUT = "data/clean/v0.1.jsonl"  # сырой файл после загрузки
OUTPUT = "data/clean/v0.1_enriched.jsonl"

with open(INPUT, "r", encoding="utf-8") as f_in, open(OUTPUT, "w", encoding="utf-8") as f_out:
    for i, line in enumerate(f_in):
        entry = json.loads(line)
    
        hf_type = entry.get("type", "bridge")
        if hf_type == "bridge":
            task_type = "multi_hop"
            skills = ["retrieval", "reasoning"]
        else:
            task_type = "comparison"
            skills = ["retrieval", "comparison"]
    
        enriched = {
            "task_id": f"hp_{i:03d}",
            "source_id": entry.get("id", None),
            "input": entry["question"],
            "expected_output": entry["answer"],
            "task_type": task_type,
            "required_skills": skills,
            "verification_method": "exact_match",
            "source": "hotpotqa/hotpot_qa",
            "source_type": hf_type,
            "level": entry.get("level", "unknown"),
            "context_docs": None,
        }
        f_out.write(json.dumps(enriched, ensure_ascii=False) + "\n")

print(f"Обогащено {i+1} записей")