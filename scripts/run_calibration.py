import json
from llm_judge import check_equivalence_llm
from sklearn.metrics import cohen_kappa_score

# Загружаем ручные мутации
with open("data/calibration/manual_mutations.jsonl") as f:
    mutations = [json.loads(line) for line in f]

results = []
for m in mutations:
    auto = check_equivalence_llm(m["original"], m["mutated"])
    results.append({
        "task_id": m["task_id"],
        "manual_verdict": m["manual_verdict"],
        "auto_verdict": auto["verdict"],
        "method": auto.get("method"),
        "match": m["manual_verdict"] == auto["verdict"],
    })

# Считаем согласие
#matches = sum(r["match"] for r in results)
#total = len(results)
#print(f"Согласие: {matches}/{total} = {matches/total:.1%}")
manual = [r["manual_verdict"] for r in results]
auto = [r["auto_verdict"] for r in results]
kappa = cohen_kappa_score(manual, auto)
print(f"Cohen's kappa: {kappa:.3f}")

# Выводим расхождения
for r in results:
    if not r["match"]:
        print(f"РАСХОЖДЕНИЕ: {r['task_id']} manual={r['manual_verdict']} auto={r['auto_verdict']} method={r['method']}")