from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Модель, обученная на MNLI (DeBERTa-v3)
model_name = "MoritzLaurer/DeBERTa-v3-base-mnli"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Метки, которые возвращает модель
label_names = ["entailment", "neutral", "contradiction"]

def nli_score(premise: str, hypothesis: str) -> dict:
    """Возвращает вероятности entailment/neutral/contradiction."""
    inputs = tokenizer(premise, hypothesis, truncation=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits[0]
    probs = torch.softmax(logits, dim=-1).tolist()
    return {name: round(float(p) * 100, 1) for name, p in zip(label_names, probs)}

def check_equivalence_nli(original: str, mutated: str, threshold: float = 0.7) -> dict:
    """
    Проверяет эквивалентность через NLI в оба направления.
    Возвращает вердикт 'equivalent' / 'rejected' / 'borderline'.
    """
    # Направление 1: оригинал влечёт мутацию
    forward = nli_score(original, mutated)
    # Направление 2: мутация влечёт оригинал
    backward = nli_score(mutated, original)
    
    fwd_ent = forward["entailment"] / 100  # нормируем в 0..1
    bwd_ent = backward["entailment"] / 100
    
    # Оба направления уверенно entailment → эквивалентно
    if fwd_ent >= threshold and bwd_ent >= threshold:
        return {"verdict": "equivalent", "confidence": min(fwd_ent, bwd_ent), "method": "nli"}
    
    # Оба направления уверенно НЕ entailment → не эквивалентно
    if fwd_ent < 0.3 and bwd_ent < 0.3:
        return {"verdict": "rejected", "confidence": 1 - max(fwd_ent, bwd_ent), "method": "nli"}
    
    # Пограничный случай → отправляем к LLM-judge
    return {"verdict": "borderline", "forward": forward, "backward": backward, "method": "nli"}