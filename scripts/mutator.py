import os
from langchain_gigachat.chat_models import GigaChat

# Инициализация клиента (ключ должен быть в переменной окружения)
llm = GigaChat(
    credentials=os.getenv("GIGACHAT_CREDENTIALS"),
    verify_ssl_certs=False,
    model="GigaChat-2",
    temperature=0.7 # 0.7 чтобы модель креативила и делала разные мутации
)
from dotenv import load_dotenv
load_dotenv()
import json
import random
import re
import os

# --- 1. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ (СТАНКИ) ---

def mutate_word(word):
    if len(word) < 3:
        return word
    typo_type = random.choice(['swap', 'delete', 'duplicate', 'case'])
    match = re.match(r"^([a-zA-Zа-яА-ЯёЁ]+)(.*)$", word)
    if not match:
        return word
    core_word = match.group(1)
    punctuation = match.group(2)
    
    if typo_type == 'swap' and len(core_word) > 2:
        idx = random.randint(0, len(core_word) - 2)
        mutated_core = core_word[:idx] + core_word[idx+1] + core_word[idx] + core_word[idx+2:]
    elif typo_type == 'delete':
        idx = random.randint(0, len(core_word) - 1)
        mutated_core = core_word[:idx] + core_word[idx+1:]
    elif typo_type == 'duplicate':
        idx = random.randint(0, len(core_word) - 1)
        mutated_core = core_word[:idx] + core_word[idx] + core_word[idx] + core_word[idx+1:]
    elif typo_type == 'case':
        idx = random.randint(0, len(core_word) - 1)
        mutated_core = core_word[:idx] + core_word[idx].swapcase() + core_word[idx+1:]
    else:
        mutated_core = core_word
    return mutated_core + punctuation

def apply_typo(text, intensity="easy"):
    words = text.split()
    if not words:
        return text
    if intensity == "easy":
        valid_indices = [i for i, w in enumerate(words) if len(w) > 3]
        if valid_indices:
            idx = random.choice(valid_indices)
            words[idx] = mutate_word(words[idx])
    elif intensity == "medium":
        valid_indices = [i for i, w in enumerate(words) if len(w) > 3]
        if valid_indices:
            idx = random.choice(valid_indices)
            words[idx] = mutate_word(words[idx])
    elif intensity == "hard":
        num_mutations = max(1, int(len(words) * 0.20))
        indices = random.sample(range(len(words)), min(num_mutations, len(words)))
        for i in indices:
            words[i] = mutate_word(words[i])
    return " ".join(words)


# --- 2. ГЛАВНАЯ ФУНКЦИЯ (НАЧАЛЬНИК ЦЕХА) ---

def main():
    # Пути к файлам
    input_file = "data/clean/v0.1_enriched.jsonl"
    output_file = "data/mutated/mutations.jsonl"
    
    # Создаем папку data/mutated, если её еще нет
    os.makedirs("data/mutated", exist_ok=True)
    
    mutations = []
    
    print("Читаю датасет...")
    # Читаем наш чистый датасет
    with open(input_file, "r", encoding="utf-8") as f:
        tasks = [json.loads(line) for line in f]
        
    print(f"Найдено задач: {len(tasks)}. Генерирую мутации для первых 10...")
    
    # Берем первые 10 задач
        # Берем первые 5 задач, чтобы не тратить слишком много токенов GigaChat на тесты
    for task in tasks[:5]:
        task_id = task["task_id"]
        original_text = task["input"]
        
        # 1. Генерируем мутацию типа typo (опечатки)
        mutations.append({
            "mutation_id": f"m_{task_id}_typo_01",
            "task_id": task_id,
            "mutation_type": "typo",
            "intensity": "medium",
            "mutated_input": apply_typo(original_text, "medium")
        })
        
        # 2. Генерируем мутацию типа paraphrase (через LLM)
        print(f"Генерирую paraphrase для {task_id}...")
        mutations.append({
            "mutation_id": f"m_{task_id}_para_01",
            "task_id": task_id,
            "mutation_type": "paraphrase",
            "intensity": "easy",
            "mutated_input": apply_paraphrase(original_text)
        })
        
        # 3. Генерируем мутацию типа missing_condition (через LLM)
        print(f"Генерирую missing_condition для {task_id}...")
        mutations.append({
            "mutation_id": f"m_{task_id}_miss_01",
            "task_id": task_id,
            "mutation_type": "missing_condition",
            "intensity": "medium",
            "mutated_input": apply_missing_condition(original_text)
        })

    # Сохраняем результат в файл mutations.jsonl
    print(f"Сохраняю результаты в {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        for m in mutations:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
            
    print(f"Готово! Успешно сгенерировано {len(mutations)} мутаций.")

def apply_paraphrase(text):
    prompt = f"""Перефразируй следующий вопрос, используя другие слова. 
    КРИТИЧЕСКОЕ ПРАВИЛО 1: НЕ ПЕРЕВОДИ ТЕКСТ. Если исходный текст на английском, ответ ДОЛЖЕН БЫТЬ НА АНГЛИЙСКОМ ЯЗЫКЕ.
    КРИТИЧЕСКОЕ ПРАВИЛО 2: НЕ меняй имена, названия, даты и числа. 
    Верни ТОЛЬКО перефразированный текст, без твоих комментариев и кавычек.
    
    Вопрос: {text}"""
    
    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        print(f"Ошибка GigaChat (paraphrase): {e}")
        return text

def apply_missing_condition(text):
    prompt = f"""Возьми следующий вопрос и удали из него ОДНО важное уточнение (например, год, город, название фильма или конкретное место). 
    КРИТИЧЕСКОЕ ПРАВИЛО 1: НЕ пиши никаких комментариев в скобках.
    КРИТИЧЕСКОЕ ПРАВИЛО 2: НЕ заменяй оставшиеся слова на другие синонимы. Просто удали часть текста.
    Вопрос должен остаться грамматически правильным.
    Верни ТОЛЬКО измененный текст.
    
    Вопрос: {text}"""
    
    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        print(f"Ошибка GigaChat (missing_condition): {e}")
        return text   
# --- 3. ТОЧКА ВХОДА ---
# Эта конструкция говорит Python: "если этот файл запущен напрямую, 
# то выполни функцию main()"
if __name__ == "__main__":
    main()