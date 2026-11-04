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
        idx = random.randint(0, len(words) - 1)
        words[idx] = mutate_word(words[idx])
    elif intensity == "medium":
        num_mutations = max(1, int(len(words) * 0.15))
        indices = random.sample(range(len(words)), min(num_mutations, len(words)))
        for i in indices:
            words[i] = mutate_word(words[i])
    elif intensity == "hard":
        num_mutations = max(1, int(len(words) * 0.35))
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
    for task in tasks[:10]:
        task_id = task["task_id"]
        original_text = task["input"]
        
        # Генерируем легкую мутацию (1 опечатка)
        mutations.append({
            "mutation_id": f"m_{task_id}_typo_easy",
            "task_id": task_id,
            "mutation_type": "typo",
            "intensity": "easy",
            "mutated_input": apply_typo(original_text, "easy")
        })
        
        # Генерируем сложную мутацию (около 35% слов с ошибками)
        mutations.append({
            "mutation_id": f"m_{task_id}_typo_hard",
            "task_id": task_id,
            "mutation_type": "typo",
            "intensity": "hard",
            "mutated_input": apply_typo(original_text, "hard")
        })

    # Сохраняем результат в файл mutations.jsonl
    print(f"Сохраняю результаты в {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        for m in mutations:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
            
    print(f"Готово! Успешно сгенерировано {len(mutations)} мутаций.")

# --- 3. ТОЧКА ВХОДА ---
# Эта конструкция говорит Python: "если этот файл запущен напрямую, 
# то выполни функцию main()"
if __name__ == "__main__":
    main()