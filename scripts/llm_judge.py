import os
import json
from langchain_gigachat.chat_models import GigaChat
from dotenv import load_dotenv
load_dotenv()

# Клиент создаётся один раз при импорте модуля
client = GigaChat(
    credentials=os.getenv("GIGACHAT_CREDENTIALS"),  # ключ из переменной окружения
    verify_ssl_certs=False,
    model="GigaChat-2-Pro",
)

PROMPT_TEMPLATE = """Ты — эксперт по проверке эквивалентности задач.
Оригинал: {original}
Мутация: {mutated}

Вопрос: Отвечает ли мутация на тот же вопрос, что и оригинал?
Если да, то правильный ответ на мутацию должен совпадать с правильным ответом на оригинал.
Если мутация меняет смысл (убрано условие, изменено число, дата, сущность, отрицание) — она НЕ эквивалентна.
Обрати внимание: если мутация меняет направление сравнения (старше/младше, больше/меньше) или отрицание (да/нет), правильный ответ меняется, поэтому это НЕ эквивалентная мутация.

Ответь строго в формате JSON:
{{"verdict": "equivalent" | "rejected", "reason": "одно предложение"}}"""


def check_equivalence_llm(original: str, mutated: str) -> dict:
    prompt = PROMPT_TEMPLATE.format(original=original, mutated=mutated)

    try:
        response = client.invoke(prompt)
        content = response.content.strip()

        # Очистка markdown-обёртки, если модель её добавила
        if content.startswith("```"):
            content = content.strip("`")
            if content.startswith("json"):
                content = content[4:].strip()

        result = json.loads(content)

        if result.get("verdict") not in ("equivalent", "rejected"):
            raise ValueError(f"Некорректный verdict: {result.get('verdict')}")

        result["method"] = "llm_gigachat"
        return result

    except json.JSONDecodeError as e:
        return {
            "verdict": "rejected",
            "reason": f"JSON parse error: {str(e)}; raw: {content[:100]}",
            "method": "llm_gigachat_error",
        }
    except Exception as e:
        return {
            "verdict": "rejected",
            "reason": f"LLM call error: {str(e)}",
            "method": "llm_gigachat_error",
        }