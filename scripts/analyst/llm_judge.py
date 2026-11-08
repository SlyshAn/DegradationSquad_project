import os
from langchain_gigachat.chat_models import GigaChat
from langchain_core.messages import SystemMessage, HumanMessage
import json

# Инициализируем GigaChat для Аналитика
llm = GigaChat(
    credentials=os.environ.get("GIGACHAT_CREDENTIALS"),
    verify_ssl_certs=False,
    model="GigaChat-2-Pro",
    temperature=0.0,
    max_tokens=100
)

def check_answer_is_correct(model_output: str, expected_output: str) -> bool:
    """Проверяет, эквивалентен ли ответ агента эталону."""
    if not model_output or not expected_output:
        return False
        
    # Быстрая проверка: если строки совпадают, сразу возвращаем True (экономия токенов)
    if model_output.strip().lower() == expected_output.strip().lower():
        return True

    prompt = f"""Оцени, является ли ответ агента правильным.
Эталонный ответ: {expected_output}
Ответ агента: {model_output}

Считается ли ответ агента правильным по смыслу? (Например, если эталон "yes", а агент написал "Да, они были одной национальности" — это правильный ответ).
Ответь строго одним словом: YES или NO."""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content.strip().upper()
        return "YES" in content
    except Exception as e:
        print(f"Ошибка LLM-судьи: {e}")
        return False
