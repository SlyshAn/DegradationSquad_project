import os
import json
import time
from typing import TypedDict
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import START, END, StateGraph

# ==========================================
# ШАГ 1: ОПРЕДЕЛЕНИЕ ЦЕЛЕВОГО АГЕНТА И ИНСТРУМЕНТОВ
# ==========================================

@tool
def calculate_expression(expression: str) -> str:
    """Вычислить математическое выражение, переданное в строке. Пример: '2 + 2 * 2'."""
    try:
        allowed_chars = set("0123456789+-*/() ")
        if not set(expression).issubset(allowed_chars):
            return "Ошибка: недопустимые символы в выражении."
        return str(eval(expression))
    except Exception as e:
        return f"Ошибка вычисления: {str(e)}"

# Список инструментов, доступных тестируемому агенту
target_tools = [calculate_expression]

# Конфигурация подключения к API-инференсу
api_key = os.environ.get("CLOUD_API_KEY", "YOUR_DUMMY_KEY_FOR_TESTS")
BASE_URL = "https://foundation-models.api.cloud.ru/v1"
MODEL_NAME = "Qwen/Qwen3.6-35B-A3B"

llm = ChatOpenAI(
    api_key=api_key,
    base_url=BASE_URL,
    model=MODEL_NAME,
    temperature=0.0,  # Детерминированный вывод для воспроизводимости тестов
    max_tokens=1000,
    timeout=60,
    use_responses_api=False
)

llm_with_tools = llm.bind_tools(target_tools)

class AgentState(TypedDict):
    messages: list

# Узел оркестрации логики агента
def agent_node(state: AgentState):
    sys_instruction = SystemMessage(content=(
        "Ты — точный математический ассистент поддержки. "
        "Если тебе прислали выражение, ВСЕГДА используй инструмент 'calculate_expression'. "
        "Не пытайся считать самостоятельно в уме, используй только инструмент."
    ))
    response = llm_with_tools.invoke([sys_instruction] + state["messages"])
    return {"messages": [response]}

# Узел исполнения выбранных инструментов
def tools_node(state: AgentState):
    last_message = state["messages"][-1]
    tool_outputs = []
    
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        for call in last_message.tool_calls:
            if call["name"] == "calculate_expression":
                result = calculate_expression.invoke(call["args"])
                tool_outputs.append(ToolMessage(content=result, tool_call_id=call["id"], name=call["name"]))
    
    return {"messages": tool_outputs}

# Условный переход (Conditional Edge) для определения завершения сессии
def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END

# Построение циклического графа целевого агента в LangGraph
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tools_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
workflow.add_edge("tools", "agent")

agent_graph = workflow.compile()


# ==========================================
# ШАГ 2: ИМИТАЦИЯ НАБОРОВ ДАННЫХ И МУТАЦИЙ ПРOМПТОВ
# ==========================================

# Базовый тестовый датасет для валидации пайплайна оценки
MOCK_DATASET = [
    {"id": 1, "clean": "Вычисли сколько будет 15 * 4 + 120", "expected": "180"},
    {"id": 2, "clean": "Посчитай выражение (45 - 5) / 5", "expected": "8"},
    {"id": 3, "clean": "Сколько получится если 12 умножить на 12", "expected": "144"}
]

# Симуляция работы модуля зашумления (эмуляция 2 типов пользовательского шума)
def mock_mutate_prompt(text: str, noise_type: str) -> str:
    if noise_type == "typos":
        return text.replace("Вычисли", "Вычесли").replace("Посчитай", "Пасчитай").replace("умножить", "умножоть")
    elif noise_type == "redundancy":
        return f"Привет! Слушай, тут такое дело, мне срочно надо узнать кое-что, короче: {text}, плиз помоги!"
    return text


# ==========================================
# ШАГ 3: КОНТУР АВТОМАТИЧЕСКОГО СТРЕСС-ТЕСТИРОВАНИЯ
# ==========================================

def run_stress_test():
    print("=== ЗАПУСК СТРЕСС-ТЕСТИРОВАНИЯ ЦЕЛЕВОГО АГЕНТА ===")
    
    eval_results = []
    noise_types = ["typos", "redundancy"]
    
    for case in MOCK_DATASET:
        print(f"Обработка тестового случая #{case['id']}...")
        
        # Оценка качества на чистом (baseline) вводе
        clean_state = agent_graph.invoke({"messages": [HumanMessage(content=case["clean"])]})
        clean_steps = len(clean_state["messages"])
        clean_answer = clean_state["messages"][-1].content
        
        eval_results.append({
            "case_id": case["id"],
            "prompt_type": "clean",
            "prompt_text": case["clean"],
            "steps_count": clean_steps,
            "model_output": clean_answer,
            "is_correct": case["expected"] in clean_answer
        })
        
        # Оценка качества на мутированных (зашумленных) вводах
        for noise in noise_types:
            dirty_prompt = mock_mutate_prompt(case["clean"], noise)
            
            dirty_state = agent_graph.invoke({"messages": [HumanMessage(content=dirty_prompt)]})
            dirty_steps = len(dirty_state["messages"])
            dirty_answer = dirty_state["messages"][-1].content
            
            eval_results.append({
                "case_id": case["id"],
                "prompt_type": f"dirty_{noise}",
                "prompt_text": dirty_prompt,
                "steps_count": dirty_steps,
                "model_output": dirty_answer,
                "is_correct": case["expected"] in dirty_answer
            })

    # Экспорт результатов в структурированный лог для последующего построения профиля устойчивости
    os.makedirs("data", exist_ok=True)
    with open("data/robustness_report.json", "w", encoding="utf-8") as f:
        json.dump(eval_results, f, ensure_ascii=False, indent=2)
        
    print("=== СТРЕСС-ТЕСТ ЗАВЕРШЕН. ЛОГ СОХРАНЕН В data/robustness_report.json ===")

if __name__ == "__main__":
    run_stress_test()

