import os
import json
import time
from typing import TypedDict
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import START, END, StateGraph

# ==========================================
# ШАГ 1: ОПРЕДЕЛЕНИЕ ИНСТРУМЕНТОВ И АГЕНТА
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

@tool
def search_wikipedia(query: str) -> str:
    """Найти информацию в Википедии по текстовому запросу. Используется для исторических, географических и фактологических вопросов."""
    # Симуляция поиска по ключевым словам дляHotpotQA
    q = query.lower()
    if "man" in q or "human" in q:
        return "В контексте HotpotQA: Человек — субъект, часто изучаемый в дисциплинах STEM. Роджер имеет 5 теннисных мячей."
    if "apple" in q:
        return "Кафетерий имел 23 яблока. Было использовано 20 яблок для приготовления обеда."
    return f"По запросу '{query}' в Википедии найдены общие факты. Специфический контекст отсутствует."

# Набор инструментов агента включает калькулятор и поиск
target_tools = [calculate_expression, search_wikipedia]

api_key = os.environ.get("CLOUD_API_KEY", "YOUR_DUMMY_KEY_FOR_TESTS")
BASE_URL = "https://cloud.ru"
MODEL_NAME = "Qwen/Qwen3.6-35B-A3B"

# Стоимость токенов для Qwen/Qwen3.6-35B-A3B (усредненные тарифы за 1 млн токенов)
PRICE_INPUT_1M_USD = 0.40   # $0.40 за 1 млн входных токенов
PRICE_OUTPUT_1M_USD = 0.40  # $0.40 за 1 млн выходных токенов

llm = ChatOpenAI(
    api_key=api_key,
    base_url=BASE_URL,
    model=MODEL_NAME,
    temperature=0.0,
    max_tokens=1000,
    timeout=90,
    use_responses_api=False
)

llm_with_tools = llm.bind_tools(target_tools)

class AgentState(TypedDict):
    messages: list

def agent_node(state: AgentState):
    sys_instruction = SystemMessage(content=(
        "Ты — точный и вежливый ассистент поддержки. "
        "Для решения математических задач ВСЕГДА вызывaй инструмент 'calculate_expression'. "
        "Для поиска фактов, имен, дат и текстовой информации ВСЕГДА вызывай 'search_wikipedia'. "
        "Не пытайся отвечать из головы, если задача требует расчетов или поиска информации."
    ))
    response = llm_with_tools.invoke([sys_instruction] + state["messages"])
    return {"messages": [response]}

def tools_node(state: AgentState):
    last_message = state["messages"][-1]
    tool_outputs = []
    
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        for call in last_message.tool_calls:
            if call["name"] == "calculate_expression":
                result = calculate_expression.invoke(call["args"])
                tool_outputs.append(ToolMessage(content=result, tool_call_id=call["id"], name=call["name"]))
            elif call["name"] == "search_wikipedia":
                result = search_wikipedia.invoke(call["args"])
                tool_outputs.append(ToolMessage(content=result, tool_call_id=call["id"], name=call["name"]))
    
    return {"messages": tool_outputs}

def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END

workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tools_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
workflow.add_edge("tools", "agent")

agent_graph = workflow.compile()

# ==========================================
# ШАГ 2: СБОР РАСШИРЕННЫХ МЕТРИК И СТОИМОСТИ
# ==========================================

def calculate_cost(usage_metadata):
    if not usage_metadata:
        return 0.0
    input_tokens = usage_metadata.get("input_tokens", 0)
    output_tokens = usage_metadata.get("output_tokens", 0)
    
    cost_in = (input_tokens / 1_000_000) * PRICE_INPUT_1M_USD
    cost_out = (output_tokens / 1_000_000) * PRICE_OUTPUT_1M_USD
    return cost_in + cost_out

def parse_trajectory(messages):
    trajectory = []
    for msg in messages:
        msg_dump = {"role": msg.type, "content": msg.content}
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            msg_dump["tool_calls"] = msg.tool_calls
        if hasattr(msg, "tool_call_id"):
            msg_dump["tool_call_id"] = msg.tool_call_id
        trajectory.append(msg_dump)
    return trajectory

def run_agent_session(prompt_text: str):
    start_time = time.perf_counter()
    
    # Запуск сессии в LangGraph
    state = agent_graph.invoke({"messages": [HumanMessage(content=prompt_text)]})
    
    latency_ms = int((time.perf_counter() - start_time) * 1000)
    messages = state["messages"]
    steps_count = len(messages)
    
    # Собираем метаданные токенов со всех ответов модели в истории
    total_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.usage_metadata:
            total_usage["input_tokens"] += msg.usage_metadata.get("input_tokens", 0)
            total_usage["output_tokens"] += msg.usage_metadata.get("output_tokens", 0)
            total_usage["total_tokens"] += msg.usage_metadata.get("total_tokens", 0)
            
    cost_usd = calculate_cost(total_usage)
    trajectory = parse_trajectory(messages)
    final_output = messages[-1].content
    
    return {
        "steps_count": steps_count,
        "latency_ms": latency_ms,
        "tokens": total_usage,
        "cost_usd": cost_usd,
        "trajectory": trajectory,
        "final_output": final_output
    }

# ==========================================
# ШАГ 3: ПРОГОН НА РЕАЛЬНЫХ МУТАЦИЯХ
# ==========================================

def run_pipeline():
    print("=== ЗАПУСК СКВОЗНОЙ ОЦЕНКИ СТЕНДА НА РЕАЛЬНЫХ МУТАЦИЯХ ===")
    
    input_file = "data/calibration/manual_mutations.jsonl"
    output_file = "data/robustness_report.json"
    
    if not os.path.exists(input_file):
        print(f"Ошибка: Файл {input_file} не найден. Проверьте DVC.")
        return

    eval_results = []
    
    with open(input_file, "r", encoding="utf-8") as f:
        mutations = [json.loads(line) for line in f]
        
    print(f"Загружено {len(mutations)} реальных пар для оценки.")
    
    for m in mutations:
        task_id = m["task_id"]
        print(f"Прогон тест-кейса: {task_id}...")
        
        # 1. Оценка чистого ввода (Baseline)
        clean_res = run_agent_session(m["original"])
        eval_results.append({
            "mutation_id": f"{task_id}_clean",
            "task_id": task_id,
            "prompt_type": "clean",
            "prompt_text": m["original"],
            "steps_count": clean_res["steps_count"],
            "latency_ms": clean_res["latency_ms"],
            "tokens": clean_res["tokens"],
            "cost_usd": clean_res["cost_usd"],
            "model_output": clean_res["final_output"],
            "trajectory": clean_res["trajectory"]
        })
        
        # 2. Оценка зашумленного ввода
        dirty_res = run_agent_session(m["mutated"])
        eval_results.append({
            "mutation_id": f"{task_id}_dirty",
            "task_id": task_id,
            "prompt_type": "dirty",
            "prompt_text": m["mutated"],
            "steps_count": dirty_res["steps_count"],
            "latency_ms": dirty_res["latency_ms"],
            "tokens": dirty_res["tokens"],
            "cost_usd": dirty_res["cost_usd"],
            "model_output": dirty_res["final_output"],
            "trajectory": dirty_res["trajectory"]
        })
        
    # Сохраняем расширенный отчет
    os.makedirs("data", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f_out:
        json.dump(eval_results, f_out, ensure_ascii=False, indent=2)
        
    print(f"=== ОЦЕНКА ЗАВЕРШЕНА. РАСШИРЕННЫЙ ЛОГ СОХРАНЕН В {output_file} ===")

if __name__ == "__main__":
    run_pipeline()

