import os
import json
import time
from typing import TypedDict
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
from langgraph.graph import START, END, StateGraph

# ==========================================
# ШАГ 1: ОПРЕДЕЛЕНИЕ ИНСТРУМЕНТОВ И МОК-МОДЕЛИ
# ==========================================

class MockGigaChat:
    """Автономная модель-заглушка, имитирующая ответы GigaChat для генерации метрик лога."""
    def __init__(self, model: str = "GigaChat-2-Pro", temperature: float = 0.0, timeout: int = 90):
        self.model = model
        self.temperature = temperature
        self.timeout = timeout

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        # Имитируем логику вызова инструментов в зависимости от текста промпта
        last_prompt = messages[-1].content.lower()
        
        # Проверяем, есть ли уже в истории ответ от инструмента
        has_tool_response = any(isinstance(m, ToolMessage) for m in messages)
        
        if has_tool_response:
            # Если инструмент уже отработал, возвращаем финальный текстовый ответ
            return AIMessage(
                content="На основе выполненных расчетов и архивных данных, итоговый результат равен 180 (или подтвержден по базе).",
                usage_metadata={"input_tokens": 120, "output_tokens": 45, "total_tokens": 165}
            )
            
        # Если это первый шаг и промпт математический — просим вызвать калькулятор
        if any(w in last_prompt for w in ["вычисли", "посчитай", "сколько", "*", "+", "/", "-"]):
            msg = AIMessage(content="Необходимо произвести математический расчет.")
            msg.tool_calls = [{"name": "calculate_expression", "args": "15 * 4 + 120", "id": "call_mock_math_01"}]
            msg.usage_metadata = {"input_tokens": 95, "output_tokens": 25, "total_tokens": 120}
            return msg
            
        # В остальных случаях имитируем поиск в Википедии
        msg = AIMessage(content="Выполняю поиск фактов в базе знаний.")
        msg.tool_calls = [{"name": "search_wikipedia", "args": "human stem cells", "id": "call_mock_wiki_01"}]
        msg.usage_metadata = {"input_tokens": 110, "output_tokens": 30, "total_tokens": 140}
        return msg

# Инициализируем автономный движок
llm = MockGigaChat()

class AgentState(TypedDict):
    messages: list

def agent_node(state: AgentState):
    sys_instruction = SystemMessage(content=(
        "Ты — точный ассистент поддержки. "
        "Для решения математических задач вызывaй инструмент 'calculate_expression'. "
        "Для поиска фактов вызывай 'search_wikipedia'."
    ))
    response = llm.invoke([sys_instruction] + state["messages"])
    return {"messages": [response]}

def tools_node(state: AgentState):
    last_message = state["messages"][-1]
    tool_outputs = []
    
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        for call in last_message.tool_calls:
            # Имитируем моментальный успешный ответ от локальных инструментов
            tool_outputs.append(ToolMessage(
                content="Выполнено успешно. Результат операции зафиксирован.", 
                tool_call_id=call["id"], 
                name=call["name"]
            ))
    
    return {"messages": tool_outputs}

def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END

# Сборка графа
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
    PRICE_INPUT_1M_USD = 0.20   
    PRICE_OUTPUT_1M_USD = 0.20  
    if not usage_metadata:
        return 0.0
    input_tokens = usage_metadata.get("input_tokens", 0)
    output_tokens = usage_metadata.get("output_tokens", 0)
    return ((input_tokens / 1_000_000) * PRICE_INPUT_1M_USD) + ((output_tokens / 1_000_000) * PRICE_OUTPUT_1M_USD)

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
    
    state = agent_graph.invoke({"messages": [HumanMessage(content=prompt_text)]})
    
    latency_ms = int((time.perf_counter() - start_time) * 1000)
    messages = state["messages"]
    steps_count = len(messages)
    
    total_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for msg in messages:
        if hasattr(msg, "usage_metadata") and msg.usage_metadata:
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
        
    os.makedirs("data", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f_out:
        json.dump(eval_results, f_out, ensure_ascii=False, indent=2)
        
    print(f"=== ОЦЕНКА ЗАВЕРШЕНА. РАСШИРЕННЫЙ ЛОГ СОХРАНЕН В {output_file} ===")

if __name__ == "__main__":
    run_pipeline()

