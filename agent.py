from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any

import numexpr as ne
from typing_extensions import TypedDict

from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import WikipediaAPIWrapper

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# ============================================================
# Константи
# ============================================================
MODEL_NAME = "gemini-2.5-flash-lite"

SYSTEM_PROMPT = SystemMessage(
    content=(
        "Ти — CodeBot, AI-асистент для code review, пояснення та збереження коду. "
        "Відповідай українською мовою. "
        "Твоя спеціалізація: пояснення коду, пошук помилок, оптимізація, рефакторинг, "
        "додавання коментарів та збереження корисних snippets. "
        "Якщо користувач просить зберегти код — використовуй tool save_snippet. "
        "Якщо користувач просить показати збережені snippets — використовуй tool list_snippets. "
        "Якщо треба щось порахувати — використовуй tool calculator. "
        "Якщо потрібна коротка довідка про технологію, бібліотеку, алгоритм або концепцію — використовуй wikipedia_search. "
        "У code review давай відповідь структуровано: "
        "1) Що робить код; "
        "2) Потенційні проблеми; "
        "3) Як покращити; "
        "4) За потреби — покращений варіант коду."
    )
)

# ============================================================
# Просте in-memory сховище snippets
# ============================================================
SNIPPETS: list[dict[str, str]] = []


def save_snippet_direct(language: str, code: str, note: str) -> dict[str, str]:
    """
    Зберігає snippet напряму з UI або tool.
    """
    snippet = {
        "language": language or "text",
        "code": code or "",
        "note": note or f"Снипет {len(SNIPPETS) + 1}",
        "created_at": datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
    }

    SNIPPETS.append(snippet)

    return snippet


def list_snippets_direct() -> list[dict[str, str]]:
    """
    Повертає список збережених snippets.
    """
    return SNIPPETS


# ============================================================
# Стан агента
# ============================================================
class AgentState(TypedDict):
    messages: Annotated[list[Any], add_messages]


# ============================================================
# Інструменти агента
# ============================================================
@tool
def save_snippet(language: str, code: str, note: str) -> str:
    """
    Зберігає фрагмент коду у колекцію snippets.

    Args:
        language: Мова програмування, наприклад python, javascript, java.
        code: Код, який потрібно зберегти.
        note: Назва або коротка нотатка до snippet.
    """
    if not code or not code.strip():
        return "Помилка: немає коду для збереження."

    snippet = save_snippet_direct(language, code, note)

    return (
        "Снипет збережено.\n"
        f"Назва: {snippet['note']}\n"
        f"Мова: {snippet['language']}\n"
        f"Дата: {snippet['created_at']}"
    )


@tool
def list_snippets() -> str:
    """
    Повертає список збережених snippets.
    """
    if not SNIPPETS:
        return "Колекція snippets поки порожня."

    lines = ["Збережені snippets:"]

    for index, snippet in enumerate(SNIPPETS, start=1):
        lines.append(
            f"{index}. {snippet['note']} "
            f"({snippet['language']}, {snippet['created_at']})"
        )

    return "\n".join(lines)


@tool
def calculator(expression: str) -> str:
    """
    Обчислює математичний вираз.
    Використовуй для будь-яких арифметичних обчислень.
    """
    expr = (expression or "").strip()

    if not expr:
        return "Помилка: порожній математичний вираз."

    allowed = set("0123456789+-*/()., eE")
    cleaned = expr.replace(" ", "")

    if not all(ch in allowed for ch in cleaned):
        return "Помилка: у виразі є недопустимі символи."

    cleaned = cleaned.replace(",", ".")

    try:
        result = ne.evaluate(cleaned)
        value = result.item() if hasattr(result, "item") else result
        return f"Результат: {expression} = {value}"
    except Exception as error:
        return f"Помилка обчислення: {error}"


wiki_client = WikipediaAPIWrapper(
    lang="uk",
    top_k_results=1,
    doc_content_chars_max=1000,
)


@tool
def wikipedia_search(query: str) -> str:
    """
    Шукає коротку довідкову інформацію в українській Wikipedia.

    Args:
        query: Пошуковий запит, наприклад Python, JavaScript, алгоритм сортування.
    """
    q = (query or "").strip()

    if not q:
        return "Помилка: порожній пошуковий запит."

    try:
        result = wiki_client.run(q)
        return result if result else f"Не знайдено інформацію за запитом: {q}"
    except Exception as error:
        return f"Помилка пошуку у Wikipedia: {error}"


TOOLS = [
    save_snippet,
    list_snippets,
    calculator,
    wikipedia_search,
]


# ============================================================
# Побудова LangGraph-агента
# ============================================================
def create_agent(api_key: str, model_name: str = MODEL_NAME):
    """
    Створює LangGraph-агента CodeBot з інструментами.
    """
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.0,
        google_api_key=api_key,
    )

    llm_with_tools = llm.bind_tools(TOOLS)

    def agent_node(state: AgentState) -> dict:
        messages = [SYSTEM_PROMPT] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    tool_node = ToolNode(TOOLS)

    builder = StateGraph(AgentState)

    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_node)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "tools", END: END},
    )
    builder.add_edge("tools", "agent")

    checkpointer = InMemorySaver()

    return builder.compile(checkpointer=checkpointer)


# ============================================================
# Допоміжні функції для UI
# ============================================================
def extract_response_text(message) -> str:
    """
    Витягує текст з відповіді LangChain/Gemini.
    """
    content = getattr(message, "content", "")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []

        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                if part.get("type") == "text":
                    parts.append(part.get("text", ""))
                elif "text" in part:
                    parts.append(part.get("text", ""))

        return "".join(parts)

    return str(content)


def extract_tools_debug(messages: list[Any]) -> list[dict]:
    """
    Витягує інформацію про tool calls для debug-блоку.
    """
    debug = []

    for message in messages:
        if isinstance(message, ToolMessage):
            debug.append(
                {
                    "type": "tool_result",
                    "content": message.content,
                    "tool_call_id": message.tool_call_id,
                }
            )
            continue

        tool_calls = getattr(message, "tool_calls", None)

        if tool_calls:
            for tool_call in tool_calls:
                debug.append(
                    {
                        "type": "tool_call",
                        "name": tool_call.get("name"),
                        "args": tool_call.get("args"),
                        "id": tool_call.get("id"),
                    }
                )

    return debug