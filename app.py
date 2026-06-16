import json
import uuid
from datetime import datetime

import pandas as pd
import streamlit as st
from google import genai
from google.genai import types
from langchain_core.messages import HumanMessage, SystemMessage

from agent import (
    MODEL_NAME,
    create_agent,
    extract_response_text,
    extract_tools_debug,
    list_snippets_direct,
    save_snippet_direct,
)


# ============================================================
# Налаштування сторінки
# ============================================================
st.set_page_config(
    page_title="CodeBot — AI код-ревʼюер",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Ініціалізація стану
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())[:8]

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = (
        "Ти CodeBot — технічний AI-асистент для code review. "
        "Відповідай українською мовою. Пояснюй код структуровано, коротко і практично. "
        "Звертай увагу на помилки, читабельність, продуктивність, безпеку та можливості рефакторингу."
    )

if "code_analysis" not in st.session_state:
    st.session_state.code_analysis = ""

if "last_debug" not in st.session_state:
    st.session_state.last_debug = []


# ============================================================
# Кешовані ресурси
# ============================================================
@st.cache_resource
def get_gemini_client(api_key: str):
    return genai.Client(api_key=api_key)


@st.cache_resource
def get_langgraph_agent(api_key: str, model_name: str):
    return create_agent(api_key, model_name)


api_key = st.secrets.get("GOOGLE_API_KEY")

if not api_key:
    st.error("❌ Не знайдено GOOGLE_API_KEY у файлі .streamlit/secrets.toml")
    st.info(
        """
        Створіть файл `.streamlit/secrets.toml` і додайте:

        ```toml
        GOOGLE_API_KEY = "ваш_api_key"
        ```
        """
    )
    st.stop()

client = get_gemini_client(api_key)


# ============================================================
# CSS
# ============================================================
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
        }

        .codebot-card {
            padding: 1rem;
            border-radius: 0.75rem;
            border: 1px solid rgba(128, 128, 128, 0.25);
            background: rgba(128, 128, 128, 0.06);
            margin-bottom: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Допоміжні функції
# ============================================================
def convert_to_gemini_history(messages: list) -> list[types.Content]:
    contents = []

    for message in messages:
        role = "user" if message["role"] == "user" else "model"

        contents.append(
            types.Content(
                role=role,
                parts=[types.Part.from_text(text=message["content"])],
            )
        )

    return contents


def stream_gemini_response(prompt: str, history: list):
    contents = []

    if st.session_state.system_prompt:
        contents.append(
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text=f"[Системні інструкції]: {st.session_state.system_prompt}"
                    )
                ],
            )
        )
        contents.append(
            types.Content(
                role="model",
                parts=[
                    types.Part.from_text(
                        text="Зрозуміло. Я буду дотримуватися цих інструкцій."
                    )
                ],
            )
        )

    contents.extend(convert_to_gemini_history(history))

    contents.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)],
        )
    )

    try:
        stream = client.models.generate_content_stream(
            model=MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )

        for chunk in stream:
            if chunk.text:
                yield chunk.text

    except Exception as error:
        yield f"\n\n❌ **Помилка Gemini API:** {error}"


def get_agent_response(prompt: str):
    agent = get_langgraph_agent(api_key, MODEL_NAME)

    config = {
        "configurable": {
            "thread_id": st.session_state.thread_id,
        }
    }

    system_message = SystemMessage(
        content=st.session_state.system_prompt,
        id="user_system_prompt",
    )

    try:
        result = agent.invoke(
            {
                "messages": [
                    system_message,
                    HumanMessage(content=prompt),
                ]
            },
            config,
        )

        final_message = result["messages"][-1]
        text = extract_response_text(final_message)
        debug = extract_tools_debug(result["messages"])

        return text, debug

    except Exception as error:
        return f"❌ **Помилка агента:** {error}", []


def build_code_prompt(selected_action: str, selected_language: str, code: str) -> str:
    action_map = {
        "💡 Пояснити": "Поясни, що робить цей код.",
        "🔧 Оптимізувати": "Оптимізуй цей код і поясни, що саме покращено.",
        "🐛 Знайти баги": "Знайди потенційні баги, edge cases та проблеми в цьому коді.",
        "📝 Додати коментарі": "Додай корисні коментарі до цього коду і коротко поясни зміни.",
    }

    task = action_map.get(selected_action, "Проаналізуй цей код.")

    return f"""
{task}

Мова програмування: {selected_language}

Код:
```{selected_language}
{code}
```
"""


def export_payload(mode: str, language: str) -> str:
    data = {
        "exported_at": datetime.now().isoformat(),
        "project": "CodeBot",
        "variant": "Варіант 11",
        "model": MODEL_NAME,
        "mode": mode,
        "language": language,
        "thread_id": st.session_state.thread_id,
        "system_prompt": st.session_state.system_prompt,
        "messages": st.session_state.messages,
        "snippets": list_snippets_direct(),
    }

    return json.dumps(data, ensure_ascii=False, indent=2)


# ============================================================
# Заголовок
# ============================================================
st.title("💻 CodeBot — AI код-ревʼюер")

st.markdown(
    """
    Веб-застосунок для пояснення, ревʼю, оптимізації та збереження фрагментів коду.  
    Працює на **Streamlit**, **Google Gemini** та **LangGraph-агенті з інструментами**.
    """
)


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.header("⚙️ Налаштування")

    mode = st.radio(
        "Режим роботи",
        ["💬 Звичайний чат", "🛠️ Агент з інструментами"],
        index=0,
    )

    language = st.selectbox(
        "Мова програмування",
        [
            "python",
            "javascript",
            "typescript",
            "java",
            "c++",
            "rust",
            "go",
            "html",
            "css",
        ],
        index=0,
    )

    action = st.radio(
        "⚡ Швидка дія",
        [
            "💡 Пояснити",
            "🔧 Оптимізувати",
            "🐛 Знайти баги",
            "📝 Додати коментарі",
        ],
        index=0,
    )

    st.info(f"Модель: **{MODEL_NAME}**")

    temperature = st.slider(
        "Температура",
        min_value=0.0,
        max_value=1.0,
        value=0.4,
        step=0.1,
        help="Для code review краще нижча температура",
    )

    max_tokens = st.number_input(
        "Макс. токенів",
        min_value=100,
        max_value=8192,
        value=2048,
        step=100,
    )

    st.divider()

    with st.expander("📝 Системний промпт", expanded=False):
        system_prompt_input = st.text_area(
            "Інструкції для моделі",
            value=st.session_state.system_prompt,
            height=140,
        )

        if st.button("💾 Зберегти промпт", use_container_width=True):
            st.session_state.system_prompt = system_prompt_input.strip()
            st.toast("Системний промпт збережено")

    st.divider()

    col_clear, col_export = st.columns(2)

    with col_clear:
        if st.button("🗑️ Очистити", use_container_width=True):
            st.session_state.messages = []
            st.session_state.thread_id = str(uuid.uuid4())[:8]
            st.session_state.code_analysis = ""
            st.session_state.last_debug = []
            st.rerun()

    with col_export:
        if st.session_state.messages or list_snippets_direct():
            st.download_button(
                "📥 Експорт",
                data=export_payload(mode, language),
                file_name=f"codebot_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True,
            )

    st.divider()

    message_count = len(st.session_state.messages)
    approximate_tokens = sum(
        len(message.get("content", "")) for message in st.session_state.messages
    ) // 4

    st.metric("Повідомлень", message_count)
    st.metric("~Токенів", approximate_tokens)
    st.metric("Снипетів", len(list_snippets_direct()))

    st.divider()

    st.header("📁 Мої снипети")

    snippets = list_snippets_direct()

    if snippets:
        for index, snippet in enumerate(snippets, start=1):
            with st.expander(f"{index}. {snippet.get('note', 'Без назви')}"):
                st.caption(f"Мова: {snippet.get('language', 'text')}")
                st.caption(f"Створено: {snippet.get('created_at', '-')}")
                st.code(
                    snippet.get("code", ""),
                    language=snippet.get("language", "text"),
                )
    else:
        st.caption("Поки немає збережених snippets.")

    st.divider()

    if "Агент" in mode:
        with st.expander("🛠️ Інструменти агента"):
            st.markdown(
                """
                Доступні tools:

                - **save_snippet** — зберегти код
                - **list_snippets** — показати snippets
                - **calculator** — математичні обчислення
                - **wikipedia_search** — коротка довідка
                """
            )

    with st.expander("ℹ️ Про застосунок"):
        st.markdown(
            """
            **CodeBot — варіант 11**

            Реалізовано:
            - чат-інтерфейс;
            - поле для коду;
            - вибір мови програмування;
            - підсвітка синтаксису через `st.code`;
            - швидкі дії для code review;
            - збереження snippets;
            - LangGraph-агент з tools;
            - експорт історії та snippets;
            - статистика використання.
            """
        )


# ============================================================
# Tabs
# ============================================================
code_tab, chat_tab, stats_tab = st.tabs(["📝 Код", "💬 Чат", "📊 Статистика"])


# ============================================================
# Вкладка: Код
# ============================================================
with code_tab:
    st.subheader("📝 Введіть код для аналізу")

    col_input, col_output = st.columns(2)

    with col_input:
        code_input = st.text_area(
            "Ваш код",
            height=320,
            placeholder="Вставте код сюди...",
            key="code_input",
        )

        note = st.text_input(
            "Назва snippet",
            placeholder="Наприклад: приклад сортування",
            key="snippet_note",
        )

        if code_input:
            st.markdown("**Попередній перегляд з підсвіткою:**")
            st.code(code_input, language=language)

        btn_analyze, btn_save, btn_copy = st.columns(3)

        with btn_analyze:
            analyze_clicked = st.button("🔍 Аналізувати", use_container_width=True)

        with btn_save:
            save_clicked = st.button("💾 Зберегти", use_container_width=True)

        with btn_copy:
            copy_clicked = st.button("📋 Копіювати", use_container_width=True)

        if copy_clicked:
            st.toast("Код готовий до копіювання")

        if save_clicked:
            if not code_input.strip():
                st.warning("Спочатку вставте код.")
            else:
                snippet = save_snippet_direct(
                    language=language,
                    code=code_input,
                    note=note.strip() or f"Снипет {len(list_snippets_direct()) + 1}",
                )
                st.success(f"Збережено: {snippet['note']}")
                st.rerun()

        if analyze_clicked:
            if not code_input.strip():
                st.warning("Спочатку вставте код.")
            else:
                prompt = build_code_prompt(action, language, code_input)

                st.session_state.messages.append(
                    {
                        "role": "user",
                        "content": prompt,
                    }
                )

                with st.spinner("CodeBot аналізує код..."):
                    if "Агент" in mode:
                        response, debug = get_agent_response(prompt)
                    else:
                        response_parts = []
                        for chunk in stream_gemini_response(
                            prompt,
                            st.session_state.messages[:-1],
                        ):
                            response_parts.append(chunk)

                        response = "".join(response_parts)
                        debug = []

                st.session_state.code_analysis = response
                st.session_state.last_debug = debug

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response,
                    }
                )

                st.rerun()

    with col_output:
        st.subheader("🔍 Результат аналізу")

        if st.session_state.code_analysis:
            st.markdown(st.session_state.code_analysis)

            if st.session_state.last_debug:
                with st.expander("Debug: tool calls / results"):
                    st.json(st.session_state.last_debug)
        else:
            st.info(
                "Результат аналізу зʼявиться тут після натискання кнопки «Аналізувати»."
            )


# ============================================================
# Вкладка: Чат
# ============================================================
with chat_tab:
    if not st.session_state.messages:
        with st.chat_message("assistant"):
            st.markdown(
                f"""
                👋 Вітаю! Я **CodeBot** на базі **{MODEL_NAME}**.

                Я можу:
                - пояснювати код;
                - оптимізувати функції;
                - знаходити баги;
                - додавати коментарі;
                - зберігати snippets;
                - показувати збережені snippets.

                Спробуйте:
                - `Поясни цей код: ...`
                - `Збережи цей снипет як "приклад сортування": ...`
                - `Покажи мої збережені коди`
                - `Як оптимізувати цей цикл?`
                """
            )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Запитайте CodeBot..."):
        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if "Агент" in mode:
                with st.spinner("🤔 CodeBot думає..."):
                    response, debug = get_agent_response(prompt)

                st.markdown(response)

                if debug:
                    with st.expander("Debug: tool calls / results"):
                        st.json(debug)

            else:
                with st.spinner("CodeBot відповідає..."):
                    response = st.write_stream(
                        stream_gemini_response(
                            prompt,
                            st.session_state.messages[:-1],
                        )
                    )

        if response:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response,
                }
            )

            st.rerun()


# ============================================================
# Вкладка: Статистика
# ============================================================
with stats_tab:
    st.subheader("📊 Статистика CodeBot")

    total_messages = len(st.session_state.messages)
    user_messages = len(
        [message for message in st.session_state.messages if message["role"] == "user"]
    )
    assistant_messages = len(
        [
            message
            for message in st.session_state.messages
            if message["role"] == "assistant"
        ]
    )
    total_chars = sum(
        len(message.get("content", "")) for message in st.session_state.messages
    )
    estimated_tokens = total_chars // 4
    snippets_count = len(list_snippets_direct())

    stat_col1, stat_col2, stat_col3 = st.columns(3)

    with stat_col1:
        st.metric("Усього повідомлень", total_messages)

    with stat_col2:
        st.metric("Повідомлень користувача", user_messages)

    with stat_col3:
        st.metric("Відповідей асистента", assistant_messages)

    stat_col4, stat_col5, stat_col6 = st.columns(3)

    with stat_col4:
        st.metric("Символів", total_chars)

    with stat_col5:
        st.metric("~Токенів", estimated_tokens)

    with stat_col6:
        st.metric("Снипетів", snippets_count)

    st.divider()

    st.subheader("📋 Історія повідомлень")

    if st.session_state.messages:
        table_data = []

        for index, message in enumerate(st.session_state.messages, start=1):
            table_data.append(
                {
                    "№": index,
                    "Роль": "Користувач" if message["role"] == "user" else "Асистент",
                    "Текст": message["content"],
                    "Символів": len(message["content"]),
                }
            )

        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("Історія повідомлень поки порожня.")

    st.divider()

    st.subheader("📁 Збережені snippets")

    snippets = list_snippets_direct()

    if snippets:
        snippet_table = []

        for index, snippet in enumerate(snippets, start=1):
            snippet_table.append(
                {
                    "№": index,
                    "Назва": snippet.get("note", ""),
                    "Мова": snippet.get("language", ""),
                    "Створено": snippet.get("created_at", ""),
                    "Символів": len(snippet.get("code", "")),
                }
            )

        st.dataframe(pd.DataFrame(snippet_table), use_container_width=True)
    else:
        st.info("Збережених snippets поки немає.")
