import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from shared import HOST, MODEL

from ex02.tools import TOOLS

PROMPT = ChatPromptTemplate.from_messages([
    ("system", "한국어로 짧게 답한다. 숫자 계산은 직접 하지 말고 반드시 도구를 사용한다."),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])


@st.cache_resource
def create_model():
    model = ChatOllama(model=MODEL, base_url=HOST, temperature=0, reasoning=True)
    return model.bind_tools(list(TOOLS.values()))   # 도구 목록을 모델에 알려준다


def run_with_tools(history, question):
    """모델이 도구를 요청하면 프로그램이 실행하고 결과를 돌려준다. 최대 5번 반복."""
    model = create_model()
    messages = PROMPT.invoke({"history": history, "question": question}).to_messages()
    calls = []
    for _ in range(5):
        response = model.invoke(messages)
        messages.append(response)
        if not response.tool_calls:          # 도구 요청이 없으면 최종 답
            return response.content, calls
        for call in response.tool_calls:
            func = TOOLS.get(call["name"])
            result = func.invoke(call["args"]) if func else f"'{call['name']}' 도구는 없습니다."
            calls.append(f"{call['name']}({call['args']}) -> {result}")
            messages.append(ToolMessage(str(result), tool_call_id=call["id"]))
    return "계산을 끝내지 못했습니다.", calls


# ---- 화면 ----
st.title("로컬 AI 대화")
if "messages" not in st.session_state:
    st.session_state.messages = []
if st.sidebar.button("대화 지우기"):
    st.session_state.messages = []
max_turns = st.sidebar.slider("기억할 대화 수(턴)", 0, 10, 3)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        for c in message.get("calls", []):
            st.caption(f"[함수 실행] {c}")
        if "sent" in message:
            st.caption(message["sent"])

if question := st.chat_input("질문을 입력하세요"):
    history = [
        HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"])
        for m in st.session_state.messages
    ]
    history = trim_messages(
        history, max_tokens=max_turns * 2, token_counter=len,
        strategy="last", start_on="human",
    )

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("생각하는 중..."):
            answer, calls = run_with_tools(history, question)
        sent = f"저장된 대화 {len(st.session_state.messages) - 1}개 중 {len(history)}개를 모델에 보냈습니다."
        st.write(answer)
        for c in calls:
            st.caption(f"[함수 실행] {c}")
        st.caption(sent)
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "calls": calls, "sent": sent}
    )