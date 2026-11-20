from shared import DOCUMENT, retrieve, stream_answer

import streamlit as st

st.title("내 문서에 질문하기")
upload = st.file_uploader("문단을 빈 줄로 구분한 UTF-8 텍스트 파일", type=["txt"])
text = upload.getvalue().decode("utf-8") if upload else DOCUMENT.read_text(encoding="utf-8")
if st.session_state.get("document") != text:
    st.session_state.document = text
    st.session_state.messages = []
with st.expander("현재 문서"):
    st.text(text)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if question := st.chat_input("문서에 관한 질문을 입력하세요"):
    matches = retrieve(text, question)
    with st.expander("검색한 문단"):
        for number, score, chunk in matches:
            st.write(f"문단 {number}, 유사도 {score:.3f}")
            st.write(chunk)
    context = "\n\n".join(f"[문단 {number}] {chunk}" for number, score, chunk in matches)
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    system = "한국어로 짧게 답한다. 다음 자료에 근거해 답하고 [문단 번호]를 표시한다. 자료에 없으면 찾을 수 없다고 답한다.\n\n" + context
    with st.chat_message("assistant"):
        info = {}
        answer = st.write_stream(
            stream_answer([{"role": "system", "content": system}, *st.session_state.messages], info)
        )
        if info.get("stopped"):
            st.caption(info["stopped"] + " 지금까지의 추론을 바탕으로 답했습니다.")
        with st.expander(f"추론 과정 ({len(info.get('reasoning', ''))}자)"):
            st.text(info.get("reasoning", ""))
        if answer and answer.strip():
            st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            st.warning("모델이 답변 본문을 보내지 않았습니다. 질문을 다시 입력하세요.")