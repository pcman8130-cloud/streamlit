"""채팅과 문서 검색 앱에서 사용하는 모델·검색 함수."""

import os
from pathlib import Path

import streamlit as st

MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:2b")
HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL = os.getenv("HF_EMBED_MODEL", "intfloat/multilingual-e5-small")
DOCUMENT = Path(__file__).parent / "data/library.txt"


def prepare():
    from sentence_transformers import SentenceTransformer

    SentenceTransformer(EMBED_MODEL)
    print("임베딩 모델 준비 완료:", EMBED_MODEL)


def stream_text(messages):
    from langchain_ollama import ChatOllama

    model = ChatOllama(model=MODEL, base_url=HOST, temperature=0, reasoning=True)
    for part in model.stream(messages):
        yield part.content or ""


@st.cache_resource
def embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBED_MODEL, local_files_only=True)


@st.cache_data
def document_vectors(chunks):
    return embedder().encode(["passage: " + chunk for chunk in chunks], normalize_embeddings=True)


def retrieve(text, question):
    chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
    if not chunks:
        return []
    vectors = document_vectors(chunks)
    query = embedder().encode(["query: " + question], normalize_embeddings=True)[0]
    scores = vectors @ query
    return [(index + 1, float(scores[index]), chunks[index]) for index in scores.argsort()[::-1][:3]]

MAX_REASONING_CHARS = 1000   # 추론이 이 글자 수를 넘으면 멈춘다
MAX_REPEAT = 3               # 같은 문장이 이 횟수만큼 나오면 멈춘다


def _repeated(reasoning, max_repeat):
    """추론 안에서 같은 문장이 max_repeat번 이상 나왔는지 확인한다."""
    import re

    counts = {}
    for sentence in re.split(r"[\n.!?。]", reasoning):
        sentence = sentence.strip()
        if len(sentence) < 10:          # 너무 짧은 조각은 우연히 겹칠 수 있어서 뺀다
            continue
        counts[sentence] = counts.get(sentence, 0) + 1
        if counts[sentence] >= max_repeat:
            return True
    return False


def stream_answer(messages, info, max_chars=MAX_REASONING_CHARS, max_repeat=MAX_REPEAT):
    """추론이 길어지거나 반복되면 멈추고, 그때까지의 추론을 바탕으로 답한다.
    info에는 추론 내용(reasoning)과 멈춘 이유(stopped)를 담아 화면에 보여 준다."""
    from langchain_ollama import ChatOllama

    thinker = ChatOllama(model=MODEL, base_url=HOST, temperature=0, reasoning=True)
    reasoning = ""
    answered = False
    info["stopped"] = None
    for part in thinker.stream(messages):
        think = part.additional_kwargs.get("reasoning_content", "")
        if think:
            reasoning += think
            if len(reasoning) > max_chars:
                info["stopped"] = f"추론이 {max_chars}자를 넘어서 멈췄습니다."
                break
            if _repeated(reasoning, max_repeat):
                info["stopped"] = f"같은 추론이 {max_repeat}번 반복되어 멈췄습니다."
                break
        if part.content:
            answered = True
            yield part.content
    info["reasoning"] = reasoning
    if answered:
        return

    # 추론을 멈췄거나 답이 비었으면: 지금까지의 추론을 붙여서 추론 없이 바로 답하게 한다
    note = "\n\n[지금까지 정리한 생각]\n" + reasoning[-max_chars:] + "\n\n더 생각하지 말고 위 내용을 바탕으로 바로 짧게 답한다."
    if messages and messages[0]["role"] == "system":
        messages = [{"role": "system", "content": messages[0]["content"] + note}, *messages[1:]]
    else:
        messages = [{"role": "system", "content": note}, *messages]
    writer = ChatOllama(model=MODEL, base_url=HOST, temperature=0, reasoning=False)
    for part in writer.stream(messages):
        if part.content:
            yield part.content