"""챗봇이 쓰는 계산 도구. 도구를 추가하면 TOOLS에도 넣는다."""

from langchain_core.tools import tool


@tool
def add(a: float, b: float) -> float:
    """덧셈(+)을 계산한다. 예: 3+5"""
    return a + b


@tool
def subtract(a: float, b: float) -> float:
    """뺄셈(-)을 계산한다. 예: 17-2. 먹거나 없어진 개수를 뺄 때 사용한다."""
    return a - b


@tool
def multiply(a: float, b: float) -> float:
    """곱셈(*)을 계산한다. 예: 4*6"""
    return a * b


@tool
def divide(a: float, b: float) -> float | str:
    """나눗셈(/)을 계산한다. 예: 10/2"""
    if b == 0:
        return "0으로 나눌 수 없습니다."
    return a / b


TOOLS = {t.name: t for t in [add, subtract, multiply, divide]}