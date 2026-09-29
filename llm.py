import os
from collections.abc import Generator

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

_client = None


# Lazy so a missing key only fails the query, not server startup.
def _get_client():
    global _client
    if _client is None:
        key = os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("GOOGLE_API_KEY is not set (add it to .env).")
        _client = genai.Client(api_key=key)
    return _client


MODEL = "gemini-2.5-flash"

_SYSTEM = (
    "Answer strictly from the provided context. "
    "If the answer isn't there, say you couldn't find it. "
    "Cite the source filename when you use information from it. "
    "Be concise and precise."
)

_CONFIG = types.GenerateContentConfig(
    system_instruction=_SYSTEM,
    temperature=0,
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
)

MAX_HISTORY_TURNS = 6


def _format_history(history: list[dict] | None) -> str:
    if not history:
        return ""
    trimmed = history[-MAX_HISTORY_TURNS:]
    lines = []
    for turn in trimmed:
        role = "User" if turn.get("role") == "user" else "Assistant"
        content = (turn.get("content") or "").strip()
        if content:
            lines.append(f"{role}: {content}")
    if not lines:
        return ""
    return "Prior conversation (for context on follow-up questions):\n" + "\n".join(lines) + "\n\n"


def _prompt(question: str, context: str, history: list[dict] | None = None) -> str:
    return (
        f"{_format_history(history)}"
        f"Context:\n{context}\n\nQuestion: {question}"
    )


def stream_generate(question: str, context: str,
                     history: list[dict] | None = None) -> Generator[str, None, None]:
    try:
        for chunk in _get_client().models.generate_content_stream(
            model=MODEL,
            contents=_prompt(question, context, history),
            config=_CONFIG,
        ):
            text = chunk.text
            if text:
                yield text
    except Exception as e:
        yield f"\n\n[LLM error: {e}]"
