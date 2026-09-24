"""
llm.py — Gemini wrapper.

Two modes:
  generate()        — blocking, returns the full answer string.
  stream_generate() — sync generator, yields text tokens as they arrive.
                      Use this for the streaming SSE endpoint.
"""

import os
from collections.abc import Generator

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

_client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

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


def _prompt(question: str, context: str) -> str:
    return f"Context:\n{context}\n\nQuestion: {question}"


# ── Blocking ──────────────────────────────────────────────────────
def generate(question: str, context: str) -> str:
    """Return the complete answer as a single string."""
    try:
        resp = _client.models.generate_content(
            model=MODEL,
            contents=_prompt(question, context),
            config=_CONFIG,
        )
        return resp.text or "(The model returned an empty or filtered response.)"
    except Exception as e:
        return f"LLM error: {e}"


# ── Streaming ─────────────────────────────────────────────────────
def stream_generate(question: str, context: str) -> Generator[str, None, None]:
    """
    Yield text tokens as they are produced by the model.

    Designed to be driven from a background thread:

        def _run():
            for tok in stream_generate(q, ctx):
                queue.put(tok)
            queue.put(None)   # sentinel

        threading.Thread(target=_run, daemon=True).start()
    """
    try:
        for chunk in _client.models.generate_content_stream(
            model=MODEL,
            contents=_prompt(question, context),
            config=_CONFIG,
        ):
            text = chunk.text
            if text:
                yield text
    except Exception as e:
        yield f"\n\n[LLM error: {e}]"
