import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

_client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

SYSTEM = (
    "Answer strictly from the provided context. "
    "If the answer isn't there, say you couldn't find it. "
    "Cite the filename when you use information from it."
)

MODEL = "gemini-2.5-flash"


def generate(question: str, context: str) -> str:
    prompt = f"Context:\n{context}\n\nQuestion: {question}"
    try:
        resp = _client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM,
                temperature=0,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True,
                ),
            ),
        )
        # Bug fix: resp.text can be None when the response is blocked or empty
        return resp.text or "(The model returned an empty or filtered response.)"
    except Exception as e:
        return f"LLM error: {e}"
