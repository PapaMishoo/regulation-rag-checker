import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

PROVIDER = "gemini"
MODEL = "gemini-3.5-flash"  # the one place to change if you want to try another model

_client = None


def _get_client():
    global _client
    if _client is None:
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise SystemExit("No GEMINI_API_KEY found. Check that your .env file has GEMINI_API_KEY=... in it.")
        _client = genai.Client(api_key=key)
    return _client


def generate(system_prompt, user_prompt, max_tokens=2000):
    """Ask the model for a JSON reply. Returns (text, input_tokens, output_tokens)."""
    client = _get_client()
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=0,                       # same input should give (nearly) the same output
        response_mime_type="application/json",
        max_output_tokens=max_tokens,
    )
    last_error = None
    for attempt in range(4):
        try:
            response = client.models.generate_content(model=MODEL, contents=user_prompt, config=config)
            usage = response.usage_metadata
            return (
                response.text or "",
                getattr(usage, "prompt_token_count", 0) or 0,
                getattr(usage, "candidates_token_count", 0) or 0,
            )
        except Exception as error:
            last_error = error
            if any(code in str(error) for code in ("429", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE")):
                time.sleep(10 * (attempt + 1))  # free tier is busy or rate-limited: wait, then retry
                continue
            break
    raise RuntimeError(str(last_error))