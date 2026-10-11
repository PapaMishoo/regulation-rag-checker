import os
import re
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

PROVIDER = "gemini"
MODEL = "gemini-3.5-flash-lite"  # the one place to change if you want to try another model
MAX_WAIT = 90  # seconds; if the service asks for a longer wait than this, give up instead of retrying

_client = None


class QuotaExhausted(Exception):
    """The free tier says to come back much later (for example, the daily limit is used up)."""


def _get_client():
    global _client
    if _client is None:
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise SystemExit("No GEMINI_API_KEY found. Check that your .env file has GEMINI_API_KEY=... in it.")
        _client = genai.Client(api_key=key)
    return _client


def _retry_seconds(message):
    """Read hints like 'Please retry in 13h12m12.03s' from an error message."""
    match = re.search(r"retry in (?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)?", message)
    if not match or not any(match.groups()):
        return None
    hours, minutes, seconds = match.groups()
    return int(hours or 0) * 3600 + int(minutes or 0) * 60 + float(seconds or 0)


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
            message = str(error)
            if "429" in message or "RESOURCE_EXHAUSTED" in message:
                wait = _retry_seconds(message)
                if wait is None:
                    wait = 15 * (attempt + 1)
                if wait > MAX_WAIT:
                    raise QuotaExhausted(
                        f"{MODEL}: quota used up. The service says to retry in about {int(wait // 60)} minutes.")
                time.sleep(wait + 1)   # a short per-minute limit: wait as asked, then try again
                continue
            if "503" in message or "UNAVAILABLE" in message:
                time.sleep(10 * (attempt + 1))
                continue
            break
    raise RuntimeError(str(last_error))