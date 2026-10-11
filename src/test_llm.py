import os
import sys

from dotenv import load_dotenv
from google import genai

load_dotenv()
key = os.getenv("GEMINI_API_KEY")
if not key:
    raise SystemExit("No GEMINI_API_KEY found. Check that your .env file has GEMINI_API_KEY=... in it.")
print("Key found. It starts with", key[:3], "and has", len(key), "characters.")

client = genai.Client(api_key=key)

if len(sys.argv) == 1:
    # Part 1: list the models this key can use for writing text
    print("\nModels this key can use for text generation:")
    for m in client.models.list():
        if "generateContent" in (getattr(m, "supported_actions", None) or []):
            print("  ", m.name.replace("models/", ""))
    print("\nNow run this file again with one of the names above, for example:")
    print("  python src/test_llm.py NAME")
else:
    # Part 2: try a tiny request with the model name you chose
    model = sys.argv[1]
    try:
        response = client.models.generate_content(model=model, contents="Reply with the single word: ready")
        print("\nIt worked. The model replied:", response.text)
    except Exception as error:
        print("\nThe request failed:")
        print(error)