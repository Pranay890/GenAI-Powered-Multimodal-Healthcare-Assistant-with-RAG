import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY missing in .env")

genai.configure(api_key=api_key)

print("Listing available Gemini models:\n")
for m in genai.list_models():
    print(m.name)
