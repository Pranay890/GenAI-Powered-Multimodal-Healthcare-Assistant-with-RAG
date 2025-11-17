import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY missing in .env")

# Configure Gemini
genai.configure(api_key=api_key)


class DiagnoSenseBackend:
    def __init__(self):
        # Choose a supported model
        self.model = genai.GenerativeModel("models/gemini-2.5-flash")

    def ask(self, symptoms):
        """
        Generates analysis with *guaranteed* likelihood percentages so that
        Streamlit progress bars work correctly.
        """

        prompt = f"""
        You are DiagnoSense, an advanced intelligent medical assistant.

        PATIENT SYMPTOMS:
        {symptoms}

        YOUR TASKS:
        -----------------------------
        1. Provide a short symptom summary.
        2. List 3–6 possible diseases with EXACT likelihood percentages.

           IMPORTANT FORMAT (do NOT change):
           - Disease Name: 75%
           - Another Disease: 40%
           - Third Disease: 25%

        3. Give reasoning briefly for each disease.
        4. Provide:
            - Risk level (Low / Medium / High)
            - What tests may be needed
            - Warning signs to watch for
            - Home remedies / first aid (if safe)
        5. Suggest whether the patient needs:
            - A General Physician
            - A Specialist (Cardiologist, ENT, Neuro, etc.)
            - OR Emergency Care

        RULES:
        -----------------------------
        • Percentages MUST follow exact pattern: Disease: 45%
        • No extra symbols or different formatting.
        • Keep explanation clear, not too long.
        """

        response = self.model.generate_content(prompt)
        return response.text
