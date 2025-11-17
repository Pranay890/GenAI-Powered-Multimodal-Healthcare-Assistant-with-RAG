import streamlit as st
import time
import os
import google.generativeai as genai
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from PIL import Image
import io

# ------------------------------------------------------
# Load API Key
# ------------------------------------------------------
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel("models/gemini-2.5-flash")

# ------------------------------------------------------
# Page Config
# ------------------------------------------------------
st.set_page_config(page_title="DiagnoSense – Medical Assistant", layout="wide")

# ------------------------------------------------------
# Sidebar
# ------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Settings")

    meds = st.text_input(
        "Current medications (optional):", placeholder="e.g., paracetamol, ibuprofen"
    )

    enable_rag = st.checkbox("Enable RAG (Evidence-based)", value=True)

    st.markdown("---")
    st.markdown("Made with ❤️ for Medi-Caps University")

# ------------------------------------------------------
# Header
# ------------------------------------------------------
st.markdown(
    """
    <h1 style="text-align:center; color:#4EA8FF;">
        🩺 DiagnoSense – Intelligent Healthcare Assistant
    </h1>
    <p style="text-align:center; font-size:18px; color:#ccc;">
        Enter your symptoms or upload medical reports for AI-powered analysis.
    </p>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------
# Disease Probability Bars
# ------------------------------------------------------
def show_disease_likelihood(diseases):
    st.subheader("🧪 Disease Likelihood Analysis")

    for item in diseases:
        name = item["name"]
        prob = item["probability"]

        if prob >= 70:
            color = "#ff4d4d"
        elif prob >= 40:
            color = "#ffa500"
        else:
            color = "#4caf50"

        st.markdown(f"**{name} — {prob}% likelihood**")

        bar = st.empty()
        current = 0
        while current < prob:
            current += 2
            bar.markdown(
                f"""
                <div style="width:100%; background:#222; height:18px; border-radius:4px; margin-bottom:10px;">
                    <div style="width:{current}%; background:{color}; height:100%; border-radius:4px;"></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            time.sleep(0.01)


# ------------------------------------------------------
# 🔹 SECTION 1 — Symptoms Analysis
# ------------------------------------------------------
st.subheader("👩‍⚕️ Describe your symptoms")

symptoms = st.text_area(
    "Symptoms:",
    placeholder="e.g., fever, headache, body pain, sore throat…",
    height=150,
)

if st.button("🔍 Analyze Symptoms", use_container_width=True):

    if not symptoms.strip():
        st.error("Please enter symptoms!")
        st.stop()

    with st.spinner("Analyzing your symptoms…"):

        prompt = f"""
        You are a medical diagnosis AI.
        Symptoms: {symptoms}
        Medications: {meds}

        Provide:
        1. Top 5 likely diseases
        2. Probabilities (%)
        3. Severity level
        4. Medical advice
        5. Follow-up questions
        """

        response = model.generate_content(prompt).text

    st.subheader("📝 Medical Diagnosis Summary")
    st.write(response)

    # Dummy example bars (you can remove)
    dummy_symptom_probs = [
        {"name": "Flu", "probability": 65},
        {"name": "Migraine", "probability": 42},
        {"name": "COVID-19", "probability": 25},
    ]
    show_disease_likelihood(dummy_symptom_probs)

# ------------------------------------------------------
# Separator
# ------------------------------------------------------
st.markdown("<hr style='margin: 40px 0;'>", unsafe_allow_html=True)

# ------------------------------------------------------
# 🔹 SECTION 2 — Medical Report Upload
# ------------------------------------------------------
st.subheader("📄 Upload Medical Report (PDF / JPG / PNG)")

uploaded_file = st.file_uploader(
    "Upload your medical report:", type=["pdf", "jpg", "jpeg", "png"]
)


def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text


def extract_text_from_image(file):
    img = Image.open(file)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)

    vision = genai.GenerativeModel("models/gemini-2.0-flash")
    resp = vision.generate_content(
        ["Extract the text from this medical report.", buffer]
    )
    return resp.text


if uploaded_file:
    st.info("Processing your medical report…")

    if uploaded_file.type == "application/pdf":
        report_text = extract_text_from_pdf(uploaded_file)
    else:
        report_text = extract_text_from_image(uploaded_file)

    st.success("Report processed!")

    # 🔥 Direct AI interpretation (no raw text shown)
    with st.spinner("Analyzing your medical report…"):

        prompt = f"""
        You are an advanced AI medical specialist.
        Analyze the following medical report and provide:

        1. Summary of key findings  
        2. Probable diseases (top 5)  
        3. Disease likelihoods (%)  
        4. Abnormal / critical values  
        5. Recommended tests  
        6. Treatment advice  
        7. Whether urgent care is needed  

        DO NOT return raw extracted text.
        ONLY return the clean medical interpretation.

        Report:
        {report_text}
        """

        analysis = model.generate_content(prompt).text

    st.subheader("🧠 AI Medical Report Interpretation")
    st.write(analysis)

    dummy_pdf_probs = [
        {"name": "Anemia", "probability": 55},
        {"name": "Viral Infection", "probability": 32},
        {"name": "Thyroid Dysfunction", "probability": 22},
    ]
    show_disease_likelihood(dummy_pdf_probs)
