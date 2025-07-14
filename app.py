# app.py
import streamlit as st
from utils import predict_eligibility
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os

@st.cache_resource
def load_model():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

policy_path = r'C:\Users\Ayush J\OneDrive\Desktop\celebal\RAG\policy.txt'
if not os.path.exists(policy_path):
    st.error("Missing policy.txt in data/ folder")
    st.stop()

with open(policy_path, "r", encoding="utf-8") as f:
    policy_text = f.read()


st.set_page_config(page_title="Loan Chatbot (Light HF)", layout="centered")
st.title("🏦 Loan Eligibility Assistant")
st.markdown("Fill in the form below to check loan eligibility and get an AI explanation.")

with st.form("loan_form"):
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["Yes", "No"])
        credit_history = st.selectbox("Credit History", [1.0, 0.0])
    with col2:
        applicant_income = st.number_input("Applicant Income", min_value=0)
        coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
        loan_amount = st.number_input("Loan Amount", min_value=0)
        loan_term = st.selectbox("Loan Term", [360.0, 180.0, 120.0, 60.0])
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    submitted = st.form_submit_button("Check Eligibility")

if submitted:
    user_input = {
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history,
        "Property_Area": property_area
    }


    eligibility = predict_eligibility(user_input)
    st.subheader(f"🔍 Loan Eligibility: **{eligibility}**")

   
    prompt = f"""
Based on the following policy:
\"\"\"{policy_text}\"\"\"

Here is the customer application:
{user_input}

Explain why the customer is **{eligibility}** for a home loan.
"""
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).input_ids
    output = model.generate(input_ids, max_new_tokens=150)
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    st.markdown("### 💬 AI Explanation:")
    st.success(response)
