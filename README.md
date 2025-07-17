# 🧠 RAG: Loan Eligibility Chatbot using ML + Generative AI (Local)

This project combines a traditional machine learning model with a lightweight generative AI model from Hugging Face to create an intelligent chatbot. The goal is to **predict loan eligibility** and provide a **natural language explanation** based on official loan policies.

---

## 📌 Project Overview

🎯 **Business Case**:  
Dream Housing Finance wants to automate the home loan eligibility check and explain the decision clearly to users.

🔧 **Solution**:
- **ML Model** (XGBoost) trained on customer application data.
- **Hugging Face Model** (`google/flan-t5-base`) used to generate human-readable explanations using loan policies.
- **Streamlit UI** to make it interactive.
- 💡 No OpenAI, no internet needed after setup (fully local).

---

## 🗂️ Folder Structure

RAG/
├── app.py ← Streamlit chatbot app
├── train_model.py ← Script to train and save ML model + encoders
├── utils.py ← ML prediction + preprocessing functions
├── loan_eligibility_model.pkl ← Trained XGBoost model
├── encoders.pkl ← LabelEncoders for each categorical column
├── data/
│ ├── train.csv ← Training dataset
│ ├── test.csv ← Optional: test dataset
│ └── policy.txt ← Loan policy used for GenAI explanation

## 🚀 How It Works

### 1. User enters loan application info in a simple form.
### 2. ML model predicts whether the user is eligible.
### 3. If eligible or not, a local Hugging Face model explains the decision using loan policy context.

---

## 🧪 Technologies Used

| Component         | Tech                                |
|------------------|--------------------------------------|
| ML Model         | `XGBoost`                            |
| GenAI Model      | `google/flan-t5-base` from 🤗         |
| UI               | `Streamlit`                          |
| NLP Libraries    | `transformers`, `torch`              |
| Preprocessing    | `scikit-learn`, `LabelEncoder`       |

## ✅ Installation & Setup

### 1. Clone or Download the Repository
```bash
git clone https://github.com/yourusername/RAG.git
cd RAG

2. Create a Virtual Environment (Recommended)
bash
Copy
Edit
python -m venv venv
venv\Scripts\activate  # On Windows

3. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
If no requirements.txt, just run:

bash
Copy
Edit
pip install streamlit transformers torch sentence-transformers scikit-learn joblib xgboost pandas

🧠 Train the ML Model
bash
Copy
Edit
python train_model.py
This generates:

loan_eligibility_model.pkl

encoders.pkl

💬 Run the Chatbot
bash
Copy
Edit
streamlit run app.py
Open your browser at: http://localhost:8501
http://10.62.94.115:8501


📘 Sample policy.txt Format
txt
Copy
Edit
Dream Housing Finance provides loans to salaried and self-employed individuals.
Applicants with a credit history and monthly income above ₹25,000 are preferred.
Loan amount should not exceed 60% of net monthly income.
Married applicants with fewer dependents and stable jobs are prioritized.
📸 Screenshots
Add screenshots of the Streamlit app output, form, and explanation block here.

📄 License
This project is licensed under the MIT License – feel free to use, improve, and share.

🙌 Acknowledgements
Hugging Face Transformers

Streamlit

XGBoost



### ✅ How to Use:

1. Save this as `README.md` inside your `RAG/` folder.
2. You can optionally upload this project to GitHub to share it publicly.

Would you like a matching `requirements.txt` file too?
