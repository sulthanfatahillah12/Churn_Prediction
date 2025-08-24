import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Bank Churn Predictor", page_icon="🏦", layout="wide")

# === Title & Intro ===
st.markdown("<h1 style='text-align:center;color:#2E86C1;'> Bank Customer Churn Predictor</h1>", unsafe_allow_html=True)
st.write("Masukkan data profil dan keuangan nasabah, lalu sistem akan memprediksi apakah pelanggan akan churn atau tetap loyal.")

# === Load Model ===
model = joblib.load("final_model_churnn.joblib")

def get_prediction(data: pd.DataFrame):
    pred = model.predict(data)
    pred_proba = model.predict_proba(data)
    return pred, pred_proba

# === Tabs untuk Input ===
tab1, tab2 = st.tabs([" Customer Profile", " Financial Info"])

with tab1:
    surname = st.text_input("Customer Surname (optional)", "")
    age = st.slider("Age", 18, 100, 30)
    gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
    geography = st.selectbox("Country", ["France", "Germany", "Spain"])

with tab2:
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
    balance = st.number_input("Account Balance (USD)", min_value=0.0, value=50000.0, step=100.0)
    estimated_salary = st.number_input("Estimated Salary (USD)", min_value=0.0, value=60000.0, step=100.0)
    tenure = st.select_slider("Bank Tenure (Years)", options=list(range(0, 11)), value=5)
    num_of_products = st.radio("Number of Products", [1, 2, 3, 4], horizontal=True)
    has_cr_card = st.checkbox("Has Credit Card?")
    is_active_member = st.checkbox("Active Member?")

# === Data Preparation ===
input_data = {
    "Surname": [surname.strip() if surname.strip() else "Not Provided"],
    "CreditScore": [credit_score],
    "Geography": [geography],
    "Gender": [gender],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [1 if has_cr_card else 0],
    "IsActiveMember": [1 if is_active_member else 0],
    "EstimatedSalary": [estimated_salary]
}
data = pd.DataFrame(input_data)

# === Preview Data ===
st.markdown("###  Input Data Preview")
st.dataframe(data.style.highlight_max(axis=0, color="lightblue"))

# === Prediction Button ===
st.markdown("<hr>", unsafe_allow_html=True)
predict_btn = st.button("Run Prediction")

if predict_btn:
    data_for_pred = data.drop(columns=["Surname"])
    pred, pred_proba = get_prediction(data_for_pred)

    label_map = {0: "LOYAL", 1: "CHURN"}
    label_pred = label_map[pred[0]]
    proba_loyal = pred_proba[0][0]
    proba_churn = pred_proba[0][1]

    # === Hasil Prediksi ===
    st.markdown("### 🎯 Prediction Result")
    if pred[0] == 1:
        st.markdown(f"""
        <div style="background-color:#FADBD8;padding:15px;border-radius:10px;text-align:center">
        <h3 style="color:#C0392B"> Customer likely to <b>CHURN</b></h3>
        <p><i>Name: {data['Surname'][0]}</i></p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background-color:#D5F5E3;padding:15px;border-radius:10px;text-align:center">
        <h3 style="color:#27AE60"> Customer likely to remain <b>LOYAL</b></h3>
        <p><i>Name: {data['Surname'][0]}</i></p>
        </div>
        """, unsafe_allow_html=True)

    # === Probabilities ===
    st.markdown("###  Prediction Probabilities")
    st.write(f"**Loyal Probability:** {proba_loyal:.1%}")
    st.progress(int(proba_loyal * 100))

    st.write(f"**Churn Probability:** {proba_churn:.1%}")
    st.progress(int(proba_churn * 100))