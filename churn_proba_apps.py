import streamlit as st
import pandas as pd
import joblib

# =========================
# APP CONFIGURATION
# =========================
st.set_page_config(
    page_title="Leprince Dongmo's Churn Prediction App",
    page_icon="📉",
    layout="wide",
)

# =========================
# CUSTOM STYLE (CSS)
# =========================
st.markdown("""
<style>
/* Main title color */
h1 {
    color: #1f4e79;
    text-align: center;
}

/* Subtitles */
h2, h3 {
    color: #2e75b6;
}

/* Custom welcome message */
.welcome-box {
    padding: 15px;
    background-color: #e8f1fa;
    border-radius: 10px;
    border-left: 6px solid #2e75b6;
    margin-bottom: 20px;
}

/* Footer credit */
.footer {
    text-align: center;
    margin-top: 50px;
    padding-top: 10px;
    font-size: 14px;
    color: gray;
}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
model = joblib.load("churn_model.pkl")

# =========================
# HEADER
# =========================
st.title("📉 Customer Churn Probability App")
st.markdown("""
<div class="welcome-box">
    <h3>👋 Welcome to the Churn Probability Platform</h3>
    This application helps businesses predict the likelihood that a customer will churn.<br>
    Upload your dataset and instantly receive churn probability scores for each customer!
</div>
""", unsafe_allow_html=True)

# =========================
# FILE UPLOAD SECTION
# =========================
st.header("📂 Upload Customer Dataset")
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("🔍 Data Preview")
    st.dataframe(df.head())

    # Prediction
    predictions = model.predict_proba(df)[:, 1]
    df["Churn_Probability"] = predictions

    st.subheader("📈 Churn Predictions")
    st.dataframe(df)

    # Highlight the highest-risk customers
    st.subheader("🚨 Top 10 High-Risk Customers")
    top10 = df.sort_values("Churn_Probability", ascending=False).head(10)
    st.dataframe(top10)

    # Export button
    st.download_button(
        "⬇️ Download Predictions as CSV",
        df.to_csv(index=False),
        file_name="churn_predictions.csv",
        mime="text/csv"
    )

# =========================
# MANUAL INPUT (OPTIONAL)
# =========================
st.header("🔧 Manual Input Section (Coming Soon)")
st.info("This section will allow you to enter a customer profile manually.")

# =========================
# FOOTER
# =========================
st.markdown("""
<div class="footer">
    Created with ❤️ by <strong>Leprince Dongmo</strong> — Powered by Machine Learning  
</div>
""", unsafe_allow_html=True)
