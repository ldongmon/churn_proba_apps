import streamlit as st
import pandas as pd
import joblib
import numpy as np
import sklearn

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

/* Warning box */
.warning-box {
    padding: 15px;
    background-color: #fff3cd;
    border-radius: 10px;
    border-left: 6px solid #ffc107;
    margin-bottom: 20px;
    color: #856404;
}

/* Error box */
.error-box {
    padding: 15px;
    background-color: #f8d7da;
    border-radius: 10px;
    border-left: 6px solid #dc3545;
    margin-bottom: 20px;
    color: #721c24;
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
# MODEL CONTRACT
# =========================
# Columns the Random Forest pipeline was trained on (Kaggle "Bank Customer Churn" dataset).
# Extra columns in the uploaded file (CustomerId, Surname, RowNumber, churn, ...) are ignored.
FEATURES = [
    "CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance",
    "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary",
    "Satisfaction Score", "Card Type", "Point Earned",
]

# Single reference definition of the risk bands (same as the modelling notebook)
RISK_BANDS = [(0.20, "Low risk (0-20%)"),
              (0.40, "Moderate (20-40%)"),
              (0.70, "High risk (40-70%)"),
              (1.01, "Critical (70-100%)")]


def risk_level(p):
    for cut, label in RISK_BANDS:
        if p < cut:
            return label


# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return joblib.load("churn_model.pkl")


try:
    model = load_model()
    features = list(getattr(model, "feature_names_in_", FEATURES))
    st.sidebar.success("✅ Model loaded successfully!")
except Exception as e:
    st.title("📉 Customer Churn Probability App")
    st.markdown(f"""
    <div class="error-box">
        <strong>⚠️ The model could not be loaded.</strong><br>
        <small>{str(e)[:200]}</small><br>
        <small>Installed scikit-learn: {sklearn.__version__}. The model was saved with the version pinned in requirements.txt.</small>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


def score(df):
    """Return a copy of df with Churn_Probability and Risk_Level, or raise ValueError."""
    missing = [col for col in features if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    out = df.copy()
    out["Churn_Probability"] = model.predict_proba(df[features])[:, 1]
    out["Risk_Level"] = out["Churn_Probability"].apply(risk_level)
    return out


# =========================
# HEADER
# =========================
st.title("📉 Customer Churn Probability App")

st.markdown("""
<div class="welcome-box">
    <h3>👋 Welcome to the Churn Probability Platform</h3>
    This application scores bank customers with a Random Forest model and ranks them by churn risk.<br>
    Upload your dataset and instantly receive a churn probability and a risk band for each customer!
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="warning-box">
    <strong>How to read the score.</strong> On a held-out test set the model reaches a precision of 0.80
    and a recall of 0.47 on churners. The probabilities are not calibrated: they rank customers reliably,
    but they are not a literal percentage chance of leaving.
</div>
""", unsafe_allow_html=True)

# =========================
# FILE UPLOAD SECTION
# =========================
st.header("📂 Upload Customer Dataset")

# Instructions for data format
with st.expander("📋 Required Data Format"):
    st.markdown("""
    **Required columns** (same names as the Kaggle *Bank Customer Churn* dataset):
    - `CreditScore`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `EstimatedSalary`, `Point Earned` (numeric)
    - `HasCrCard`, `IsActiveMember` (0 or 1)
    - `Satisfaction Score` (1 to 5)
    - `Geography` (France, Spain, Germany)
    - `Gender` (Male, Female)
    - `Card Type` (DIAMOND, GOLD, SILVER, PLATINUM)

    **Note:** any other column (for example `CustomerId` or `Surname`) is kept in the output
    but is not used by the model.
    """)

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        st.subheader("🔍 Data Preview")
        st.dataframe(df.head())
        st.info(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")

        if df.empty:
            st.error("The CSV file is empty. Please upload a file containing data.")
        else:
            with st.spinner("Calculating churn probabilities..."):
                try:
                    df = score(df)
                    lead = ["Churn_Probability", "Risk_Level"]
                    others = [col for col in df.columns if col not in lead]

                    st.success("✅ Predictions completed!")

                    # Display results
                    st.subheader("📈 Churn Predictions")
                    st.dataframe(df[lead + others].head(20))

                    # Statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        at_risk = len(df[df["Churn_Probability"] > 0.5])
                        st.metric("At-risk Customers (>50%)", at_risk)
                    with col2:
                        avg_risk = df["Churn_Probability"].mean()
                        st.metric("Average Score", f"{avg_risk:.2f}")
                    with col3:
                        max_risk = df["Churn_Probability"].max()
                        st.metric("Maximum Score", f"{max_risk:.2f}")

                    # Risk bands
                    st.subheader("📊 Risk Bands")
                    labels = [label for _, label in RISK_BANDS]
                    counts = df["Risk_Level"].value_counts().reindex(labels, fill_value=0)
                    st.dataframe(pd.DataFrame({
                        "Customers": counts,
                        "Share (%)": (counts / len(df) * 100).round(1),
                    }))

                    # Risk distribution
                    st.subheader("📊 Score Distribution")
                    hist_values = np.histogram(df["Churn_Probability"], bins=20, range=(0, 1))[0]
                    st.bar_chart(pd.DataFrame(
                        {"count": hist_values},
                        index=[f"{i / 20:.2f}" for i in range(20)],
                    ))

                    # Top 10 high-risk customers
                    st.subheader("🚨 Top 10 High-Risk Customers")
                    top10 = df.sort_values("Churn_Probability", ascending=False).head(10)
                    st.dataframe(top10[lead + others])

                    # Download button
                    csv_data = df.to_csv(index=False)
                    st.download_button(
                        "⬇️ Download Predictions (CSV)",
                        csv_data,
                        file_name="churn_predictions.csv",
                        mime="text/csv",
                        help="Download all predictions with churn probabilities"
                    )

                except ValueError as e:
                    st.error(f"Prediction error: {str(e)}")
                    st.info("Check the required columns in the 'Required Data Format' section above.")
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
                    st.info("""
                    **Possible solutions:**
                    1. Check that your data has the correct format
                    2. Ensure column values match the categories listed above
                    3. Remove rows with missing values
                    """)

    except Exception as e:
        st.error(f"Error reading CSV file: {str(e)}")
        st.info("Make sure the file is a valid and properly formatted CSV.")

# =========================
# QUICK TEST SECTION
# =========================
st.header("🧪 Quick Test")
with st.expander("Test with sample data"):
    if st.button("Generate test data"):
        # Five illustrative bank customers
        test_data = pd.DataFrame({
            "CreditScore": [619, 502, 850, 645, 376],
            "Geography": ["France", "France", "Spain", "Spain", "Germany"],
            "Gender": ["Female", "Female", "Female", "Male", "Female"],
            "Age": [42, 42, 43, 44, 29],
            "Tenure": [2, 8, 2, 8, 4],
            "Balance": [0.0, 159660.80, 125510.82, 113755.78, 115046.74],
            "NumOfProducts": [1, 3, 1, 2, 4],
            "HasCrCard": [1, 1, 1, 1, 1],
            "IsActiveMember": [1, 0, 1, 0, 0],
            "EstimatedSalary": [101348.88, 113931.57, 79084.10, 149756.71, 119346.88],
            "Satisfaction Score": [2, 3, 5, 5, 2],
            "Card Type": ["DIAMOND", "DIAMOND", "GOLD", "DIAMOND", "DIAMOND"],
            "Point Earned": [464, 377, 425, 484, 282],
        })

        st.write("Generated test data:")
        st.dataframe(test_data)

        # Make predictions
        try:
            st.write("Prediction results:")
            st.dataframe(score(test_data))
        except Exception as e:
            st.error(f"Test error: {str(e)}")

# =========================
# SIDEBAR
# =========================
st.sidebar.header("⚙️ Configuration")
st.sidebar.markdown(f"**Model:** {type(model.steps[-1][1]).__name__}")

if st.sidebar.checkbox("Show technical information"):
    st.sidebar.write("**Model Details:**")
    st.sidebar.write(f"- Steps: {[name for name, _ in model.steps]}")
    st.sidebar.write(f"- Features: {len(features)}")

    st.sidebar.write("**Library Versions:**")
    st.sidebar.write(f"- scikit-learn: {sklearn.__version__}")
    st.sidebar.write(f"- pandas: {pd.__version__}")
    st.sidebar.write(f"- numpy: {np.__version__}")

# Instructions
st.sidebar.header("ℹ️ Instructions")
st.sidebar.markdown("""
1. **Prepare your data** in CSV format
2. **Upload the file** to the application
3. **View** the predictions
4. **Download** the results

**Common issues:**
- Incorrect CSV format
- Missing columns
- Missing data
""")

# =========================
# FOOTER
# =========================
st.markdown("""
<div class="footer">
    Created with ❤️ by <strong>Leprince Dongmo</strong> — Powered by Machine Learning
</div>
""", unsafe_allow_html=True)
