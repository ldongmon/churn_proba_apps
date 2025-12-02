import streamlit as st
import pandas as pd
import joblib
import traceback
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

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
# LOAD MODEL WITH COMPATIBILITY FIX
# =========================
@st.cache_resource
def load_model():
    try:
        # Try to load model with compatibility
        import sklearn
        st.sidebar.info(f"Scikit-learn version: {sklearn.__version__}")
        
        # Define missing class for compatibility
        try:
            from sklearn.compose._column_transformer import _RemainderColsList
        except ImportError:
            # Create dummy class for compatibility
            class _RemainderColsList(list):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
            
            # Add to sklearn.compose._column_transformer
            import sklearn.compose._column_transformer
            sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList
        
        # Load the model
        model = joblib.load("churn_model.pkl")
        st.sidebar.success("✅ Model loaded successfully!")
        return model, True, "Original Model"
        
    except Exception as e:
        st.sidebar.error(f"⚠️ Loading error: {str(e)[:100]}...")
        
        # Option 1: Create a simple demonstration model
        st.sidebar.info("Creating a demonstration model...")
        
        # Create a demonstration pipeline similar to what the original model might be
        numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
        categorical_features = ['Contract', 'InternetService', 'PaymentMethod']
        
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        # Create a complete pipeline
        demo_model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=50,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            ))
        ])
        
        # Train with dummy data
        np.random.seed(42)
        n_samples = 100
        X_demo = pd.DataFrame({
            'tenure': np.random.randint(1, 72, n_samples),
            'MonthlyCharges': np.random.uniform(20, 120, n_samples),
            'TotalCharges': np.random.uniform(50, 8000, n_samples),
            'Contract': np.random.choice(['Month-to-month', 'One year', 'Two years'], n_samples),
            'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
            'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples)
        })
        
        y_demo = np.random.binomial(1, 0.3, n_samples)  # 30% churn rate
        
        demo_model.fit(X_demo, y_demo)
        
        return demo_model, False, "Demonstration Model"

# Load the model
model, model_loaded, model_type = load_model()

# =========================
# HEADER
# =========================
st.title("📉 Customer Churn Probability App")

# Display model type
if model_type == "Demonstration Model":
    st.markdown(f"""
    <div class="warning-box">
        <strong>⚠️ DEMONSTRATION MODE</strong><br>
        The application is using a demonstration model.<br>
        <small>Reason: Scikit-learn version incompatibility (1.6.1 → 1.7.2)</small>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
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

# Instructions for data format
with st.expander("📋 Required Data Format"):
    st.markdown("""
    **Recommended columns (if using the demonstration model):**
    - `tenure`: Duration in months (numeric)
    - `MonthlyCharges`: Monthly fees (numeric)
    - `TotalCharges`: Total fees (numeric)
    - `Contract`: Contract type (categorical)
    - `InternetService`: Internet service (categorical)
    - `PaymentMethod`: Payment method (categorical)
    
    **Note:** If using your own model, ensure the columns
    match exactly those used during training.
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
            try:
                # Check required columns for demonstration model
                if model_type == "Demonstration Model":
                    demo_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 
                                'Contract', 'InternetService', 'PaymentMethod']
                    missing_cols = [col for col in demo_cols if col not in df.columns]
                    if missing_cols:
                        st.warning(f"Missing columns for demo model: {missing_cols}")
                        st.info("The model will work with available columns, but results may be less accurate.")
                
                # Prediction
                with st.spinner("Calculating churn probabilities..."):
                    try:
                        predictions = model.predict_proba(df)[:, 1]
                        df["Churn_Probability"] = predictions
                        
                        st.success("✅ Predictions completed!")
                        
                        # Display results
                        st.subheader("📈 Churn Predictions")
                        st.dataframe(df[["Churn_Probability"] + list(df.columns[:-1])].head(20))
                        
                        # Statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            at_risk = len(df[df["Churn_Probability"] > 0.5])
                            st.metric("At-risk Customers (>50%)", at_risk)
                        with col2:
                            avg_risk = df["Churn_Probability"].mean()
                            st.metric("Average Risk", f"{avg_risk:.2%}")
                        with col3:
                            max_risk = df["Churn_Probability"].max()
                            st.metric("Maximum Risk", f"{max_risk:.2%}")
                        
                        # Risk distribution
                        st.subheader("📊 Risk Distribution")
                        hist_values = np.histogram(df["Churn_Probability"], bins=20, range=(0, 1))[0]
                        st.bar_chart(pd.DataFrame({"count": hist_values}))
                        
                        # Top 10 high-risk customers
                        st.subheader("🚨 Top 10 High-Risk Customers")
                        top10 = df.sort_values("Churn_Probability", ascending=False).head(10)
                        st.dataframe(top10[["Churn_Probability"] + list(df.columns[:-1])])
                        
                        # Download button
                        csv_data = df.to_csv(index=False)
                        st.download_button(
                            "⬇️ Download Predictions (CSV)",
                            csv_data,
                            file_name="churn_predictions.csv",
                            mime="text/csv",
                            help="Download all predictions with churn probabilities"
                        )
                        
                    except Exception as e:
                        st.error(f"Prediction error: {str(e)}")
                        st.info("""
                        **Possible solutions:**
                        1. Check that your data has the correct format
                        2. Ensure columns match the model
                        3. Try with less data
                        """)
                        
            except Exception as e:
                st.error(f"Error: {str(e)}")
                
    except Exception as e:
        st.error(f"Error reading CSV file: {str(e)}")
        st.info("Make sure the file is a valid and properly formatted CSV.")

# =========================
# QUICK TEST SECTION
# =========================
st.header("🧪 Quick Test")
with st.expander("Test with sample data"):
    if st.button("Generate test data"):
        # Create test data
        test_data = pd.DataFrame({
            'tenure': [1, 12, 24, 36, 48],
            'MonthlyCharges': [29.85, 56.95, 89.99, 45.30, 75.50],
            'TotalCharges': [29.85, 683.40, 2159.76, 1630.80, 3624.00],
            'Contract': ['Month-to-month', 'One year', 'Two years', 'Month-to-month', 'One year'],
            'InternetService': ['DSL', 'Fiber optic', 'Fiber optic', 'DSL', 'Fiber optic'],
            'PaymentMethod': ['Electronic check', 'Bank transfer', 'Credit card', 'Mailed check', 'Bank transfer']
        })
        
        st.write("Generated test data:")
        st.dataframe(test_data)
        
        # Make predictions
        try:
            predictions = model.predict_proba(test_data)[:, 1]
            test_data["Churn_Probability"] = predictions
            st.write("Prediction results:")
            st.dataframe(test_data)
        except Exception as e:
            st.error(f"Test error: {str(e)}")

# =========================
# SIDEBAR
# =========================
st.sidebar.header("⚙️ Configuration")
st.sidebar.markdown(f"**Model Type:** {model_type}")

if st.sidebar.checkbox("Show technical information"):
    st.sidebar.write("**Model Details:**")
    st.sidebar.write(f"- Type: {type(model)}")
    if hasattr(model, 'steps'):
        st.sidebar.write(f"- Steps: {[name for name, _ in model.steps]}")
    
    st.sidebar.write("**Library Versions:**")
    try:
        import sklearn, pandas, numpy
        st.sidebar.write(f"- scikit-learn: {sklearn.__version__}")
        st.sidebar.write(f"- pandas: {pandas.__version__}")
        st.sidebar.write(f"- numpy: {numpy.__version__}")
    except:
        pass

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
    Created with ❤️ by <strong>Leprince Dongmo</strong> — Powered by Machine Learning<br>
    <small>To fix compatibility error, use scikit-learn==1.6.1</small>
</div>
""", unsafe_allow_html=True)