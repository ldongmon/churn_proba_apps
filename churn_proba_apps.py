import streamlit as st
import pandas as pd
import joblib
import traceback
from sklearn.ensemble import RandomForestClassifier
import numpy as np

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
# LOAD MODEL WITH ERROR HANDLING
# =========================
@st.cache_resource
def load_model():
    try:
        model = joblib.load("churn_model.pkl")
        st.sidebar.success("✅ Modèle chargé avec succès!")
        return model, True
    except Exception as e:
        st.sidebar.warning("⚠️ Mode démo activé")
        st.sidebar.info("Le modèle principal n'a pas pu être chargé. Utilisation d'un modèle de démonstration.")
        
        # Créer un modèle factice pour la démo
        demo_model = RandomForestClassifier(n_estimators=10, random_state=42)
        
        # Entraîner avec des données factices (pour éviter les erreurs de prédiction)
        X_demo = np.random.randn(100, 10)  # 100 échantillons, 10 caractéristiques
        y_demo = np.random.randint(0, 2, 100)  # Labels binaires aléatoires
        demo_model.fit(X_demo, y_demo)
        
        return demo_model, False

# Charger le modèle
model, model_loaded = load_model()

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

# Afficher un avertissement si en mode démo
if not model_loaded:
    st.markdown("""
    <div class="warning-box">
        <strong>⚠️ ATTENTION : Mode Démonstration</strong><br>
        L'application fonctionne en mode démo avec un modèle factice. 
        Pour utiliser le modèle réel, assurez-vous que le fichier <code>churn_model.pkl</code> est présent et compatible.
    </div>
    """, unsafe_allow_html=True)

# =========================
# FILE UPLOAD SECTION
# =========================
st.header("📂 Upload Customer Dataset")
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        
        st.subheader("🔍 Data Preview")
        st.dataframe(df.head())
        
        # Vérifier si le dataframe n'est pas vide
        if df.empty:
            st.error("Le fichier CSV est vide. Veuillez uploader un fichier contenant des données.")
        else:
            # Vérifier les colonnes nécessaires (ajuster selon votre modèle)
            st.info(f"Dataset shape: {df.shape}")
            
            try:
                # Prédiction
                predictions = model.predict_proba(df)[:, 1]
                df["Churn_Probability"] = predictions
                
                st.subheader("📈 Churn Predictions")
                st.dataframe(df)
                
                # Highlight the highest-risk customers
                st.subheader("🚨 Top 10 High-Risk Customers")
                top10 = df.sort_values("Churn_Probability", ascending=False).head(10)
                st.dataframe(top10[["Churn_Probability"] + list(df.columns[:-1])])
                
                # Statistiques
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Customers at Risk (>0.5)", 
                             len(df[df["Churn_Probability"] > 0.5]))
                with col2:
                    st.metric("Average Risk", 
                             f"{df['Churn_Probability'].mean():.2%}")
                with col3:
                    st.metric("Highest Risk", 
                             f"{df['Churn_Probability'].max():.2%}")
                
                # Export button
                csv_data = df.to_csv(index=False)
                st.download_button(
                    "⬇️ Download Predictions as CSV",
                    csv_data,
                    file_name="churn_predictions.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"Erreur lors des prédictions: {str(e)}")
                st.info("Assurez-vous que votre dataset a le bon format (mêmes colonnes que le modèle d'entraînement).")
                
                # Afficher les colonnes disponibles pour debug
                st.write("Colonnes disponibles dans votre dataset:")
                st.write(list(df.columns))
                
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier: {str(e)}")

# =========================
# MANUAL INPUT (OPTIONAL)
# =========================
st.header("🔧 Manual Input Section")
with st.expander("Entrer manuellement les données d'un client"):
    st.info("Cette fonctionnalité est en développement. Pour l'instant, utilisez le upload de fichier CSV.")
    
    # Exemple de formulaire simple
    col1, col2 = st.columns(2)
    with col1:
        tenure = st.number_input("Tenure (mois)", min_value=0, max_value=100, value=12)
        monthly_charges = st.number_input("Charges mensuelles", min_value=0.0, value=50.0)
    with col2:
        contract = st.selectbox("Type de contrat", ["Month-to-month", "One year", "Two years"])
        internet_service = st.selectbox("Service internet", ["DSL", "Fiber optic", "No"])
    
    if st.button("Estimer la probabilité de churn"):
        st.warning("Fonctionnalité en développement - utilisez le upload CSV pour des prédictions complètes")

# =========================
# SIDEBAR INFO
# =========================
st.sidebar.header("ℹ️ Informations")
st.sidebar.info("""
**Instructions:**
1. Upload un fichier CSV avec les données clients
2. L'application calcule la probabilité de churn
3. Téléchargez les résultats

**Format attendu:**
- Données numériques/catégorielles
- Mêmes colonnes que le modèle d'entraînement
- Pas de valeurs manquantes
""")

# =========================
# DEBUG SECTION (optionnel - à désactiver en production)
# =========================
if st.sidebar.checkbox("Mode Debug"):
    st.sidebar.write("**Informations du modèle:**")
    st.sidebar.write(f"Type: {type(model)}")
    st.sidebar.write(f"Mode démo: {not model_loaded}")
    
    if hasattr(model, 'feature_importances_'):
        st.sidebar.write("Le modèle a des importances de caractéristiques")
    if hasattr(model, 'n_features_in_'):
        st.sidebar.write(f"Nombre de caractéristiques attendues: {model.n_features_in_}")

# =========================
# FOOTER
# =========================
st.markdown("""
<div class="footer">
    Created with ❤️ by <strong>Leprince Dongmo</strong> — Powered by Machine Learning  
</div>
""", unsafe_allow_html=True)