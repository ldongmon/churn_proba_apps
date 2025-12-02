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
        # Essayer de charger le modèle avec compatibilité
        import sklearn
        st.sidebar.info(f"Scikit-learn version: {sklearn.__version__}")
        
        # Définir la classe manquante pour la compatibilité
        try:
            from sklearn.compose._column_transformer import _RemainderColsList
        except ImportError:
            # Créer une classe factice pour la compatibilité
            class _RemainderColsList(list):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
            
            # Ajouter à sklearn.compose._column_transformer
            import sklearn.compose._column_transformer
            sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList
        
        # Charger le modèle
        model = joblib.load("churn_model.pkl")
        st.sidebar.success("✅ Modèle chargé avec succès!")
        return model, True, "Modèle original"
        
    except Exception as e:
        st.sidebar.error(f"⚠️ Erreur de chargement: {str(e)[:100]}...")
        
        # Option 1: Créer un modèle de démonstration simple
        st.sidebar.info("Création d'un modèle de démonstration...")
        
        # Créer un pipeline de démonstration similaire à ce que le modèle original pourrait être
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
        
        # Créer un pipeline complet
        demo_model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=50,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            ))
        ])
        
        # Entraîner avec des données factices
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
        
        y_demo = np.random.binomial(1, 0.3, n_samples)  # 30% de churn
        
        demo_model.fit(X_demo, y_demo)
        
        return demo_model, False, "Modèle de démonstration"

# Charger le modèle
model, model_loaded, model_type = load_model()

# =========================
# HEADER
# =========================
st.title("📉 Customer Churn Probability App")

# Afficher le type de modèle
if model_type == "Modèle de démonstration":
    st.markdown(f"""
    <div class="warning-box">
        <strong>⚠️ MODE DÉMONSTRATION</strong><br>
        L'application utilise un modèle de démonstration.<br>
        <small>Raison: Incompatibilité de version scikit-learn (1.6.1 → 1.7.2)</small>
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

# Instructions pour le format de données
with st.expander("📋 Format de données requis"):
    st.markdown("""
    **Colonnes recommandées (si vous utilisez le modèle de démonstration):**
    - `tenure`: Durée en mois (numérique)
    - `MonthlyCharges`: Frais mensuels (numérique)
    - `TotalCharges`: Frais totaux (numérique)
    - `Contract`: Type de contrat (catégoriel)
    - `InternetService`: Service internet (catégoriel)
    - `PaymentMethod`: Méthode de paiement (catégoriel)
    
    **Note:** Si vous utilisez votre propre modèle, assurez-vous que les colonnes
    correspondent exactement à celles utilisées pendant l'entraînement.
    """)

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        
        st.subheader("🔍 Data Preview")
        st.dataframe(df.head())
        st.info(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
        
        if df.empty:
            st.error("Le fichier CSV est vide. Veuillez uploader un fichier contenant des données.")
        else:
            try:
                # Vérifier les colonnes requises pour le modèle de démonstration
                if model_type == "Modèle de démonstration":
                    demo_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 
                                'Contract', 'InternetService', 'PaymentMethod']
                    missing_cols = [col for col in demo_cols if col not in df.columns]
                    if missing_cols:
                        st.warning(f"Colonnes manquantes pour le modèle de démo: {missing_cols}")
                        st.info("Le modèle fonctionnera avec les colonnes disponibles, mais les résultats peuvent être moins précis.")
                
                # Prédiction
                with st.spinner("Calcul des probabilités de churn..."):
                    try:
                        predictions = model.predict_proba(df)[:, 1]
                        df["Churn_Probability"] = predictions
                        
                        st.success("✅ Prédictions terminées!")
                        
                        # Afficher les résultats
                        st.subheader("📈 Churn Predictions")
                        st.dataframe(df[["Churn_Probability"] + list(df.columns[:-1])].head(20))
                        
                        # Statistiques
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            at_risk = len(df[df["Churn_Probability"] > 0.5])
                            st.metric("Clients à risque (>50%)", at_risk)
                        with col2:
                            avg_risk = df["Churn_Probability"].mean()
                            st.metric("Risque moyen", f"{avg_risk:.2%}")
                        with col3:
                            max_risk = df["Churn_Probability"].max()
                            st.metric("Risque maximum", f"{max_risk:.2%}")
                        
                        # Distribution des risques
                        st.subheader("📊 Distribution des risques")
                        hist_values = np.histogram(df["Churn_Probability"], bins=20, range=(0, 1))[0]
                        st.bar_chart(pd.DataFrame({"count": hist_values}))
                        
                        # Top 10 clients à risque
                        st.subheader("🚨 Top 10 Clients à Haut Risque")
                        top10 = df.sort_values("Churn_Probability", ascending=False).head(10)
                        st.dataframe(top10[["Churn_Probability"] + list(df.columns[:-1])])
                        
                        # Bouton de téléchargement
                        csv_data = df.to_csv(index=False)
                        st.download_button(
                            "⬇️ Télécharger les prédictions (CSV)",
                            csv_data,
                            file_name="churn_predictions.csv",
                            mime="text/csv",
                            help="Téléchargez toutes les prédictions avec les probabilités de churn"
                        )
                        
                    except Exception as e:
                        st.error(f"Erreur lors de la prédiction: {str(e)}")
                        st.info("""
                        **Solutions possibles:**
                        1. Vérifiez que vos données ont le bon format
                        2. Assurez-vous que les colonnes correspondent au modèle
                        3. Essayez avec moins de données
                        """)
                        
            except Exception as e:
                st.error(f"Erreur: {str(e)}")
                
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier CSV: {str(e)}")
        st.info("Assurez-vous que le fichier est un CSV valide et correctement formaté.")

# =========================
# QUICK TEST SECTION
# =========================
st.header("🧪 Test Rapide")
with st.expander("Tester avec des données exemple"):
    if st.button("Générer des données de test"):
        # Créer des données de test
        test_data = pd.DataFrame({
            'tenure': [1, 12, 24, 36, 48],
            'MonthlyCharges': [29.85, 56.95, 89.99, 45.30, 75.50],
            'TotalCharges': [29.85, 683.40, 2159.76, 1630.80, 3624.00],
            'Contract': ['Month-to-month', 'One year', 'Two years', 'Month-to-month', 'One year'],
            'InternetService': ['DSL', 'Fiber optic', 'Fiber optic', 'DSL', 'Fiber optic'],
            'PaymentMethod': ['Electronic check', 'Bank transfer', 'Credit card', 'Mailed check', 'Bank transfer']
        })
        
        st.write("Données de test générées:")
        st.dataframe(test_data)
        
        # Faire des prédictions
        try:
            predictions = model.predict_proba(test_data)[:, 1]
            test_data["Churn_Probability"] = predictions
            st.write("Résultats des prédictions:")
            st.dataframe(test_data)
        except Exception as e:
            st.error(f"Erreur lors du test: {str(e)}")

# =========================
# SIDEBAR
# =========================
st.sidebar.header("⚙️ Configuration")
st.sidebar.markdown(f"**Type de modèle:** {model_type}")

if st.sidebar.checkbox("Afficher les informations techniques"):
    st.sidebar.write("**Détails du modèle:**")
    st.sidebar.write(f"- Type: {type(model)}")
    if hasattr(model, 'steps'):
        st.sidebar.write(f"- Étapes: {[name for name, _ in model.steps]}")
    
    st.sidebar.write("**Versions des bibliothèques:**")
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
1. **Préparez vos données** en CSV
2. **Upload le fichier** dans l'application
3. **Visualisez** les prédictions
4. **Téléchargez** les résultats

**Problèmes courants:**
- Format CSV incorrect
- Colonnes manquantes
- Données manquantes
""")

# =========================
# FOOTER
# =========================
st.markdown("""
<div class="footer">
    Créé avec ❤️ par <strong>Leprince Dongmo</strong> — Propulsé par Machine Learning<br>
    <small>Pour résoudre l'erreur de compatibilité, utilisez scikit-learn==1.6.1</small>
</div>
""", unsafe_allow_html=True)