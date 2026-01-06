import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import sys
import catboost
import os
from catboost import CatBoostClassifier, Pool
import json
#st.write("Python:", sys.version)
#st.write("CatBoost:", catboost.__version__)
st.markdown(
    """
    <style>
    /* Main app background */
    .stApp {
        background: linear-gradient(
            135deg,
            #e3f2fd 0%,
            #f1f8ff 40%,
            #e8f5e9 100%
        );
    }

    /* Make content containers look like cards */
    .block-container {
        background-color: rgba(255, 255, 255, 0.85);
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0px 10px 30px rgba(0, 0, 0, 0.08);
    }

    /* Title styling */
    h1, h2, h3 {
        color: #0d47a1;
    }

    /* Buttons */
    .stButton > button {
        background-color: #1976d2;
        color: white;
        border-radius: 10px;
        padding: 0.5rem 1.5rem;
        font-size: 16px;
        border: none;
    }

    .stButton > button:hover {
        background-color: #0d47a1;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Operational Capacity Prediction",
    page_icon="🏥",
    layout="centered"
)



# ============================================================
# LOAD FEATURE METADATA (CRITICAL)
# ============================================================

BASE_DIR = os.path.dirname(__file__)

with open(os.path.join(BASE_DIR, "feature_metadata.json"), "r") as f:
    metadata = json.load(f)

FEATURE_COLUMNS = metadata["feature_columns"]
CAT_FEATURES = metadata["cat_features"]



# ============================================================
# LOAD MODEL
# ============================================================

#@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    import os

    MODEL_PATH = os.path.join(
     os.path.dirname(__file__),
     "catboost_operational_capacity.cbm"
)

    model.load_model(MODEL_PATH)

    return model

model = load_model()

# ============================================================
# APP TITLE
# ============================================================

st.title("🏥 Health Facility Operational Capacity Predictor")
st.markdown(
    """
    This tool estimates the probability that a health facility achieves  
    **operational capacity ≥ 80%**, based on service availability and facility characteristics.
    """
)

st.divider()

# ============================================================
# USER INPUTS
# ============================================================

st.subheader("Facility Information")

health_district = st.selectbox(
    "Health District",
    ["Biyem-assi", "Etoug-Ebe", "Mvog-Betsi", "Other"]
)

category_fosa = st.selectbox(
    "Category of FOSA Manager",
    ["4", "5", "6"]
)

status_fosa = st.selectbox(
    "Status of FOSA",
    ["Audience", "Confessional prvate", "Private secular"]
)
gender_fosa_manager = st.radio(
    "Gender of FOSA Manager",
    ["Male", "Feminine"]
)

qualification_fosa_manager = st.selectbox(
    "Qualification of FOSA Manager",
    ["IDE", "MD", "TMS", "Other"]
)


qualification_respondent = st.selectbox(
    "Qualification of Respondent",
    ["IDE", "MD", "TMS", "Other"]
)

gender_respondent = st.radio(
    "Gender of Respondent",
    ["Male", "Feminine"]
)
st.subheader("Service Availability")

def yes_no(label):
    return st.radio(label, ["Yes", "No"], horizontal=True)

childbirth_services = yes_no("Childbirth services available")
pmtct_services = yes_no("PMTCT services available")
prenatal_care_services = yes_no("Prenatal care services available")
vaccination_services = st.selectbox(
    "Vaccination services",
    ["Yes, declared", "No"]
)
family_planning_services = yes_no("Family planning services available")
blood_transfusion = yes_no("Blood transfusion services available")

# ============================================================
# PREDICTION
# ============================================================

if st.button("🔍 Predict Operational Capacity"):
    
    user_inputs = {
        "health_district": health_district,
        "category_fosa":category_fosa,
        "status_fosa":status_fosa,
        "gender_fosa_manager": gender_fosa_manager,
        "qualification_fosa_manager": qualification_fosa_manager,
        "qualification_respondent":qualification_respondent,
        "gender_respondent": gender_respondent,
        "childbirth_services": childbirth_services,
        "pmtct_services": pmtct_services,
        "prenatal_care_services": prenatal_care_services,
        "vaccination_services": vaccination_services,
        "family_planning_services": family_planning_services,
        "blood_transfusion": blood_transfusion,
    }


     # 🚨 CRITICAL: enforce SAME order as training
    input_data = pd.DataFrame(
        [[user_inputs[col] for col in FEATURE_COLUMNS]],
        columns=FEATURE_COLUMNS
    )

     # CatBoost expects strings for categorical vars
    input_data = input_data.fillna("Missing").astype(str)

    pool = Pool(
        input_data,
        cat_features=CAT_FEATURES
    )
    prob = model.predict_proba(pool)[0, 1]
    pred = model.predict(pool)[0]

    st.divider()
    st.subheader("Prediction Result")

    # ================= VISUAL OUTPUT =================

    st.metric(
        label="Probability of Operational Capacity ≥ 80%",
        value=f"{prob*100:.1f}%"
    )

    # Risk category
    if prob >= 0.5:
        st.success("✅ High operational capacity predicted (≥80%)")
    else:
        st.error("⚠️ Low operational capacity predicted (<80%)")

    # Probability bar
    fig, ax = plt.subplots(figsize=(6, 1))
    ax.barh([0], [prob], color="#2ecc71" if prob >= 0.5 else "#e74c3c")
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Probability")
    ax.set_title("Predicted Probability")

    st.pyplot(fig)

# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption(
    "Model: CatBoost | Categorical features handled natively | Developed for academic research"
)

st.markdown("""
    <hr>
    <p style='text-align: center; font-size:16px;'>
        <b>DISCLAIMER</b><br>
         This application is without any warranty. No author accepts responsibility to anyone for the consequences of using it or for whether it serves any particular purpose or works at all
    </p>
""", unsafe_allow_html=True)
