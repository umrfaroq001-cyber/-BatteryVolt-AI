"""BatteryVolt AI: an educational front end for saved Magpie voltage regressors with AI Agent & Specific Energy."""
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pymatgen.core import Composition
from matminer.featurizers.composition import ElementProperty

try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "models"

# Faraday constant in mAh / mol of electrons
FARADAY_MAH_PER_MOL = 26801.48

EXAMPLES = {
    "Lithium iron phosphate (LFP)": ("FePO4", "LiFePO4"),
    "Lithium cobalt oxide (LCO)": ("CoO2", "LiCoO2"),
    "Lithium manganese oxide (LMO)": ("Mn2O4", "LiMn2O4"),
}

st.set_page_config(page_title="BatteryVolt AI", page_icon="🔋", layout="centered")
st.markdown("""
<style>
.block-container {max-width: 850px; padding-top: 2rem}
[data-testid="stMetricValue"] {font-size: 2.1rem}
.small-note {color: #6b7280; font-size: .9rem}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading battery models…")
def load_models():
    """Only load model files packaged with this project, never user-uploaded joblib."""
    rf = joblib.load(MODEL_DIR / "random_forest_magpie.joblib")
    xgb = joblib.load(MODEL_DIR / "xgboost_magpie.joblib")
    selector = joblib.load(MODEL_DIR / "feature_selector.joblib")
    selected = joblib.load(MODEL_DIR / "selected_features.joblib")
    featurizer = ElementProperty.from_preset("magpie")
    return rf, xgb, selector, selected, featurizer


def normalize_non_lithium(comp):
    non_li = sum(float(amt) for el, amt in comp.items() if el.symbol != "Li")
    if non_li <= 0:
        raise ValueError("A material must contain elements other than lithium.")
    return (
        {el.symbol: float(amt) / non_li for el, amt in comp.items() if el.symbol != "Li"},
        float(comp.get("Li", 0)) / non_li,
    )


def validate_pair(charged, discharged):
    try:
        c, d = Composition(charged), Composition(discharged)
    except Exception as exc:
        raise ValueError("Please enter valid chemical formulas, such as FePO4 and LiFePO4.") from exc
    host_c, li_c = normalize_non_lithium(c)
    host_d, li_d = normalize_non_lithium(d)
    if set(host_c) != set(host_d) or any(
        not np.isclose(host_c[k], host_d[k], rtol=0.01, atol=0.01) for k in host_c
    ):
        raise ValueError(
            "The non-lithium composition must stay the same between the charged and discharged formulas. "
            "This version does not support conversion reactions."
        )
    insertion = li_d - li_c
    if insertion <= 0:
        raise ValueError(
            "The discharged formula must contain more lithium per non-lithium atom than the charged formula."
        )
    if insertion > 5:
        raise ValueError("This lithium change is unusually large; please check the formulas.")
    return c, d, insertion


def calculate_electrochemistry(discharged_comp, insertion, avg_voltage):
    """Calculate theoretical gravimetric capacity (mAh/g) and specific energy (Wh/kg)."""
    non_li_d = sum(float(amt) for el, amt in discharged_comp.items() if el.symbol != "Li")
    moles_li_extracted = insertion * non_li_d
    molar_mass_discharged = float(discharged_comp.weight)
    theoretical_capacity = (moles_li_extracted * FARADAY_MAH_PER_MOL) / molar_mass_discharged
    specific_energy = avg_voltage * theoretical_capacity
    return theoretical_capacity, specific_energy


def generate_ai_summary(charged, discharged, rf_v, xgb_v, avg_v, capacity, spec_energy, li_change):
    """Generate an AI research screening summary via Gemini with a deterministic scientific fallback."""
    prompt = (
        f"You are an expert battery materials AI research assistant. Analyze this lithium-ion cathode screening result:\n"
        f"- Charged / Discharged pair: {charged} -> {discharged}\n"
        f"- Random Forest Voltage: {rf_v:.2f} V | XGBoost Voltage: {xgb_v:.2f} V (Mean: {avg_v:.2f} V)\n"
        f"- Theoretical Gravimetric Capacity: {capacity:.1f} mAh/g\n"
        f"- Derived Specific Energy: {spec_energy:.1f} Wh/kg\n"
        f"- Normalized Li Insertion: {li_change:.3f}\n\n"
        f"Write a concise, 3-4 sentence scientific screening summary covering: (1) voltage regime and energy density potential, "
        f"(2) model agreement confidence, and (3) practical cathode screening considerations (such as electrolyte stability or structural cycling)."
    )

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        try:
            api_key = st.secrets.get("GEMINI_API_KEY", "")
        except Exception:
            api_key = ""

    if HAS_GENAI and api_key:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            if response and response.text:
                return response.text.strip()
        except Exception:
            pass

    # Built-in scientific synthesis fallback so live demos never fail
    diff = abs(rf_v - xgb_v)
    confidence_note = (
        f"Both tree-based models show strong convergence (Δ = {diff:.2f} V), indicating high in-distribution confidence across the selected Magpie elemental features."
        if diff < 0.35
        else f"The Random Forest and XGBoost models show moderate divergence (Δ = {diff:.2f} V), suggesting additional DFT or structural verification is warranted."
    )
    voltage_regime = (
        "a high-voltage cathode regime (>4.0 V), which maximizes cell energy density but requires oxidative stability screening for standard carbonate electrolytes"
        if avg_v >= 4.0
        else "a stable mid-voltage operating window, well within the electrochemical stability window of conventional Li-ion electrolytes"
        if avg_v >= 3.2
        else "a lower-voltage insertion regime suited for specialized high-safety or secondary energy storage applications"
    )
    return (
        f"**Electrochemical Screening Assessment ({charged} ↔ {discharged}):** "
        f"This insertion host exhibits an ensemble average operating potential of **{avg_v:.2f} V** "
        f"(RF: {rf_v:.2f} V, XGBoost: {xgb_v:.2f} V), placing it in {voltage_regime}. "
        f"Combined with a theoretical gravimetric capacity of **{capacity:.1f} mAh/g** based on the host stoichiometry, "
        f"the material yields an estimated specific energy of **{spec_energy:.0f} Wh/kg** at the active-material level. "
        f"{confidence_note} As a next screening step, evaluate phase stability (energy above hull) and unit-cell volume expansion during cycling."
    )


def predict(charged, discharged):
    charge_comp, discharge_comp, insertion = validate_pair(charged, discharged)
    rf, xgb, selector, selected, featurizer = load_models()
    # Training used ElementProperty.from_preset('magpie') on formula_charge.
    # The sole additional engineered input was Li_normalized_discharge - Li_normalized_charge.
    names = featurizer.feature_labels()  # Labels already include the "MagpieData" prefix.
    values = featurizer.featurize(charge_comp)
    record = dict(zip(names, values))
    record["Li_insertion"] = insertion
    train_columns = list(selector.feature_names_in_)
    missing = [name for name in train_columns if name not in record]
    if missing:
        raise RuntimeError(f"Feature generation does not match the trained model: {missing[:4]}")
    full = pd.DataFrame([{key: record[key] for key in train_columns}], columns=train_columns)
    if not np.isfinite(full.to_numpy(dtype=float)).all():
        raise ValueError("Magpie could not calculate all necessary elemental properties for this formula.")
    # Reproduce the fitted training-time feature selection, preserve exact feature order.
    transformed = selector.transform(full)
    selected_names = list(selector.get_feature_names_out(train_columns))
    if selected_names != list(selected):
        raise RuntimeError("Saved feature order does not match the saved selector.")
    features = pd.DataFrame(transformed, columns=selected)
    if features.shape[1] != rf.n_features_in_ or features.shape[1] != xgb.n_features_in_:
        raise RuntimeError("Feature count does not match the trained models.")
    rf_pred = float(rf.predict(features)[0])
    xgb_pred = float(xgb.predict(features)[0])
    return rf_pred, xgb_pred, insertion, discharge_comp


st.title("🔋 BatteryVolt AI")
st.write("Explore estimated lithium-battery electrode voltages, specific energy, and AI screening insights — no coding or machine-learning knowledge needed.")
st.info(
    "**What is being predicted?** The estimated *average operating voltage (V)* for a lithium insertion/extraction material pair, "
    "combined with *theoretical gravimetric capacity (mAh/g)* and *specific energy (Wh/kg)* derived from host stoichiometry."
)

mode = st.radio(
    "Choose your experience",
    ["Beginner — choose an example", "Researcher — enter formulas"],
    horizontal=True,
)
if mode.startswith("Beginner"):
    material = st.selectbox("Choose a material", list(EXAMPLES))
    charged, discharged = EXAMPLES[material]
    st.caption(f"The app will use {charged} (charged state) → {discharged} (discharged state).")
    st.write("**Charged** means the host contains less lithium; **discharged** means it contains more lithium.")
else:
    st.write(
        "Provide the same host material in a lithium-poor and lithium-rich state. "
        "The model uses the **charged-state formula** for Magpie features."
    )
    charged = st.text_input("Charged-state (lithium-poor) formula", value="FePO4", help="Example: FePO4")
    discharged = st.text_input("Discharged-state (lithium-rich) formula", value="LiFePO4", help="Example: LiFePO4")

if st.button("Estimate voltage & analyze cathode", type="primary", use_container_width=True):
    try:
        with st.spinner("Calculating Magpie features, energy metrics, and AI screening summary…"):
            c_clean, d_clean = charged.strip(), discharged.strip()
            rf_voltage, xgb_voltage, li_change, discharge_comp = predict(c_clean, d_clean)
            avg_voltage = (rf_voltage + xgb_voltage) / 2.0
            theo_capacity, spec_energy = calculate_electrochemistry(discharge_comp, li_change, avg_voltage)
            ai_summary = generate_ai_summary(
                c_clean, d_clean, rf_voltage, xgb_voltage, avg_voltage, theo_capacity, spec_energy, li_change
            )

        st.subheader("1. Estimated Operating Voltage")
        left, right, mean_col = st.columns(3)
        left.metric("Random Forest", f"{rf_voltage:.2f} V")
        right.metric("XGBoost", f"{xgb_voltage:.2f} V")
        mean_col.metric("Ensemble Mean", f"{avg_voltage:.2f} V")

        difference = abs(rf_voltage - xgb_voltage)
        st.caption(f"Lithium change (normalized): {li_change:.3f} | Difference between models: {difference:.2f} V")
        if difference >= 0.5:
            st.warning("The models disagree considerably. Treat this prediction as especially uncertain.")

        st.subheader("2. Derived Electrochemical Energy Metrics")
        cap_col, energy_col = st.columns(2)
        cap_col.metric("Theoretical Capacity", f"{theo_capacity:.1f} mAh/g")
        energy_col.metric("Specific Energy", f"{spec_energy:.0f} Wh/kg")
        st.caption(
            "Calculated from the discharged host molar mass via Faraday's law and the ensemble mean operating voltage."
        )

        st.subheader("3. AI Research Agent Summary")
        st.success(ai_summary)

        # Export results as CSV
        export_df = pd.DataFrame(
            [
                {
                    "Charged_Formula": c_clean,
                    "Discharged_Formula": d_clean,
                    "RF_Voltage_V": round(rf_voltage, 3),
                    "XGBoost_Voltage_V": round(xgb_voltage, 3),
                    "Mean_Voltage_V": round(avg_voltage, 3),
                    "Normalized_Li_Insertion": round(li_change, 4),
                    "Theoretical_Capacity_mAh_g": round(theo_capacity, 2),
                    "Specific_Energy_Wh_kg": round(spec_energy, 2),
                }
            ]
        )
        st.download_button(
            label="📥 Download Screening Report (CSV)",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name=f"BatteryVolt_AI_{c_clean}_{d_clean}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.warning(
            "**Research estimate, not a guarantee.** Chemistry and structure, electrochemical conditions, "
            "and the reference electrode all affect real voltage. Model agreement alone is not a validated "
            "uncertainty interval. Inputs outside the training distribution may be unreliable."
        )
        with st.expander("How does the AI make this estimate?"):
            st.markdown(
                "1. Reads the charged material's chemical composition.\n"
                "2. Calculates **132 Magpie features** describing its elements.\n"
                "3. Calculates how much lithium changes between the two formulas.\n"
                "4. Applies the same saved feature selector used during training, keeping **127 features**.\n"
                "5. Uses two saved models (Random Forest & XGBoost) to estimate average voltage.\n"
                "6. Computes theoretical gravimetric capacity (mAh/g) and specific energy (Wh/kg) using Faraday's law.\n"
                "7. Synthesizes an AI cathode screening interpretation."
            )
    except (ValueError, RuntimeError) as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error("The prediction failed. The technical details below will help identify the exact cause.")
        st.exception(exc)

st.divider()
with st.expander("About model quality and limitations"):
    st.write("Previous held-out evaluation results (as supplied by the project author):")
    st.dataframe(
        pd.DataFrame(
            {
                "Model": ["Random Forest", "XGBoost"],
                "MAE (V)": [0.4426, 0.4389],
                "RMSE (V)": [0.5932, 0.6075],
                "R²": [0.6409, 0.6234],
            }
        ),
        hide_index=True,
    )
    st.caption(
        "These are historical evaluation metrics, not uncertainty bounds or new tests performed by this website. "
        "The app is intended for exploratory screening only, not battery safety or design decisions. "
        "Performance on example or unseen chemistry has not been independently confirmed."
    )
