"""BatteryVolt AI: an educational front end for saved Magpie voltage regressors."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pymatgen.core import Composition
from matminer.featurizers.composition import ElementProperty

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "models"

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
    return {el.symbol: float(amt) / non_li for el, amt in comp.items() if el.symbol != "Li"}, float(comp.get("Li", 0)) / non_li


def validate_pair(charged, discharged):
    try:
        c, d = Composition(charged), Composition(discharged)
    except Exception as exc:
        raise ValueError("Please enter valid chemical formulas, such as FePO4 and LiFePO4.") from exc
    host_c, li_c = normalize_non_lithium(c)
    host_d, li_d = normalize_non_lithium(d)
    if set(host_c) != set(host_d) or any(not np.isclose(host_c[k], host_d[k], rtol=0.01, atol=0.01) for k in host_c):
        raise ValueError("The non-lithium composition must stay the same between the charged and discharged formulas. This version does not support conversion reactions.")
    insertion = li_d - li_c
    if insertion <= 0:
        raise ValueError("The discharged formula must contain more lithium per non-lithium atom than the charged formula.")
    if insertion > 5:
        raise ValueError("This lithium change is unusually large; please check the formulas.")
    return c, insertion


def predict(charged, discharged):
    charge_comp, insertion = validate_pair(charged, discharged)
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
    return float(rf.predict(features)[0]), float(xgb.predict(features)[0]), insertion

st.title("🔋 BatteryVolt AI")
st.write("Explore estimated lithium-battery electrode voltages — no coding or machine-learning knowledge needed.")
st.info("**What is being predicted?** The estimated *average voltage* for a lithium insertion/extraction material pair, as learned from the model’s training data — not the exact voltage of a complete battery.")

mode = st.radio("Choose your experience", ["Beginner — choose an example", "Researcher — enter formulas"], horizontal=True)
if mode.startswith("Beginner"):
    material = st.selectbox("Choose a material", list(EXAMPLES))
    charged, discharged = EXAMPLES[material]
    st.caption(f"The app will use {charged} (charged state) → {discharged} (discharged state).")
    st.write("**Charged** means the host contains less lithium; **discharged** means it contains more lithium.")
else:
    st.write("Provide the same host material in a lithium-poor and lithium-rich state. The model uses the **charged-state formula** for Magpie features.")
    charged = st.text_input("Charged-state (lithium-poor) formula", value="FePO4", help="Example: FePO4")
    discharged = st.text_input("Discharged-state (lithium-rich) formula", value="LiFePO4", help="Example: LiFePO4")

if st.button("Estimate voltage", type="primary", use_container_width=True):
    try:
        with st.spinner("Calculating chemical features and estimating voltage…"):
            rf_voltage, xgb_voltage, li_change = predict(charged.strip(), discharged.strip())
        st.subheader("Estimated average voltage")
        left, right = st.columns(2)
        left.metric("Random Forest", f"{rf_voltage:.2f} V")
        right.metric("XGBoost", f"{xgb_voltage:.2f} V")
        difference = abs(rf_voltage - xgb_voltage)
        st.write(f"**How to read this:** The two models estimate an average voltage of {rf_voltage:.2f} V and {xgb_voltage:.2f} V for the selected change in lithium content.")
        st.caption(f"Lithium change (normalized): {li_change:.3f} | Difference between models: {difference:.2f} V")
        if difference >= 0.5:
            st.warning("The models disagree considerably. Treat this prediction as especially uncertain.")
        st.warning("**Research estimate, not a guarantee.** Chemistry and structure, electrochemical conditions, and the reference electrode all affect real voltage. Model agreement alone is not a validated uncertainty interval. Inputs outside the training distribution may be unreliable.")
        with st.expander("How does the AI make this estimate?"):
            st.markdown("1. Reads the charged material's chemical composition.\n2. Calculates **132 Magpie features** describing its elements.\n3. Calculates how much lithium changes between the two formulas.\n4. Applies the same saved feature selector used during training, keeping **127 features**.\n5. Uses two saved models to estimate average voltage.")
        # CSV export temporarily disabled while diagnosing prediction compatibility.
    except (ValueError, RuntimeError) as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error("The prediction failed. The technical details below will help identify the exact cause.")
        st.exception(exc)

st.divider()
with st.expander("About model quality and limitations"):
    st.write("Previous held-out evaluation results (as supplied by the project author):")
    st.dataframe(pd.DataFrame({"Model":["Random Forest", "XGBoost"], "MAE (V)":[0.4426,0.4389], "RMSE (V)":[0.5932,0.6075], "R²":[0.6409,0.6234]}), hide_index=True)
    st.caption("These are historical evaluation metrics, not uncertainty bounds or new tests performed by this website. The app is intended for exploratory screening only, not battery safety or design decisions. Performance on example or unseen chemistry has not been independently confirmed.")
