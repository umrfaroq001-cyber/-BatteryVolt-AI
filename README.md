## Contributors

- **[Umer Farooq](https://github.com/umrfaroq001-cyber)** — *Machine Learning & Feature Engineering*: Designed the 132 Magpie + `Li_insertion` feature pipeline, `VarianceThreshold` selection, and trained the Random Forest and XGBoost models.
- **[Ch Aazib](https://github.com/aazibtariq1)** — *Application & Backend Development*: Built the two-mode Streamlit interface (`app.py`), chemical formula validation, and safe `.joblib` model loading logic.
- **[Rabia Noor](https://github.com/rabia202128-dot)** — *Deployment, QA & Documentation*: Managed Python 3.11 dependencies (`requirements.txt`), Streamlit Community Cloud deployment, edge-case testing, and project documentation.



# BatteryVolt AI — beginner-friendly lithium electrode voltage estimation

A two-mode Streamlit application powered by the saved Random Forest and XGBoost models.

## Run locally

Use **Python 3.11** (recommended for dependency compatibility).

```bash
cd BatteryVolt_AI
python -m pip install -r requirements.txt
streamlit run app.py
```

The app loads four model artifacts from `models/`. Only load `.joblib` files you trust. Models serialized with scikit-learn 1.6.1 may not load reliably under other scikit-learn versions.

## Publish for everyone

1. Upload the **contents** of this folder (including `models/`) to a GitHub repository. Confirm you have permission to publish the trained models and any data-derived artifacts.
2. Visit https://share.streamlit.io/ and connect the repository.
3. Select `app.py` as the main file and deploy. For runtime consistency, use a compatible Python release (3.11 recommended).
4. Share the deployed link. Users do not need to install Python.

## How features are generated

Training used **Magpie features of the charged-state formula**, plus `Li_insertion`, the normalized lithium-per-non-lithium change from charged to discharged formula:

`Li_insertion = Li_discharge / nonLi_discharge - Li_charge / nonLi_charge`

132 Magpie + 1 lithium feature = 133 original features. The saved VarianceThreshold selector reduces them to 127 features. Both saved regressors then make predictions.

The app validates that the non-lithium host is approximately conserved: it is **not** designed for conversion reactions or non-lithium chemistries.

## Important limitations

- Predictions are exploratory, not certified measurements. Voltage depends on chemistry, structure, thermodynamic reference, and measurement conditions.
- Both model predictions are shown separately; agreement between models does not establish predictive confidence.
- Reported validation metrics are from the prior research workflow and have **not** been revalidated for this app.
- Out-of-distribution materials may lead to unreliable results.
- No untrusted uploaded model files are accepted by the web interface.
