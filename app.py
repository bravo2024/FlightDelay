"""app.py - FlightDelay: Flight Delay Prediction Dashboard.

A time series / tabular ML platform for predicting flight delays with:
- Cyclical temporal encoding (sin/cos for hours)
- Gradient Boosting for delay classification
- Permutation feature importance
- Hourly and route-level delay analysis
- Business cost optimization
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import streamlit as st
import json
from src.data import load_flights, fetch_nycflights, engineer_temporal_features
from src.model import (
    train_gradient_boosting, train_logistic_regression, evaluate,
    permutation_importance, temporal_analysis
)
from src.core import temporal_split, StandardScaler
from src.visualizations import (
    plot_delay_by_hour, plot_delay_distribution, plot_roc_curve,
    plot_feature_importance, plot_confusion_matrix, plot_metrics_comparison,
    plot_cost_analysis
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="FlightDelay | Flight Delay Prediction", layout="wide", page_icon="✈")

# ---------------------------------------------------------------------------
# CSS + Hero
# ---------------------------------------------------------------------------
st.markdown("""
<style>
.hero {
    padding: 1.4rem 1.6rem;
    border-radius: 1rem;
    background: linear-gradient(135deg, #0c4a6e 0%, #0369a1 55%, #0ea5e9 100%);
    color: white;
    margin-bottom: 1rem;
}
.hero h1 { margin-bottom: 0.2rem; }
.hero p  { margin-bottom: 0; opacity: 0.92; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>✈ FlightDelay</h1>
    <p>Flight delay prediction with temporal feature engineering · FAA 15-min threshold · Cyclical encoding</p>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "results" not in st.session_state:
    st.session_state.results = None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙ Configuration")
    data_source = st.radio(
        "Dataset",
        ["nycflights13 — 2013 BTS (live)", "Synthetic (demo)", "Real CSV"],
        index=0,
    )
    sample_n = st.slider("Sample size (synthetic/CSV)", 10000, 200000, 100000, 10000)

    st.divider()
    st.subheader("Algorithms")
    selected_models = st.multiselect(
        "Select models to train",
        ["Gradient Boosting", "Logistic Regression"],
        default=["Gradient Boosting", "Logistic Regression"]
    )

    st.divider()
    st.subheader("Hyperparameters")
    n_estimators = st.slider("Boosting rounds", 50, 500, 200, 50)
    learning_rate = st.slider("Learning rate", 0.01, 0.5, 0.1, 0.01)
    max_depth = st.slider("Max tree depth", 3, 12, 5, 1)

    st.divider()
    st.subheader("Business Parameters")
    cost_fn = st.number_input("Cost per missed delay ($)", 50, 2000, 200, 50)
    cost_fp = st.number_input("Cost per false alarm ($)", 10, 500, 50, 10)

    st.divider()
    st.caption("Built with NumPy · Streamlit")
    st.code("streamlit run app.py", language="bash")


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading flight data...")
def load_data(source, n, seed=42):
    if "nycflights13" in source:
        return fetch_nycflights()
    if source == "Real CSV":
        return load_flights(sample_n=n, seed=seed)
    return load_flights(sample_n=n, seed=seed)   # synthetic fallback


data = load_data(data_source, sample_n)


# ---------------------------------------------------------------------------
# Top metrics
# ---------------------------------------------------------------------------
cols = st.columns(5)
cols[0].metric("Flights", f"{data['n_samples']:,}")
cols[1].metric("Features", len(data["features"]))
cols[2].metric("Delay Rate", f"{data['delay_rate']:.1%}")
cols[3].metric("FAA Threshold", "≥15 min")
cols[4].metric("Cyclical Encoding", "sin/cos")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_data, tab_model, tab_analysis, tab_route, tab_cost = st.tabs([
    "🔍 Data Explorer", "🧪 Model Lab", "📊 Feature Analysis",
    "🗺 Route Analysis", "💰 Cost Optimizer"
])


# ===== TAB 1: Data Explorer =====
with tab_data:
    st.subheader("Flight Data Overview")

    st.dataframe(data["df"].head(100), use_container_width=True)

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Class Distribution** (FAA: delay > 15 min)")
        st.pyplot(plot_delay_distribution(data["y"]))

    with c2:
        st.markdown("**Delay Rate by Departure Hour**")
        hourly = temporal_analysis(data["y"], np.zeros(len(data["y"])),
                                   data["X"][:, data["features"].index("dep_hour")])
        st.pyplot(plot_delay_by_hour(hourly))

    st.divider()
    st.markdown("""
    **Cyclical Encoding Explained:**

    Hours are cyclic: 23:00 is close to 00:00. Naive encoding (0-23) creates artificial distance.
    Solution: map hour h to (sin(2πh/24), cos(2πh/24)) on the unit circle.

    ```python
    hour_sin = sin(2π × hour / 24)
    hour_cos = cos(2π × hour / 24)
    ```

    This preserves the circular nature: distance(23, 0) = distance(12, 13).
    """)


# ===== TAB 2: Model Lab =====
with tab_model:
    st.subheader("Model Training & Evaluation")

    if not selected_models:
        st.warning("Select at least one algorithm in the sidebar.")
    elif st.button("🚀 Train Models", key="train_btn"):
        with st.spinner("Splitting data..."):
            X_train, X_test, y_train, y_test = temporal_split(
                data["X"], data["y"], test_size=0.25
            )
            scaler = StandardScaler().fit(X_train)
            X_train_s = scaler.transform(X_train)
            X_test_s = scaler.transform(X_test)

        trained_results = {}

        if "Gradient Boosting" in selected_models:
            with st.spinner("Training Gradient Boosting..."):
                gb_result = train_gradient_boosting(
                    X_train_s, y_train, n_estimators=n_estimators,
                    learning_rate=learning_rate, max_depth=max_depth
                )
                gb_eval = evaluate(gb_result, X_test_s, y_test)
                trained_results["Gradient Boosting"] = gb_eval
                trained_results["gb_result"] = gb_result

        if "Logistic Regression" in selected_models:
            with st.spinner("Training Logistic Regression..."):
                lr_result = train_logistic_regression(X_train_s, y_train)
                lr_eval = evaluate(lr_result, X_test_s, y_test)
                trained_results["Logistic Regression"] = lr_eval

        trained_results["scaler"] = scaler
        trained_results["X_test"] = X_test_s
        trained_results["y_test"] = y_test

        st.session_state.results = trained_results
        st.success("Training complete!")

    if st.session_state.results:
        results = st.session_state.results
        y_test = results["y_test"]

        # Metrics comparison
        metrics_dict = {k: v["metrics"] for k, v in results.items() if isinstance(v, dict) and "metrics" in v}
        if metrics_dict:
            st.pyplot(plot_metrics_comparison(metrics_dict))

            # Detailed metrics table
            st.markdown("**Detailed Metrics**")
            table = []
            for name, m in metrics_dict.items():
                table.append({
                    "Model": name,
                    "Accuracy": f"{m['accuracy']:.4f}",
                    "Precision": f"{m['precision']:.4f}",
                    "Recall": f"{m['recall']:.4f}",
                    "F1": f"{m['f1']:.4f}",
                    "AUC": f"{m['roc_auc']:.4f}",
                })
            st.table(table)

        # ROC curves
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**ROC Curve — Gradient Boosting**")
            st.pyplot(plot_roc_curve(y_test, results["Gradient Boosting"]["y_proba"], "GBM"))
        with c2:
            st.markdown("**Confusion Matrix — Gradient Boosting**")
            cm = results["Gradient Boosting"]["metrics"]["confusion_matrix"]
            st.pyplot(plot_confusion_matrix(cm))


# ===== TAB 3: Feature Analysis =====
with tab_analysis:
    st.subheader("Feature Importance")

    st.markdown("""
    **Permutation Importance** (model-agnostic):

    For each feature j:
    1. Compute baseline accuracy on test set
    2. Shuffle feature j randomly
    3. Measure accuracy drop: I_j = acc_baseline - acc_shuffled
    4. Repeat 10 times, report mean ± std

    Features with high importance significantly affect model predictions.
    """)

    if st.session_state.results and "gb_result" in st.session_state.results:
        X_test = st.session_state.results["X_test"]
        y_test = st.session_state.results["y_test"]

        with st.spinner("Computing permutation importance..."):
            imp = permutation_importance(
                st.session_state.results["gb_result"],
                X_test, y_test, data["features"], n_repeats=10
            )

        st.pyplot(plot_feature_importance(imp["importances"], imp["std"], imp["feature_names"]))

        st.markdown("**Mathematical Interpretation**")
        top_3 = imp["sorted_idx"][:3]
        for i, idx in enumerate(top_3):
            st.markdown(f"{i+1}. **{imp['feature_names'][idx]}**: Δ accuracy = {imp['importances'][idx]:.4f} ± {imp['std'][idx]:.4f}")
    else:
        st.info("Train Gradient Boosting to see permutation feature importance.")


# ===== TAB 4: Route Analysis =====
with tab_route:
    st.subheader("Hourly Delay Patterns")

    if st.session_state.results:
        X_test = st.session_state.results["X_test"]
        y_test = st.session_state.results["y_test"]

        # Use first available model's probabilities
        model_results = {k: v for k, v in st.session_state.results.items()
                        if isinstance(v, dict) and "y_proba" in v}
        if model_results:
            first_model = list(model_results.keys())[0]
            y_proba = model_results[first_model]["y_proba"]

            hour_idx = data["features"].index("dep_hour")
            hourly = temporal_analysis(y_test, y_proba, X_test[:, hour_idx])

            st.pyplot(plot_delay_by_hour(hourly))

            st.markdown("""
            **Key Insights:**
            - Morning rush (7-9 AM): highest delay rates due to cascading effects from overnight
            - Evening rush (5-7 PM): second peak from accumulated delays throughout the day
            - Late night (12-5 AM): lowest delay rates but fewer flights
            - Midday (12-2 PM): moderate delays, often from weather or mechanical issues
            """)

            # Hourly statistics table
            st.markdown("**Hourly Statistics**")
            hourly_table = []
            for h in sorted(hourly.keys()):
                hourly_table.append({
                    "Hour": f"{h:02d}:00",
                    "Flights": f"{hourly[h]['count']:,}",
                    "Delay Rate": f"{hourly[h]['delay_rate']:.1%}",
                })
            st.table(hourly_table)


# ===== TAB 5: Cost Optimizer =====
with tab_cost:
    st.subheader("Business Cost Optimization")

    st.markdown(f"""
    **Misclassification Costs:**
    - **Missed Delay (FN):** ${cost_fn} — passenger misses connection, compensation required
    - **False Alarm (FP):** ${cost_fp} — unnecessary rebooking, wasted resources
    """)

    if st.session_state.results:
        y_test = st.session_state.results["y_test"]

        # Use first available model's probabilities
        model_results = {k: v for k, v in st.session_state.results.items()
                        if isinstance(v, dict) and "y_proba" in v}
        if model_results:
            cost_model = st.selectbox("Model for cost analysis", list(model_results.keys()), key="cost_model")
            y_proba = model_results[cost_model]["y_proba"]

        # Threshold sweep
        thresholds = np.linspace(0.1, 0.9, 81)
        total_cost = []
        fnr_list = []
        fpr_list = []

        for t in thresholds:
            y_pred = (y_proba >= t).astype(int)
            tp = ((y_pred == 1) & (y_test == 1)).sum()
            fp = ((y_pred == 1) & (y_test == 0)).sum()
            fn = ((y_pred == 0) & (y_test == 1)).sum()
            tn = ((y_pred == 0) & (y_test == 0)).sum()

            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr_list.append(fnr)
            fpr_list.append(fpr)
            total_cost.append(fnr * cost_fn + fpr * cost_fp)

        st.pyplot(plot_cost_analysis(thresholds, np.array(fpr_list),
                                     np.array(fnr_list), cost_fn, cost_fp))

        best_idx = np.argmin(total_cost)
        st.markdown(f"""
        **Optimal Operating Point:**
        - Threshold: **{thresholds[best_idx]:.2f}**
        - Total cost: **${total_cost[best_idx]:,.0f}**

        **Mathematical Formulation:**
        ```
        min  FNR(τ) × C_FN + FPR(τ) × C_FP
        s.t. 0 ≤ τ ≤ 1
        ```

        The optimal τ balances the trade-off between catching delays (high recall)
        and avoiding false alarms (high precision).
        """)

        # Route analysis
        st.divider()
        st.subheader("Route Delay Analysis")
        if "ORIGIN" in data["df"].columns:
            from src.model import route_analysis
            routes = route_analysis(data["df"])
            if routes:
                st.table(routes[:15])
        else:
            st.info("Route analysis requires real flight data with ORIGIN/DEST columns.")


# ---------------------------------------------------------------------------
# Deploy notes
# ---------------------------------------------------------------------------
st.divider()
with st.expander("Deployment & production notes"):
    st.markdown("""
    **FlightDelay** — Production deployment:

    1. **Data pipeline**: Ingest BTS TranStats data daily via Airflow
    2. **Feature store**: Pre-compute carrier/airport frequency encodings
    3. **Model retraining**: Weekly retraining with sliding window
    4. **Real-time scoring**: ONNX export for <10ms inference
    5. **Monitoring**: Track delay rate drift, feature distribution shifts
    """)
    st.code("pip install -r requirements.txt\nstreamlit run app.py", language="bash")
