import streamlit as st
import pandas as pd
import joblib
import numpy as np

# =========================
# Page config + styling
# =========================
st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📱",
    layout="wide"
)

# =========================
# Load model (cached)
# =========================
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

try:
    model = load_model()
except FileNotFoundError:
    st.error("Model file 'model.pkl' not found. Please ensure it is in the same directory as streamlit_app.py.")
    st.stop()

# =========================
# Feature Engineering (ONLY if your model was trained on FE data)
# If you trained gb_tuned on X_train_fe / X_test_fe, keep this ON.
# If you trained directly on raw X_train / X_test, set USE_FE = False.
# =========================
USE_FE = True

def add_features_telco(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()

    # tenure_group bins
    if "tenure" in X.columns:
        bins = [-1, 12, 24, 48, 72, np.inf]
        labels = ["0-12", "13-24", "25-48", "49-72", "73+"]
        X["tenure_group"] = pd.cut(X["tenure"], bins=bins, labels=labels)

    # services_count = count of add-on services that are "Yes"
    service_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    existing = [c for c in service_cols if c in X.columns]
    X["services_count"] = (X[existing] == "Yes").sum(axis=1) if existing else 0

    # is_month_to_month
    if "Contract" in X.columns:
        X["is_month_to_month"] = (X["Contract"] == "Month-to-month").astype(int)

    return X

# =========================
# Sidebar: controls
# =========================
st.sidebar.header("⚙️ Controls")

threshold = st.sidebar.slider(
    "Decision Threshold",
    min_value=0.10, max_value=0.90, value=0.50, step=0.01,
    help="Lower threshold catches more churners (higher recall) but may increase false alarms (lower precision)."
)

show_inputs = st.sidebar.checkbox("Show input row", value=True)
show_explain = st.sidebar.checkbox("Show interpretation", value=True)

st.sidebar.divider()
st.sidebar.caption("Model file: model.pkl")
st.sidebar.caption("App file: streamlit_app.py")

# =========================
# Main: title + tabs
# =========================
st.title("📱 Telco Customer Churn Predictor")
st.write("Enter customer details to predict churn risk and churn probability.")

tab_predict, tab_about = st.tabs(["🔮 Predict", "ℹ️ About"])

with tab_predict:
    # =========================
    # Input Form
    # =========================
    with st.form("churn_form"):
        st.subheader("Customer Demographics")
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            gender = st.selectbox("Gender", ["Male", "Female"])
        with c2:
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        with c3:
            partner = st.selectbox("Has Partner?", ["No", "Yes"])
        with c4:
            dependents = st.selectbox("Has Dependents?", ["No", "Yes"])

        st.subheader("Account Information")
        c5, c6, c7, c8 = st.columns(4)

        with c5:
            tenure = st.slider("Tenure (Months)", 0, 72, 12)
        with c6:
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        with c7:
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        with c8:
            payment_method = st.selectbox(
                "Payment Method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
            )

        c9, c10 = st.columns(2)
        with c9:
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=300.0, value=70.0, step=1.0)
        with c10:
            total_charges = st.number_input(
                "Total Charges ($)",
                min_value=0.0, max_value=50000.0, value=500.0, step=10.0,
                help="If you're unsure, leave as default. (In real systems, TotalCharges may be derived.)"
            )

        st.subheader("Services Subscribed")
        s1, s2, s3 = st.columns(3)

        with s1:
            phone_service = st.selectbox("Phone Service", ["No", "Yes"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

        with s2:
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])

        with s3:
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

        submitted = st.form_submit_button("Predict Churn")

    # =========================
    # Prediction Logic
    # =========================
    if submitted:
        senior_citizen_val = 1 if senior_citizen == "Yes" else 0

        input_data = pd.DataFrame({
            "gender": [gender],
            "SeniorCitizen": [senior_citizen_val],
            "Partner": [partner],
            "Dependents": [dependents],
            "tenure": [tenure],
            "PhoneService": [phone_service],
            "MultipleLines": [multiple_lines],
            "InternetService": [internet_service],
            "OnlineSecurity": [online_security],
            "OnlineBackup": [online_backup],
            "DeviceProtection": [device_protection],
            "TechSupport": [tech_support],
            "StreamingTV": [streaming_tv],
            "StreamingMovies": [streaming_movies],
            "Contract": [contract],
            "PaperlessBilling": [paperless],
            "PaymentMethod": [payment_method],
            "MonthlyCharges": [monthly_charges],
            "TotalCharges": [total_charges],
        })

        # Apply FE if your training used X_train_fe / X_test_fe
        input_for_model = add_features_telco(input_data) if USE_FE else input_data

        # Predict
        proba = float(model.predict_proba(input_for_model)[0, 1])
        pred = int(proba >= threshold)

        # =========================
        # Output (metrics + visuals)
        # =========================
        st.divider()

        out1, out2, out3 = st.columns(3)
        with out1:
            st.metric("Churn Probability", f"{proba:.2%}")
        with out2:
            st.metric("Decision Threshold", f"{threshold:.2f}")
        with out3:
            st.metric("Predicted Label", "Churn" if pred == 1 else "No Churn")

        st.progress(proba)

        if pred == 1:
            st.error("⚠️ High Risk of Churn")
            st.write("Suggested action: target this customer with retention outreach (e.g., contract upgrade / value add).")
        else:
            st.success("✅ Low Risk of Churn")
            st.write("Suggested action: standard engagement; monitor if conditions change (e.g., price increases).")

        # Show input row (transparency)
        if show_inputs:
            with st.expander("Show input row used for prediction", expanded=False):
                st.dataframe(input_for_model)

        # Business interpretation (short, defensible)
        if show_explain:
            with st.expander("How to interpret this result", expanded=False):
                st.write(
                    "- **Higher probability** means the customer looks similar to past churners.\n"
                    "- The **threshold** controls how aggressive retention targeting is.\n"
                    "  - Lower threshold → catch more churners (higher recall), but more false alarms.\n"
                    "  - Higher threshold → fewer false alarms (higher precision), but miss more churners."
                )

        # Download prediction record (useful + simple)
        result_row = input_for_model.copy()
        result_row["churn_probability"] = proba
        result_row["prediction"] = pred

        csv = result_row.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download prediction as CSV",
            data=csv,
            file_name="churn_prediction.csv",
            mime="text/csv"
        )

with tab_about:
    st.subheader("About this app")
    st.write(
        "This app predicts **Telco Customer Churn** using a trained **Gradient Boosting** model.\n\n"
        "Deployment flow (as per your slides):\n"
        "1) Save trained model (`model.pkl`)\n"
        "2) Load model in Streamlit\n"
        "3) Collect unseen user input\n"
        "4) Apply the same preprocessing/feature engineering\n"
        "5) Predict churn probability and decision label\n"
    )

    st.info(
        "If you trained your final model on **feature-engineered data** (tenure_group/services_count/is_month_to_month), "
        "keep `USE_FE = True`. If your final model was trained on raw features only, set `USE_FE = False`."
    )
