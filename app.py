import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# Page config
st.set_page_config(
    page_title="Churn Risk Scorer",
    page_icon="🔴",
    layout="wide"
)

# Load model and feature columns
@st.cache_resource
def load_model():
    model = joblib.load("model/churn_model.pkl")
    feature_cols = joblib.load("model/feature_columns.pkl")
    return model, feature_cols

model, feature_cols = load_model()

def preprocess_upload(df):
    # Fix TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])

    # Drop customerID if present
    customer_ids = df["customerID"].copy() if "customerID" in df.columns else pd.Series(range(len(df)))
    df = df.drop(columns=["customerID"], errors="ignore")

    # Drop Churn column if present
    df = df.drop(columns=["Churn"], errors="ignore")

    # Binary columns
    binary_cols = [
        "Partner", "Dependents", "PhoneService", "PaperlessBilling",
        "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    for col in binary_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0, "No phone service": 0, "No internet service": 0})

    # One-hot encode
    df = pd.get_dummies(df, columns=["gender", "InternetService", "Contract", "PaymentMethod"])

    # Align columns to match training data exactly
    df = df.reindex(columns=feature_cols, fill_value=0)

    return df, customer_ids

def generate_churn_reason(row):
    reasons = []
    if row["Contract_Month-to-month"] == 1:
        reasons.append("on a month-to-month contract with no long-term commitment")
    if row["tenure"] < 12:
        reasons.append(f"only been a customer for {int(row['tenure'])} months")
    if row["MonthlyCharges"] > 70:
        reasons.append(f"paying a high monthly charge of ${row['MonthlyCharges']}")
    if row["OnlineSecurity"] == 0:
        reasons.append("no online security add-on")
    if row["TechSupport"] == 0:
        reasons.append("no tech support subscription")
    if reasons:
        return "At risk due to: " + ", ".join(reasons) + "."
    return "Multiple moderate risk factors detected."

st.title("🔴 Customer Churn Risk Scorer")
st.markdown("Upload a CSV of customer data to get churn risk scores and AI-generated reasons.")

uploaded_file = st.file_uploader("Upload customer CSV", type=["csv"])

if uploaded_file:
    raw_df = pd.read_csv(uploaded_file)
    st.subheader("Raw Data Preview")
    st.dataframe(raw_df.head())

    with st.spinner("Scoring customers..."):
        processed_df, customer_ids = preprocess_upload(raw_df.copy())

        # Generate risk scores
        risk_scores = (model.predict_proba(processed_df)[:, 1] * 100).round(1)

        # Build results table
        results = pd.DataFrame({
            "CustomerID": customer_ids.values,
            "ChurnRisk": risk_scores,
            "RiskLevel": pd.cut(risk_scores,
                                bins=[0, 40, 70, 100],
                                labels=["🟢 Low", "🟡 Medium", "🔴 High"])
        })

        # Add AI reasons for high risk only
        high_risk_mask = risk_scores > 60
        processed_df_reset = processed_df.reset_index(drop=True)
        results["Reason"] = ""
        results.loc[high_risk_mask, "Reason"] = processed_df_reset[high_risk_mask].apply(
            generate_churn_reason, axis=1
        )

    # Summary metrics
    st.subheader("Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", len(results))
    col2.metric("High Risk", int((risk_scores > 70).sum()))
    col3.metric("Average Risk Score", f"{risk_scores.mean():.1f}")

    # Risk distribution chart
    st.subheader("Risk Score Distribution")
    fig = px.histogram(results, x="ChurnRisk", nbins=20,
                       color_discrete_sequence=["#ef4444"],
                       labels={"ChurnRisk": "Churn Risk Score"})
    st.plotly_chart(fig, use_container_width=True)

    # Full results table
    st.subheader("Customer Risk Table")
    st.dataframe(
        results.sort_values("ChurnRisk", ascending=False),
        use_container_width=True
    )

    # Download button
    csv = results.sort_values("ChurnRisk", ascending=False).to_csv(index=False)
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name="churn_risk_results.csv",
        mime="text/csv"
    )