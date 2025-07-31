import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# ------------------------------------
# Load and preprocess dataset
@st.cache_data
def load_data():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

    # Convert 'TotalCharges' to numeric, drop NaNs created by coercion
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna(subset=['TotalCharges']).reset_index(drop=True)

    # Encode categorical variables
    cat_cols = df.select_dtypes(include='object').columns.drop('customerID')
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    return df, encoders

df, encoders = load_data()

X = df.drop(['Churn', 'customerID'], axis=1)
y = df['Churn']

# ------------------------------------
# Train model (cached to avoid retraining on every run)
@st.cache_data
def train_model():
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)
    return model

model = train_model()

# ------------------------------------
# Sidebar input widgets for user data
st.sidebar.header("Input Customer Data")

def user_input_features(encoders):
    tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.sidebar.slider("Monthly Charges (€)", 0.0, 120.0, 70.0)
    total_charges = st.sidebar.slider("Total Charges (€)", 0.0, 9000.0, 2000.0)

    # Mapping encoded values to human-readable labels
    contract_map = {0: 'Month-to-month', 1: 'One year', 2: 'Two year'}
    payment_map = {0:'Bank transfer (automatic)', 1:'Credit card (automatic)', 2:'Electronic check', 3:'Mailed check'}
    internet_map = {0:'DSL', 1:'Fiber optic', 2:'No'}

    # Selectboxes with readable labels
    contract_label = st.sidebar.selectbox("Contract Type", list(contract_map.values()))
    payment_label = st.sidebar.selectbox("Payment Method", list(payment_map.values()))
    internet_label = st.sidebar.selectbox("Internet Service", list(internet_map.values()))

    # Convert labels back to encoded integers for the model
    contract = {v: k for k, v in contract_map.items()}[contract_label]
    payment = {v: k for k, v in payment_map.items()}[payment_label]
    internet = {v: k for k, v in internet_map.items()}[internet_label]

    # Build user input dictionary matching model features
    user_data = {
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Contract': contract,
        'PaymentMethod': payment,
        'InternetService': internet,
        # Other features will be filled with defaults
    }

    return user_data

user_inputs = user_input_features(encoders)

# ------------------------------------
# Prepare complete input DataFrame for prediction

# Create a DataFrame with all features filled with zeros
input_df = pd.DataFrame(np.zeros((1, X.shape[1])), columns=X.columns)

# Overwrite columns with user input values
for col, val in user_inputs.items():
    if col in input_df.columns:
        input_df.loc[0, col] = val

# Fill remaining features with mode (most frequent) values from training data
for col in input_df.columns:
    if col not in user_inputs:
        if col in df.columns and df[col].dtype in [np.int64, np.int32]:
            input_df.loc[0, col] = df[col].mode()[0]

# ------------------------------------
# Streamlit app display

st.title("📞 Telco Customer Churn Prediction")
st.markdown(
    "Predict whether a customer is likely to churn based on their profile and subscription details."
)

st.subheader("Input Customer Data")
st.write(input_df)

# Make prediction and show results
prediction = model.predict(input_df)[0]
prediction_proba = model.predict_proba(input_df)[0][1]

st.subheader("Prediction Result")
if prediction == 1:
    st.error("⚠️ This customer is **likely to churn**.")
else:
    st.success("✅ This customer is **unlikely to churn**.")

st.write(f"**Probability of churn:** {prediction_proba:.2f}")

# Show feature importance chart
st.subheader("Top 10 Features Influencing Churn")
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x=importances.values, y=importances.index, ax=ax, palette="viridis")
ax.set_title("Feature Importance")
st.pyplot(fig)
