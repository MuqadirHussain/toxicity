import shap
import streamlit as st
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load example dataset
st.title("SHAP Explanation in Streamlit")

# Generate sample data
X, y = shap.datasets.adult()
X = X.iloc[:1000, :]  # Use a smaller subset for speed
y = y[:1000]

# Train a simple model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# Explain the model's predictions using SHAP
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# Streamlit app to visualize SHAP values
st.subheader("SHAP Summary Plot")
st.set_option('deprecation.showPyplotGlobalUse', False)
shap.summary_plot(shap_values, X_test)
st.pyplot()

st.subheader("Feature Importance")
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': np.abs(shap_values.values).mean(axis=0)})
importance_df = importance_df.sort_values(by="Importance", ascending=False)

st.dataframe(importance_df)
