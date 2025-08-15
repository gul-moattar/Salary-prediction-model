import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os
import datetime
import time

# Page config
st.set_page_config(
    page_title="Salary Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model
@st.cache_resource
def load_model():
    return pickle.load(open("salary_model.pkl", "rb"))

model = load_model()

# Load dataset
@st.cache_data
def load_dataset():
    return pd.read_csv("dataset.csv")

df = load_dataset()

# Encoded mappings
education_options = df["Education Level"].dropna().unique().tolist()
job_title_options = df["Job Title"].dropna().unique().tolist()

edu_mapping = {val: idx for idx, val in enumerate(education_options)}
job_mapping = {val: idx for idx, val in enumerate(job_title_options)}

col1 = st.columns([1])[0]
with col1:
    st.markdown(
        "<h1 style='text-align: center;'>Interactive Salary Prediction App</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align: center;'>Predict your salary using Machine Learning trained on real-world employee data.</p>",
        unsafe_allow_html=True
    )

st.markdown("---")

# Tabs layout
tabs = st.tabs([" Predict", " Model Performance", " Logs", " Visualizations", "ℹ About"])

# --- Predict Tab ---
with tabs[0]:
    st.subheader(" Predict Your Salary")

    education = st.selectbox("Select Your Education Level", education_options)
    job_title = st.selectbox("Select Your Job Title", job_title_options)
    experience = st.slider("Years of Experience", 0.0, 40.0, 2.0, 0.5)

    if st.button("Generate Prediction"):
        edu_encoded = edu_mapping.get(education, 0)
        job_encoded = job_mapping.get(job_title, 0)
        input_data = np.array([[experience, edu_encoded, job_encoded]])

        with st.spinner("Generating prediction..."):
            time.sleep(1.2)
            predicted_salary = model.predict(input_data)[0]

        st.markdown("###  **Prediction Result**")
        col1, col2 = st.columns(2)
        col1.metric(label="Predicted Salary", value=f"${predicted_salary:,.2f}")
        col2.metric(label="Years of Experience", value=f"{experience} years")

        # Save log
        log_entry = {
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'education_level': education,
            'job_title': job_title,
            'years_experience': experience,
            'predicted_salary': predicted_salary
        }

        log_file = 'prediction_log.csv'
        if os.path.exists(log_file):
            log_df = pd.read_csv(log_file)
            log_df = pd.concat([log_df, pd.DataFrame([log_entry])], ignore_index=True)
        else:
            log_df = pd.DataFrame([log_entry])

        log_df.to_csv(log_file, index=False)
        st.success(" Prediction logged successfully!")

        # Download result
        st.download_button("⬇ Download Prediction", data=pd.DataFrame([log_entry]).to_csv(index=False), file_name="prediction.csv")

        # Plotly chart for salary trend
        st.markdown("###  Predicted Salary Trend")
        x_vals = np.linspace(0, 40, 100)
        y_vals = [model.predict([[x, edu_encoded, job_encoded]])[0] for x in x_vals]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', name='Trend', line=dict(color='#00BFFF')))
        fig.add_trace(go.Scatter(x=[experience], y=[predicted_salary], mode='markers',
                                 name='Your Prediction', marker=dict(size=10, color='red')))
        fig.update_layout(title=f"Salary Trend for {job_title} with {education}",
                          xaxis_title='Years of Experience',
                          yaxis_title='Salary',
                          template='plotly_white',
                          height=400)
        st.plotly_chart(fig, use_container_width=True)

# --- Model Performance Tab ---
with tabs[1]:
    st.markdown("##  Model Evaluation Metrics")
    st.markdown("""
    | Metric   | Value     |
    |----------|-----------|
    | MAE      | 9,873.85  |
    | MSE      | 193,135,734.42 |
    | RMSE     | 13,897.33 |
    | R² Score | 0.9194    |
    """)

# --- Logs Tab ---
with tabs[2]:
    st.markdown("##  Prediction Logs")
    if os.path.exists("prediction_log.csv"):
        log_data = pd.read_csv("prediction_log.csv")
        st.dataframe(log_data.tail(20), use_container_width=True)
        st.download_button("⬇ Download Full Log", data=log_data.to_csv(index=False), file_name="prediction_log.csv", mime="text/csv")
    else:
        st.info("No predictions logged yet.")

# --- Visualizations Tab ---
with tabs[3]:
    st.markdown("##  Salary Visual Insights")

    # Job Title-wise average salary
    job_avg = df.groupby("Job Title")["Salary"].mean().sort_values()
    st.plotly_chart(go.Figure(data=[go.Bar(x=job_avg.index, y=job_avg.values, marker_color='indigo')],
                              layout=go.Layout(title="Average Salary by Job Title", xaxis_title="Job Title", yaxis_title="Salary")), use_container_width=True)

    # Education level boxplot
    edu_order = df["Education Level"].unique()
    box_data = [go.Box(y=df[df["Education Level"] == lvl]["Salary"], name=lvl) for lvl in edu_order]
    st.plotly_chart(go.Figure(data=box_data, layout=go.Layout(title="Salary by Education Level", yaxis_title="Salary")), use_container_width=True)

    # Experience vs Salary Scatter
    st.plotly_chart(go.Figure(data=[go.Scatter(x=df["Years of Experience"], y=df["Salary"], mode='markers', marker=dict(color='orange'))],
                              layout=go.Layout(title="Experience vs Salary", xaxis_title="Years of Experience", yaxis_title="Salary")), use_container_width=True)

# --- About Tab ---
with tabs[4]:
    st.info("""
    This salary prediction app was built as part of the **2025 Summer Internship at DIGIPEX Solutions LLC**.

    -  Built with Python, Streamlit, and Scikit-learn  
    -  Predicts salary using Education Level, Job Title, and Years of Experience  
    -  Model used: Random Forest Regressor  
    -  Visualization shows salary trend across experience years
    """)

# Footer
st.markdown("""
<hr style="border:1px solid #ccc;">
<p style='text-align:center; font-size: 14px;'>
 Created by <a href='https://github.com/Malik9544' target='_blank'>guL-moattar</a> |
<a href='https://salarypredictionmodel-8tfx9nxanp55wrqoxgbgm3.streamlit.app/' target='_blank'>Live App</a> |
<a href='https://github.com/Malik9544/Salary_prediction_Model' target='_blank'>GitHub Repo</a>
</p>
""", unsafe_allow_html=True)
