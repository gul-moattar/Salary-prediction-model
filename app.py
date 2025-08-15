import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
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

# Page header
st.markdown(
    "<h1 style='text-align: center;'>Salary Prediction Web App</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center;'>Predict your salary on real-world employee data.</p>",
    unsafe_allow_html=True
)
st.markdown("---")

# Tabs layout (only 3 tabs now)
tabs = st.tabs([" Predict", " Model Performance", " Visualizations"])

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

        # Download result
        log_entry = {
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'education_level': education,
            'job_title': job_title,
            'years_experience': experience,
            'predicted_salary': predicted_salary
        }
        st.download_button("⬇ Download Prediction", 
                           data=pd.DataFrame([log_entry]).to_csv(index=False), 
                           file_name="prediction.csv")

        # Salary trend chart
        st.markdown("###  Predicted Salary Trend")
        x_vals = np.linspace(0, 40, 100)
        y_vals = [model.predict([[x, edu_encoded, job_encoded]])[0] for x in x_vals]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_vals, 
            y=y_vals, 
            mode='lines', 
            name='Trend', 
            fill='tozeroy', 
            line=dict(color='#00BFFF')
        ))
        fig.add_trace(go.Scatter(
            x=[experience], 
            y=[predicted_salary], 
            mode='markers',
            name='Your Prediction', 
            marker=dict(size=10, color='red')
        ))
        fig.update_layout(
            title=f"Salary Trend for {job_title} with {education}",
            xaxis_title='Years of Experience',
            yaxis_title='Salary',
            template='plotly_white',
            height=400
        )
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

# --- Visualizations Tab ---
with tabs[2]:
    st.markdown("##  Salary Visual Insights")

    # 1. Job Title-wise average salary - Horizontal Bar Chart
    job_avg = df.groupby("Job Title")["Salary"].mean().sort_values()
    fig1 = go.Figure(data=[go.Bar(
        x=job_avg.values,
        y=job_avg.index,
        orientation='h',
        marker=dict(color=job_avg.values, colorscale='Blues')
    )])
    fig1.update_layout(
        title="Average Salary by Job Title",
        xaxis_title="Salary",
        yaxis_title="Job Title",
        template='plotly_white'
    )
    st.plotly_chart(fig1, use_container_width=True)

    # 2. Salary by Education Level - Violin Plot
    edu_order = df["Education Level"].unique()
    fig2 = go.Figure()
    for lvl in edu_order:
        fig2.add_trace(go.Violin(
            y=df[df["Education Level"] == lvl]["Salary"],
            name=lvl,
            box_visible=True,
            meanline_visible=True
        ))
    fig2.update_layout(
        title="Salary Distribution by Education Level (Violin Plot)",
        yaxis_title="Salary",
        template='plotly_white'
    )
    st.plotly_chart(fig2, use_container_width=True)

    # 3. Experience vs Salary - Bubble Chart
    fig3 = go.Figure(data=[go.Scatter(
        x=df["Years of Experience"],
        y=df["Salary"],
        mode='markers',
        marker=dict(
            size=df["Salary"] / df["Salary"].max() * 30,  # bubble size relative to salary
            color=df["Salary"],
            colorscale='Viridis',
            showscale=True
        ),
        text=df["Job Title"]
    )])
    fig3.update_layout(
        title="Experience vs Salary (Bubble Chart)",
        xaxis_title="Years of Experience",
        yaxis_title="Salary",
        template='plotly_white'
    )
    st.plotly_chart(fig3, use_container_width=True)
