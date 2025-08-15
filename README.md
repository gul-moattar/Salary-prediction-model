#  Salary Prediction App using Machine Learning

 A modern, interactive Streamlit web app that predicts salary based on **Education Level**, **Job Title**, and **Years of Experience**, using a trained **Random Forest Regressor** model.
<h3>App Preview</h3>
<img src="https://i.postimg.cc/GpPbyTLW/Screenshot-2025-08-07-210224.png" alt="App Screenshot" width="800"/>




---

##  Live Demo

 [Click here to try the live app](https://salary-prediction-model-nqxpe3jvccimkaz26rxxsm.streamlit.app/))

---

##  Features

✅ Predicts salary using real-world trained machine learning model  
✅ Built with **Scikit-learn**, **Pandas**, and **Streamlit**  
✅ Clean and modern UI with interactive sidebar inputs  
✅ Real-time predictions with a dynamic Plotly salary trend chart  
✅ Downloadable prediction logs  
✅ Profile icon integration for personalization  
✅ Performance metrics displayed with expand/collapse view

---

##  Model Used

- **Random Forest Regressor**
- Trained on a dataset with the following features:
  - Age
  - Gender
  - Education Level
  - Job Title
  - Years of Experience
  - Salary (Target)

---

##  Tech Stack

| Tool / Library     | Usage                        |
|--------------------|------------------------------|
| `streamlit`        | Frontend web app framework   |
| `scikit-learn`     | Machine Learning model       |
| `pandas`           | Data manipulation & logging  |
| `numpy`            | Input formatting             |
| `plotly`           | Interactive trend chart      |
| `matplotlib`       | (Backup charting, optional)  |

---

##  Project Structure

salary-prediction-ml/
│
├── app.py # Streamlit app code
├── salary_model.pkl # Trained ML model
├── prediction_log.csv # Logs of predictions
├── requirements.txt # Dependencies
├── README.md # Project documentation
└── notebook.ipynb # Model training & EDA

##  Installation

```bash
# Clone the repo
git clone https://github.com/Malik9544/Salary_prediction_Model.git
cd Salary_prediction_Model

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py

 Logs
Every prediction is logged with:

Timestamp

Education Level

Job Title

Years of Experience

Predicted Salary

📂 Log file: prediction_log.csv
📥 Downloadable from within the app interface.


