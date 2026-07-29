# 🏦 Loan Status Prediction using Machine Learning

## 🚀 Live Application

**[Streamlit Deployment](https://deployment-project--loan-status-prediction-sindhura-nk.streamlit.app/)**  
---

# 📌 Project Overview

Loan approval is one of the most critical decisions made by financial institutions. This project leverages Machine Learning to predict whether a loan application is likely to be approved based on an applicant's demographic, financial, and credit-related information.

The project covers the complete Machine Learning lifecycle—from data preprocessing and model building to deployment using Streamlit.

---

# 🎯 Problem Statement

Build a classification model that predicts the loan approval status of an applicant based on various personal and financial attributes.

The prediction helps financial institutions:

- Reduce manual loan verification effort
- Improve decision-making consistency
- Identify eligible applicants quickly
- Minimize lending risks

---

# 📂 Dataset Features

The dataset consists of applicant information such as:

- Gender
- Marital Status
- Number of Dependents
- Education
- Self Employed
- Applicant Income
- Co-applicant Income
- Loan Amount
- Loan Amount Term
- Credit History
- Property Area

**Target Variable**

- Loan Status (Approved / Not Approved)

---

# 🛠️ Tasks Performed

## 1. Data Collection

- Imported the loan dataset
- Inspected dataset dimensions
- Checked data types
- Understood feature descriptions

---

## 2. Exploratory Data Analysis (EDA)

Performed exploratory analysis to understand the data.

### Data Exploration

- Dataset shape
- Feature information
- Summary statistics
- Duplicate value check
- Missing value analysis

### Visualization

- Distribution of categorical variables
- Distribution of numerical variables
- Loan approval comparison
- Credit history analysis
- Income analysis
- Property area analysis

---

## 3. Data Preprocessing

Prepared the dataset for machine learning by performing:

- Handling missing values
- Encoding categorical variables
- Feature transformation
- Data cleaning
- Preparing input and target variables

---

## 4. Feature Engineering

- Converted categorical variables into numerical format
- Selected important features
- Removed unnecessary columns (if applicable)

---

## 5. Train-Test Split

Split the dataset into training and testing datasets to evaluate model performance on unseen data.

---

## 6. Model Building

Implemented a Machine Learning classification model to predict loan approval.

Typical workflow:

- Model initialization
- Model training
- Prediction on test data

---

## 7. Model Evaluation

Evaluated model performance using classification metrics such as:

- Accuracy Score
- Confusion Matrix
- Classification Report

These metrics helped determine how effectively the model predicts loan approval.

---

## 8. Model Serialization

Saved the trained model using **Pickle** so it can be reused without retraining.

Saved files include:

- Trained ML model
- Required preprocessing objects (if applicable)

---

## 9. Streamlit Web Application

Developed an interactive web application using Streamlit.

### Features

- User-friendly interface
- Input applicant details
- Real-time loan prediction
- Instant prediction result
- Clean and responsive layout

---

# ⚙️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Pickle
- Streamlit

---

# 📁 Project Structure

```
Loan-Status-Prediction/
│
├── app.py
├── loan_status_model.pkl
├── requirements.txt
├── Loan_Status_Prediction.ipynb
├── dataset.csv
├── README.md
└── assets/
```

---

# ▶️ How to Run the Project

## Clone the Repository

```bash
git clone https://github.com/sindhura-nk/Deployment-Project--Loan-Status-Prediction.git
```

## Navigate to the Project Folder

```bash
cd Deployment-Project--Loan-Status-Prediction
```

## Install Required Packages

```bash
pip install -r requirements.txt
```

## Run the Streamlit Application

```bash
streamlit run app.py
```

---

# 📊 Sample Workflow

1. Open the Streamlit application.
2. Enter applicant details.
3. Click **Predict Loan Status**.
4. View the prediction result.

---

# 💡 Key Learning Outcomes

Through this project, the following concepts were implemented:

- Data Cleaning
- Missing Value Handling
- Exploratory Data Analysis
- Feature Encoding
- Machine Learning Classification
- Model Evaluation
- Model Serialization
- Streamlit Deployment
- End-to-End Machine Learning Pipeline

---

# 🚀 Future Enhancements

- Hyperparameter tuning
- Compare multiple classification algorithms
- Probability score prediction
- SHAP explainability
- Cloud deployment
- Database integration
- User authentication
- Prediction history

---

# 👩‍💻 Author

**Sindhura Kuntamukkula**

[GitHub](https://github.com/sindhura-nk)

---

# ⭐ If you found this project useful...

Please consider giving this repository a **Star ⭐** to support the project.
