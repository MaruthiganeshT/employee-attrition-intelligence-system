# Employee Attrition Intelligence System 🏢

## Project Overview

This project predicts whether an employee is likely to leave a company using Machine Learning. The goal is to help organizations identify employees who may be at risk of attrition and understand the factors that contribute to their decision.

The project was built using the IBM HR Analytics Employee Attrition dataset and covers the complete Machine Learning workflow, including data preprocessing, feature engineering, model training, evaluation, explainability, and deployment using Streamlit.

---

## Features

* Data cleaning and preprocessing
* Feature engineering based on employee-related factors
* Multiple Machine Learning models:

  * Logistic Regression
  * Random Forest
  * XGBoost
* Model comparison using:

  * Accuracy
  * Precision
  * Recall
  * F1 Score
  * ROC-AUC
* SHAP explainability for model interpretation
* Interactive Streamlit dashboard for real-time predictions

---

## Dataset

Dataset: IBM HR Analytics Employee Attrition Dataset

The dataset contains employee information such as:

* Age
* Monthly Income
* Job Satisfaction
* Years at Company
* Overtime Status
* Distance From Home
* Department
* Job Role

Target Variable:

```text
Attrition
0 = Employee Stays
1 = Employee Leaves
```

---

## Project Structure

```text
project1/
│
├── app/
│   └── app.py
│
├── src/
│   ├── data_loader.py
│   ├── eda.py
│   ├── preprocessing.py
│   ├── model_trainer.py
│   └── explainer.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
├── eda_reports/
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Machine Learning Pipeline

### 1. Data Preprocessing

* Removed unnecessary columns
* Handled missing values
* Encoded categorical variables
* Scaled numerical features
* Created additional engineered features

### 2. Feature Engineering

The following features were created:

* TotalSatisfaction
* CareerStability
* IncomePerYearAtCompany
* OvertimeSatisfactionRisk

### 3. Model Training

The following models were trained and compared:

* Logistic Regression
* Random Forest
* XGBoost

The best model was selected using F1 Score.

### 4. Explainability

SHAP was used to understand:

* Global feature importance
* Individual employee predictions

### 5. Deployment

The final model was deployed using Streamlit.

---

## Running the Project

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Pipeline

```bash
python src/preprocessing.py
python src/model_trainer.py
python src/explainer.py
```

### Launch Dashboard

```bash
streamlit run app/app.py
```

---

## Sample Dashboard Features

* Attrition Risk Prediction
* Risk Category Classification
* Local SHAP Waterfall Explanation
* Global Feature Importance
* Model Performance Metrics

---

## Future Improvements

* Hyperparameter tuning
* Model deployment on cloud platforms
* Additional explainability visualizations
* Real-time HR analytics integration

---

## Author

**Thurpu Maruthi Ganesh**

Built as a Machine Learning portfolio project to strengthen practical skills in Data Science, Machine Learning, Explainable AI, and Model Deployment.
