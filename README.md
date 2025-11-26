# **Health Status Prediction App**

A Streamlit-based application that predicts a user's health status using a machine-learning model trained on a dataset of 10,000 records. The app provides predictions, personalized recommendations, and a history dashboard with visual insights.

---

## **Features**

### **Health Status Prediction**

Accepts ten health-related inputs:

* Physical Activity
* Nutrition Score
* Stress Level
* Mindfulness
* Sleep Hours
* Hydration
* BMI
* Alcohol
* Smoking
* Overall Health Score

Outputs the predicted health category.

### **Personalized Recommendations**

Generates improvement suggestions based on weak lifestyle parameters, including areas like activity, sleep, stress, hydration, BMI, alcohol use, and smoking.

### **History Dashboard**

Maintains session-based history and displays:

* Overall Health Score trend
* Distribution of predicted health categories
* Table of all previous inputs and predictions

---

## **Project Structure**

* `app.py` – Streamlit application
* `model.pkl` – Trained machine-learning model
* `requirements.txt` – Project dependencies
* `README.md` – Documentation





## **Model Information**

* Dataset Size: 10,000 rows and 11 columns
* Train/Test Split: 70% / 30%
* Logistic Regression Accuracy: 0.9893
* LightGBM Accuracy: 0.9973
* Final model stored in `model.pkl`

---

## **Requirements**

* streamlit
* numpy
* pandas
* scikit-learn
* lightgbm

---

