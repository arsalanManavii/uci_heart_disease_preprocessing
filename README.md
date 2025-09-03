# 📊 UCI Heart Disease Preprocessing

## 📌 Overview
This project preprocesses the **UCI Heart Disease dataset** to clean and prepare data for machine learning models.  

It is my **second hands-on preprocessing project**, following my earlier work on the Titanic dataset. While the Titanic dataset gave me a foundation in handling missing values and encoding categorical features, this project pushed me further into **more complex medical data preprocessing**.

---

## 🎯 Goal of this project
- Practice **more advanced preprocessing** on a healthcare dataset  
- Handle **mixed data types** (categorical + continuous + ordinal) more carefully  
- Improve my understanding of how preprocessing impacts **model accuracy** in medical prediction tasks  
- Strengthen my skills in building **reusable preprocessing workflows**  

---

## ⭐ Features
- Loading and inspecting the **UCI Heart Disease dataset**  
- Identifying and handling missing values in a medical dataset  
- Encoding categorical variables (e.g., sex, chest pain type, fasting blood sugar)  
- Scaling continuous features (e.g., age, cholesterol, maximum heart rate)  
- Splitting the data into training and testing sets for ML pipelines  

---

## ⚙️ Setup & Installation
1. **Clone the repository**
   ```bash
   git clone https://github.com/arsalanManavii/uci_heart_disease_preprocessing.git
   cd uci_heart_disease_preprocessing
2. **Install dependencies**
   ```bash
   pip install pandas numpy scikit-learn seaborn
3. **Run the preprocessing script**
   ```bash
   python uci_heart_disease.py

## 🚀 My Progression (Titanic → Heart Disease)
- Titanic Project: Focused on learning the basics — missing values, encoding categorical data, simple feature scaling.
- Heart Disease Project: Applied those skills to a real medical dataset, with more diverse features and preprocessing challenges.

## 💯 Next Steps to Practice
- Exploratory Data Analysis (EDA)
  * Visualize patient distributions, correlations, and key risk factors.
- Pipeline Automation
  * Use `scikit-learn`’s `Pipeline` or `ColumnTransformer` to modularize preprocessing.
