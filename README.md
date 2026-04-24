# 🎓 Student Performance Prediction (Machine Learning Project)

## 📌 Project Overview
This project focuses on predicting students' **math scores** based on their academic and demographic features.  
Two machine learning models were implemented and compared:

- Linear Regression  
- Random Forest Regressor  

The goal is to analyze which model performs better in predicting student performance.

---

## 📊 Dataset Description
The dataset contains 1000 student records with the following features:

- gender  
- race/ethnicity  
- parental level of education  
- lunch type  
- test preparation course  
- reading score  
- writing score  
- math score (target variable)

---

## 🔍 Data Preprocessing
- No missing values were found in the dataset  
- Categorical variables were converted using **One-Hot Encoding**  
- Feature selection:
  - reading score  
  - writing score  
  - demographic and educational features  
- Train/Test split: 80% / 20%

---

## 🤖 Models Used

### 1️⃣ Linear Regression
A baseline regression model used for comparison.

- Simple linear relationship assumption  
- Fast and interpretable  
- Lower ability to capture complex patterns  

Performance:
- R² ≈ 0.68  
- MSE ≈ 76  

---

### 2️⃣ Random Forest Regressor
:contentReference[oaicite:0]{index=0}

An advanced ensemble learning model based on multiple decision trees.

- Handles non-linear relationships  
- Reduces overfitting  
- More accurate predictions  

Performance:
- R² ≈ 0.85  
- RMSE ≈ 5.98  
- MAE ≈ 4.62  

---

## 📈 Model Comparison

| Model | R² Score | RMSE | Performance |
|------|----------|------|-------------|
| Linear Regression | 0.68 | ~8.7 | Baseline |
| Random Forest | 0.85 | 5.98 | Better accuracy |

---

## 🔮 Prediction Example
Example input:

- Writing score: 73  
- Reading score: 80  
- Test preparation: completed  

Predicted Math Score:
≈ 65–73 (depending on model)

---

## 📁 Project Structure





---


## 🚀 Key Insights
- Writing and reading scores are strong predictors of math performance  
- Random Forest performs significantly better than Linear Regression  
- Non-linear models are more suitable for this dataset  

---

## 👨‍💻 Author
Machine Learning Student Project  
Focus: Regression models and performance comparison

---

## 📌 Future Improvements
- Add more models (Decision Tree, SVR)  
- Feature importance analysis  
- Hyperparameter tuning  
- Cross-validation for better evaluation  
