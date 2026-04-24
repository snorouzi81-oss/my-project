import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score ,mean_absolute_error,median_absolute_error

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
#print(BASE_DIR)
DATA_PATH = BASE_DIR / "Data"/ "StudentsPerformance.csv"
#DATA_DIR.mkdir(exist_ok=True)

PLOT_DIR = BASE_DIR / "Plots"
PLOT_DIR.mkdir(exist_ok=True)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "Models" / "math_model_RandomForest.pkl" 

# خواندن داده‌ها
df = pd.read_csv(DATA_PATH)

print("Columns: ", df.columns)
print(df.head())

# بررسی اطلاعات کلی داده‌ها
print(df.info())
print(df.isna().sum())

# # ================== Visualization ==================
#fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Reading vs Math
#sns.scatterplot(data=df, x="reading score", y="math score", ax=axes[0,0])
#axes[0,0].set_title("Reading vs Math Score")
#plt.savefig(PLOT_DIR/"ReadingMathScore.png")


#  Writing vs Math
#sns.scatterplot(data=df, x="writing score", y="math score", ax=axes[0,1])
#axes[0,1].set_title("Writing vs Math Score")
#plt.savefig(PLOT_DIR/"WritingMathScore.png")


# Test prep vs Math
#sns.boxplot(data=df, x="test preparation course", y="math score", ax=axes[1,0])
#axes[1,0].set_title("Test Preparation vs Math Score")
#plt.savefig(PLOT_DIR/"Test PreparationMathScore.png")


# Distribution Math Score
#sns.histplot(df["math score"], kde=True, ax=axes[1,1])
#axes[1,1].set_title("Distribution of Math Scores")
#plt.savefig(PLOT_DIR/"DistributionMathScores.png")
#plt.show()

# ================== Feature Selection ==================
# این بار همه featureها رو استفاده می‌کنیم (خیلی مهم!)
X = df.drop("math score", axis=1)
y = df["math score"]

# تبدیل همه ویژگی‌های دسته‌ای به عددی
X = pd.get_dummies(X, drop_first=True)

# ================== Train/Test Split ==================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================== Model ==================
model = RandomForestRegressor(
    n_estimators=200,      # تعداد درخت‌ها
    max_depth=10,          # جلوگیری از overfitting
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

model.fit(X_train, y_train)

# ذخیره مدل

joblib.dump(X.columns, BASE_DIR / "Models/model_columns_RandomForest.pkl")
joblib.dump(model, MODEL_PATH)


# ================== Prediction ==================
predictions = model.predict(X_test)

# مقایسه
results_df = X_test.copy()
results_df["actual_score"] = y_test.values
results_df["predicted_score"] = predictions

print("Comparison table:")
print(results_df.head())

# ================== Evaluation ==================
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
med_ae = median_absolute_error(y_test, predictions)
print(f"Median AE: {med_ae}")



print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")
print(f"MAE: {mae}")
print(f"RMSE: {rmse}")

# ================== New Data Prediction ==================
new_data = pd.DataFrame({
    "gender": ["female"],
    "race/ethnicity": ["group C"],
    "parental level of education": ["bachelor's degree"],
    "lunch": ["standard"],
    "test preparation course": ["completed"],
    "reading score": [80],
    "writing score": [73]
})

# تبدیل به دامی
new_data = pd.get_dummies(new_data, drop_first=True)

# هماهنگ‌سازی ستون‌ها با داده آموزشی
new_data = new_data.reindex(columns=X.columns, fill_value=0)

# پیش‌بینی
new_prediction = model.predict(new_data)
print(f"Predicted score for new data: {new_prediction[0]}")