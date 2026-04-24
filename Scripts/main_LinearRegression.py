import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
#print(BASE_DIR)
DATA_PATH = BASE_DIR / "Data"/ "StudentsPerformance.csv"
#DATA_DIR.mkdir(exist_ok=True)

PLOT_DIR = BASE_DIR / "Plots"
PLOT_DIR.mkdir(exist_ok=True)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "Models" / "math_model_LinearRegression.pkl" 

# خواندن داده‌ها
df = pd.read_csv(DATA_PATH)
print("Columns: ", df.columns)
print(df.head())

# بررسی اطلاعات کلی داده‌ها
print(df.info())
print(df.isna().sum())

# نمایش نمودارهای بررسی داده‌ها (برای بررسی روابط مختلف)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Reading vs Math
sns.scatterplot(data=df, x="reading score", y="math score", ax=axes[0,0])
axes[0,0].set_title("Reading vs Math Score")
plt.savefig(PLOT_DIR/"ReadingMathScore.png")


#  Writing vs Math
sns.scatterplot(data=df, x="writing score", y="math score", ax=axes[0,1])
axes[0,1].set_title("Writing vs Math Score")
plt.savefig(PLOT_DIR/"WritingMathScore.png")


# Test prep vs Math
sns.boxplot(data=df, x="test preparation course", y="math score", ax=axes[1,0])
axes[1,0].set_title("Test Preparation vs Math Score")
plt.savefig(PLOT_DIR/"Test PreparationMathScore.png")


# Distribution Math Score
sns.histplot(df["math score"], kde=True, ax=axes[1,1])
axes[1,1].set_title("Distribution of Math Scores")
plt.savefig(PLOT_DIR/"DistributionMathScores.png")
plt.show()



# انتخاب ویژگی‌ها و هدف
# حالا از ویژگی‌های writing score, reading score و test preparation course برای پیش‌بینی math score استفاده می‌کنیم
X = df[["writing score", "reading score", "test preparation course"]]
y = df["math score"]

# تبدیل ویژگی دسته‌ای 'test preparation course' به ویژگی عددی (One-Hot Encoding)
X = pd.get_dummies(X, drop_first=True)  # drop_first=True برای جلوگیری از خطای هم‌خطی

# تقسیم داده‌ها به آموزش و تست
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# مدل رگرسیون خطی
model = LinearRegression()
model.fit(X_train, y_train)
joblib.dump(X.columns, "Models/model_columns.pkl")

#ذخیره مدل
joblib.dump(model, MODEL_PATH)

# پیش‌بینی با مدل
predictions = model.predict(X_test)

# مقایسه پیش‌بینی‌ها با مقادیر واقعی
results_df = X_test.copy()
results_df["actual_score"] = y_test.values
results_df["predicted_score"] = predictions

print("Comparison table:")
print(results_df.head())
#ذخیره مدل
joblib.dump(model, MODEL_PATH)
# ارزیابی مدل
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")

# پیش‌بینی برای داده جدید (مثال: نمره نوشتاری 73، نمره خواندن 80 و دوره آمادگی آزمون = 'completed')
new_data = pd.DataFrame({
    "writing score": [73],
    "reading score": [80],
    "test preparation course": ['completed']  # دوره آمادگی آزمون تکمیل شده است
})

# تبدیل داده جدید به ویژگی‌های عددی و تطبیق با داده‌های آموزشی
new_data = pd.get_dummies(new_data, drop_first=True)  # تطبیق داده جدید با ویژگی‌های آموزش

# بررسی اینکه آیا ویژگی‌های داده جدید با ویژگی‌های داده‌های آموزشی تطابق دارند
new_data = new_data.reindex(columns=X.columns, fill_value=0)

# پیش‌بینی با داده جدید
new_prediction = model.predict(new_data)
print(f"Predicted score for new data: {new_prediction[0]}")