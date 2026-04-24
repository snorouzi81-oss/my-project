from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------
# مسیر درست فایل
# -----------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data"

df = pd.read_csv(DATA_DIR / "StudentsPerformance.csv")

print(df.head())
print(df.columns)

# -----------------------
# تبدیل ستون categorical
# -----------------------
df = pd.get_dummies(df, columns=["test preparation course"], drop_first=True)

# حالا ستون جدید ساخته می‌شود:
# test preparation course_none

print(df.columns)  # برای اطمینان

# -----------------------
# انتخاب ویژگی‌ها
# -----------------------
X = df[["writing score", "reading score", "test preparation course_none"]]
y = df["math score"]

# -----------------------
# تقسیم داده
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# -----------------------
# مدل درخت تصمیم
# -----------------------
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# -----------------------
# پیش‌بینی
# -----------------------
y_pred = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))

# -----------------------
# پیش‌بینی داده جدید
# -----------------------
new_data = pd.DataFrame({
    "writing score": [73],
    "reading score": [80],
    "test preparation course_none": [0]  # 0 یعنی completed
})

print("Prediction:", model.predict(new_data)[0])