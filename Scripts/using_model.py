import pandas as pd
import joblib
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
# load model + columns
MODEL_PATH = BASE_DIR / "Models" / "math_model_LinearRegression.pkl"
COLUMNS_PATH = BASE_DIR / "Models" / "model_columns.pkl"




model = joblib.load(MODEL_PATH)
columns = joblib.load(COLUMNS_PATH)
#columns = joblib.load("model_columns.pkl")

# new data
new_data = pd.DataFrame({
    "gender": ["female"],
    "race/ethnicity": ["group C"],
    "parental level of education": ["bachelor's degree"],
    "lunch": ["standard"],
    "test preparation course": ["completed"],
    "reading score": [80],
    "writing score": [73]
})

# one-hot encoding
new_data = pd.get_dummies(new_data, drop_first=True)

# ⭐ مهم‌ترین خط:
new_data = new_data.reindex(columns=columns, fill_value=0)

# prediction
prediction = model.predict(new_data)

print("Predicted score:", prediction[0])