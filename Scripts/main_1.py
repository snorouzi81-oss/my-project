# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv("../Data/StudentsPerformance.csv")
print("Columns: ",df.columns)
print(df.head())
print(df.info())
print(df.isna().sum())
print(df.dtypes)

#fig, ax = plt.subplots()

sns.lineplot(data=df, x="reading score", y="math score")
plt.title("Reading vs Math Score")
plt.show()

sns.scatterplot(data=df, x="writing score", y="math score")
plt.title("Writing vs Math Score")
plt.show()

sns.boxplot(data=df, x="test preparation course", y="math score")
plt.title("Test Preparation vs Math Score")
plt.show()

sns.histplot(df["math score"], kde=True)
plt.title("Distribution of Math Scores")
plt.show()

feature_column = "writing score"
target_column = "math score"
X = df[[feature_column]]
y = df[target_column]

#print(type(X))
#print(type(y))
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

results_df = X_test.copy()
results_df["actual_score"] = y_test.values
results_df["predicted_score"] = predictions

print("Comparison table:")
print(results_df)

print("Predicted values:")
print(predictions)

print("Actual values:")
print(y_test.values)

new_data = pd.DataFrame({feature_column: [73]})
new_prediction = model.predict(new_data)
print(f"Predicted score for {new_data} writing score:", new_prediction[0])

mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f'Mean Squared Error: {mse}')
print(f'R² Score: {r2}')



