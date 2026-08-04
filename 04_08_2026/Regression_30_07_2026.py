
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Dataset
df = pd.read_excel(
    "sample_datasets_for_classification_and_regression.xlsx",
    sheet_name="regression "
)

print(df.head())

# Remove Store column
df = df.drop("Store", axis=1)

# Features and Target
X = df[["Advertising Budget (Thousand) (X)"]]
y = df["Monthly Sales (Thousand) (Y)"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)

# Model
model = LinearRegression()

# Train
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("Coefficient :", model.coef_[0])
print("Intercept :", model.intercept_)
print("MAE :", mean_absolute_error(y_test, y_pred))
print("MSE :", mean_squared_error(y_test, y_pred))
print("R2 Score :", r2_score(y_test, y_pred))

# Graph
plt.scatter(X, y)

plt.plot(X, model.predict(X), linewidth=2)

plt.xlabel("Advertising Budget")
plt.ylabel("Monthly Sales")
plt.title("Simple Linear Regression")

plt.show()