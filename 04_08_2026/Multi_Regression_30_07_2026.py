
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Dataset
df = pd.read_excel(
    "sample_datasets_for_classification_and_regression.xlsx",
    sheet_name="multi regression "
)

print(df.head())

# Remove House ID
df = df.drop("House_ID", axis=1)

# Features
X = df.drop("Price ($ Thousand) (Y)", axis=1)

# Target
y = df["Price ($ Thousand) (Y)"]

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
print("Coefficients")
print(model.coef_)

print("\nIntercept")
print(model.intercept_)

print("\nMAE")
print(mean_absolute_error(y_test, y_pred))

print("\nMSE")
print(mean_squared_error(y_test, y_pred))

print("\nR2 Score")
print(r2_score(y_test, y_pred))

# Predict New House
new_house = [[1800, 3, 5]]

prediction = model.predict(new_house)

print("\nPredicted House Price ($ Thousand):", prediction[0])