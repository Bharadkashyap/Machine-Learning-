import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load the Excel dataset
data = pd.read_excel("Multivariate_Linear_Regression_Dataset.xlsx")

# Inspect the first few rows
print(data.head())

# Define features (X) and target (y)
X = data[["Square Feet", "Number of Bed Rooms"]]
y = data["Price of House"]

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Print coefficients and intercept
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

# Example prediction: House with 2000 sqft and 3 bedrooms
predicted_price = model.predict([[2000, 3]])
print("Predicted Price:", predicted_price[0])

# Optional: Plot actual vs predicted values
y_pred = model.predict(X)
plt.scatter(y, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()
