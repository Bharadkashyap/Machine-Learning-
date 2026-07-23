import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

# Load the Excel dataset
data = pd.read_excel("Multivariate_Linear_Regression_Dataset.xlsx")

# Define features (X) and target (y)
X = data[["Square Feet", "Number of Bed Rooms"]]
y = data["Price of House"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create KNN regressor
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)

# Predict on test set
y_pred = knn.predict(X_test)

# Show results in two feature columns + final predicted price
results = pd.DataFrame({
    "Square Feet": X_test["Square Feet"],
    "Bedrooms": X_test["Number of Bed Rooms"],
    "Predicted Price": y_pred
})

print(results.head(10))  # show first 10 predictions

# Example prediction: House with 2000 sqft and 3 bedrooms
predicted_price = knn.predict([[2000, 3]])
print("Predicted Price for 2000 sqft, 3 bedrooms:", predicted_price[0])
