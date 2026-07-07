
#program 3


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Read CSV
data = pd.read_csv("job_offer.csv")

# Encoding
encoder = LabelEncoder()

for column in data.columns:
    data[column] = encoder.fit_transform(data[column])

# Features and Target
X = data.drop("JobOffer", axis=1)
y = data["JobOffer"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create KNN Model
knn = KNeighborsClassifier(n_neighbors=3)

# Train
knn.fit(X_train, y_train)

# Prediction
y_pred = knn.predict(X_test)

# Accuracy
print("Accuracy :", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix\n")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report\n")
print(classification_report(y_test, y_pred))