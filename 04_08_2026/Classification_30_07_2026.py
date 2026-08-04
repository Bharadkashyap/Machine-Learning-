
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Dataset
df = pd.read_excel("sample_datasets_for_classification_and_regression.xlsx",
                   sheet_name="classification ")

print(df.head())

# Remove Student ID
df = df.drop("Student_ID", axis=1)

# Label Encoding
encoder = LabelEncoder()

df["Assignment_Submitted"] = encoder.fit_transform(df["Assignment_Submitted"])
df["Internet_Access"] = encoder.fit_transform(df["Internet_Access"])
df["Result"] = encoder.fit_transform(df["Result"])

# Features and Target
X = df.drop("Result", axis=1)
y = df["Result"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42
)

# Model
model = DecisionTreeClassifier(random_state=42)

# Train
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
print("\nAccuracy :", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))