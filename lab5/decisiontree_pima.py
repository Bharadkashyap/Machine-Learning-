# Importing necessary libraries
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load your dataset
data = pd.read_csv("pima-indians-diabetes.csv")

# Splitting data into features and target
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Splitting data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize Decision Tree Classifier
dtc = tree.DecisionTreeClassifier(random_state=42)

# Train the model
dtc.fit(x_train, y_train)

# Predict the class labels for x_test
y_predict = dtc.predict(x_test)

# Print accuracy
print('Accuracy of Decision Tree - Test:', accuracy_score(y_test, y_predict))

# Print confusion matrix
print('\nConfusion Matrix - Test:\n', confusion_matrix(y_test, y_predict))

# Print precision, recall and f1-score
print('\nClassification Report:\n', classification_report(y_test, y_predict))
