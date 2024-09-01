import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from dvclive import Live

# Load the dataset
file_path = 'data/student_performance.csv'
data = pd.read_csv(file_path)

# Splitting the data into features (X) and target (y)
X = data.drop(columns=['placed_or_not'])
y = data['placed_or_not']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define hyperparameters for the RandomForestClassifier
max_depth = 7
n_estimators = 80
random_state = 42

# Initialize and train the RandomForestClassifier
rf_classifier = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=random_state)
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
with Live(save_dvc_exp=True) as live:
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    live.log_metric('accuracy',accuracy)
    live.log_metric('precision',precision)
    live.log_metric('recall',recall)
    live.log_metric('f1',f1)
    live.log_param('n_estimators',n_estimators)
    live.log_param('max_depth',max_depth)

# Print the evaluation metrics
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
