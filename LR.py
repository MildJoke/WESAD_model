import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

subject_ids = [f"S{i}" for i in range(2, 18) if i != 12]  # Subjects S2 to S17 excluding S12
base_path = 'data/subject_feats/'

# Initialize storage for results
accuracy_scores = {}
classification_reports = {}

count = 0
for subject_id in subject_ids:
    print(f"\nProcessing Subject: {subject_id}")

    # Load the data
    file_path = f"{base_path}{subject_id}_feats_4.csv"
    data = pd.read_csv(file_path)

    # Select only EDA-related columns for features
    eda_columns = [col for col in data.columns if 'EDA' in col]
    X = data[eda_columns]  # Use only EDA columns as features
    y = data['2']  # Assuming '2' is the stress label

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize and train the Logistic Regression Classifier
    logistic_classifier = LogisticRegression(random_state=42)
    logistic_classifier.fit(X_train, y_train)

    # Predictions
    y_pred = logistic_classifier.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    accuracy_scores[subject_id] = accuracy
    classification_reports[subject_id] = report

    print(f"Accuracy for {subject_id}: {accuracy}")
    print(f"Classification Report for {subject_id}:\n{report}")

# Results
print("\nFinal Results:")
print("Accuracies:", accuracy_scores)

# Plot the accuracies as a line graph
plt.figure(figsize=(10, 6))
plt.plot(accuracy_scores.keys(), accuracy_scores.values(), marker='o', linestyle='-')
plt.title('Accuracy Scores for Different Subjects (Logistic Regression)')
plt.xlabel('Subject ID')
plt.ylabel('Accuracy')
plt.ylim(0, 1.0)  # Set the y-axis limit between 0 and 1
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()
plt.grid(True)  # Add grid lines
plt.show()
