import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

subject_ids = [f"S{i}" for i in range(2, 18) if i != 12]  # Subjects S2 to S17 excluding S12
base_path = 'data/subject_feats/'

# Initialize storage for results
accuracy_scores = {}
classification_reports = {}

for subject_id in subject_ids:
    print(f"\nProcessing Subject: {subject_id}")

    # Load the data
    file_path = f"{base_path}{subject_id}_feats_4.csv"
    data = pd.read_csv(file_path)

    # Assuming '2' is the label for the stress condition
    X = data.drop('2', axis=1)  # Drop the label column to form the feature set
    y = data['2']  # Use the '2' column as the target variable

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=10000)

    # Initialize and train the Random Forest Classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)

    # Predictions
    y_pred = rf_classifier.predict(X_test)

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



