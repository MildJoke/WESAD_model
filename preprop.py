import pandas as pd
import numpy as np
from scipy.stats import pearsonr, norm
import os
import pickle
import matplotlib.pyplot as plt

def normalize_data(data):
    """Normalize data using Min-Max scaling."""
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def z_test(data1, data2):
    """Perform Z-test between two data sets."""
    mean1, mean2 = np.mean(data1), np.mean(data2)
    std1, std2 = np.std(data1), np.std(data2)
    n1, n2 = len(data1), len(data2)
    
    # Z-test formula
    z = (mean1 - mean2) / np.sqrt(std1**2/n1 + std2**2/n2)
    p_value = 2 * (1 - norm.cdf(np.abs(z)))  # two-tailed p-value
    return z, p_value

subjects = ["S" + str(i) for i in range(2, 18)]
subjects.remove("S12")

for subject in subjects:
    
    # Load data from .pkl file
    pkl_path = os.path.join(subject, subject + '.pkl')
    with open(pkl_path, 'rb') as file:
        data = pickle.load(file, encoding='latin1')

    # Extracting wrist and chest data
    wrist_data = data['signal']['wrist']
    chest_data = data['signal']['chest']

    # Extract EDA data
    eda_wrist = wrist_data['EDA']
    eda_chest = chest_data['EDA']

    # Normalize the signals
    eda_wrist = normalize_data(eda_wrist.ravel())
    eda_chest = normalize_data(eda_chest.ravel())

    # Extract indices for baseline and stress conditions
    labels = data['label']

    # Truncate data to the length of labels
    min_length = min(len(eda_wrist), len(labels))
    eda_wrist = eda_wrist[:min_length]
    labels_wrist = labels[:min_length]

    min_length = min(len(eda_chest), len(labels))
    eda_chest = eda_chest[:min_length]
    labels_chest = labels[:min_length]

    # Compute correlation for the EDA data and labels
    eda_wrist_correlation, _ = pearsonr(eda_wrist, labels_wrist)
    eda_chest_correlation, _ = pearsonr(eda_chest, labels_chest)

    # Compute Z-scores and p-values
    baseline_indices = np.where(labels == 1)[0]
    stress_indices = np.where(labels == 2)[0]

    eda_wrist_z, eda_wrist_p = z_test(eda_wrist[baseline_indices], eda_wrist[stress_indices])
    eda_chest_z, eda_chest_p = z_test(eda_chest[baseline_indices], eda_chest[stress_indices])

    print(f"Subject: {subject}")
    print(f"EDA (Wrist) vs Stress Correlation: {eda_wrist_correlation}")
    print(f"EDA (Chest) vs Stress Correlation: {eda_chest_correlation}")
    print(f"EDA (Wrist) Z-score: {eda_wrist_z}, p-value: {eda_wrist_p}")
    print(f"EDA (Chest) Z-score: {eda_chest_z}, p-value: {eda_chest_p}")
    print("----------------------------")
    print("----------------------------")
