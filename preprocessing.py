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

def validate_hypothesis_v2(data_baseline, data_stress, signal_type, sample_fraction=0.1):
    """Compare the mean of baseline and stress data and validate hypothesis using a fraction of the data."""
    # Extract a smaller sample
    sample_size = int(sample_fraction * len(data_baseline))
    sample_baseline = np.random.choice(data_baseline, sample_size, replace=False)
    sample_stress = np.random.choice(data_stress, sample_size, replace=False)
    
    # Hypothesis Testing on the sample
    z, p_value = z_test(sample_baseline, sample_stress)
    
    print(f"\nHypothesis Test on {sample_fraction*100}% of {signal_type} data:")
    print(f"Z-score: {z}, p-value: {p_value}")
    
    # Decision based on p-value
    alpha = 0.05
    if p_value < alpha:
        print(f"Reject the null hypothesis for {signal_type} on the sample: There is a significant difference between the means.")
        # Validation on the entire dataset
        mean_baseline = np.mean(data_baseline)
        mean_stress = np.mean(data_stress)
        if mean_baseline != mean_stress:
            print(f"Validation on entire dataset supports this, as means of {signal_type} Baseline and Stress are different.")
        else:
            print(f"Validation on entire dataset contradicts this, as means of {signal_type} Baseline and Stress are the same.")
    else:
        print(f"Fail to reject the null hypothesis for {signal_type} on the sample: There isn't a significant difference between the means.")
        # Validation on the entire dataset
        mean_baseline = np.mean(data_baseline)
        mean_stress = np.mean(data_stress)
        if mean_baseline == mean_stress:
            print(f"Validation on entire dataset supports this, as means of {signal_type} Baseline and Stress are the same.")
        else:
            print(f"Validation on entire dataset contradicts this, as means of {signal_type} Baseline and Stress are different.")

subjects = ["S" + str(i) for i in range(2, 18)]
subjects.remove("S12")

ecg_correlations = []
eda_correlations = []
emg_correlations = []

for subject in subjects:
    
    # Load data from .pkl file
    pkl_path = os.path.join(subject, subject + '.pkl')
    with open(pkl_path, 'rb') as file:
        data = pickle.load(file, encoding='latin1')

    # Extracting chest data
    chest_data = data['signal']['chest']
    
    ecg_raw = chest_data['ECG']
    eda_raw = chest_data['EDA']
    emg_raw = chest_data['EMG']

    # Drop NaN values
    ecg_raw = ecg_raw[~np.isnan(ecg_raw)]
    eda_raw = eda_raw[~np.isnan(eda_raw)]
    emg_raw = emg_raw[~np.isnan(emg_raw)]

    # Normalize the signals
    ecg_raw = normalize_data(ecg_raw.ravel())
    eda_raw = normalize_data(eda_raw.ravel())
    emg_raw = normalize_data(emg_raw.ravel())

    

    # Extract indices for baseline and stress conditions
    labels = data['label']
    baseline_indices = np.where(labels == 1)[0]
    stress_indices = np.where(labels == 2)[0]

    # Extract data for these conditions and combine
    ecg_combined = np.concatenate([ecg_raw[baseline_indices], ecg_raw[stress_indices]])
    eda_combined = np.concatenate([eda_raw[baseline_indices], eda_raw[stress_indices]])
    emg_combined = np.concatenate([emg_raw[baseline_indices], emg_raw[stress_indices]])
    combined_labels = np.concatenate([np.zeros(len(baseline_indices)), np.ones(len(stress_indices))])

    # Compute correlation for the combined data and combined_labels
    ecg_correlation, _ = pearsonr(ecg_combined, combined_labels)
    eda_correlation, _ = pearsonr(eda_combined, combined_labels)
    emg_correlation, _ = pearsonr(emg_combined, combined_labels)

    ecg_correlations.append(ecg_correlation)
    eda_correlations.append(eda_correlation)
    emg_correlations.append(emg_correlation)

    print(f"\nSubject: {subject}")
    print(f"ECG-Baseline vs Stress Correlation: {ecg_correlation}")
    print(f"EDA-Baseline vs Stress Correlation: {eda_correlation}")
    print(f"EMG-Baseline vs Stress Correlation: {emg_correlation}")

    print("\nValidation for ECG data:")
    validate_hypothesis_v2(ecg_raw[baseline_indices], ecg_raw[stress_indices], "ECG")
    
    print("\nValidation for EDA data:")
    validate_hypothesis_v2(eda_raw[baseline_indices], eda_raw[stress_indices], "EDA")
    
    print("\nValidation for EMG data:")
    validate_hypothesis_v2(emg_raw[baseline_indices], emg_raw[stress_indices], "EMG")
    
    print("----------------------------")
    print("----------------------------")

    # Plotting Pearson Coefficients for each subject
plt.figure(figsize=(12, 7))

plt.plot(subjects, ecg_correlations, '-o', label='ECG', color='blue')
plt.plot(subjects, eda_correlations, '-o', label='EDA', color='red')
plt.plot(subjects, emg_correlations, '-o', label='EMG', color='green')

plt.title('Pearson Coefficient Across Subjects')
plt.xlabel('Subject')
plt.ylabel('Pearson Coefficient')
plt.legend()
plt.grid(True)
plt.show()
