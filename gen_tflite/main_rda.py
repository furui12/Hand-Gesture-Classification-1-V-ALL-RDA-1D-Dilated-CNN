import os
import glob
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.linalg import toeplitz, solve
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import log_loss, roc_curve

DIRECTORY_PATH = "./emg_data_directory" # PLEASE CHANGE TO THE DIRECTORY OF CSV FILES WITH EMG DATA!

# ==============================================================================
# 1. FEATURE EXTRACTION FUNCTIONS
# ==============================================================================

def wamp(x, threshold=0.01):
    """Willison Amplitude"""
    return np.sum(np.abs(np.diff(x)) > threshold)

def zc(x, threshold=0.01):
    """Zero Crossings"""
    crossings = np.sum(np.abs(np.diff(np.sign(x))) > 0)
    return crossings

def ssc(x, threshold=0.01):
    """Slope Sign Changes"""
    diff = np.diff(x)
    return np.sum((diff[:-1] * diff[1:] < 0) & (np.abs(diff[:-1]) > threshold) & (np.abs(diff[1:]) > threshold))

def ar_coeffs(x, order=4):
    """4th-order Auto-Regressive coefficients using Yule-Walker equations"""
    r = np.correlate(x, x, mode='full')
    r = r[len(x)-1 : len(x)-1+order+1]
    if r[0] == 0:
        return np.zeros(order)
    R = toeplitz(r[:-1])
    try:
        a = solve(R, -r[1:])
        return a
    except np.linalg.LinAlgError:
        return np.zeros(order)

def extract_window_features(window):
    """Extracts Hudgins set, RMS, WAMP, Log-Var, and AR(4) for a 1D array."""
    mav = np.mean(np.abs(window))
    wl = np.sum(np.abs(np.diff(window)))
    zc_val = zc(window)
    ssc_val = ssc(window)
    rms = np.sqrt(np.mean(window**2))
    wamp_val = wamp(window)
    log_var = np.log(np.var(window) + 1e-8)
    ar = ar_coeffs(window, order=4)

    # return np.array([mav, wl, zc_val, ssc_val, rms, wamp_val, log_var, *ar])
    return np.array([rms, wamp_val, log_var, *ar])

# ==============================================================================
# 2. DATA LOADING & SEGMENTATION
# ==============================================================================

def load_and_segment_data(directory_path):
    """
    Iteratively parses CSV files to ensure memory efficiency.
    Segments data (35% around midpoint of 3s intervals) and windows it.
    Randomly drops ~75% of 'Rest' (label 0) data to balance classes.
    """
    csv_files = glob.glob(os.path.join(directory_path, '*.csv'))
    label_map = {"Power": 1, "Key": 2, "Pinch": 3, "Tripod": 4}

    X_features = []
    y_labels = []

    fs = 2000
    trial_length = 3 * fs
    trim_samples = int(trial_length * 0.15)
    start_offset = trim_samples
    end_offset = trial_length - trim_samples

    window_size = int(0.200 * fs)
    step_size = int(0.050 * fs)

    for file in csv_files:
        filename = os.path.basename(file)
        task_label = 0
        for key, val in label_map.items():
            if key.lower() in filename.lower():
                task_label = val
                break

        if task_label == 0:
            continue

        try:
            df = pd.read_csv(file, usecols=[9, 15], header=0)
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue

        data = df.values
        if data.shape[0] < 36000:
            continue

        for i in range(6):
            current_label = 0 if i % 2 == 0 else task_label
            seg_start = (i * trial_length) + start_offset
            seg_end = (i * trial_length) + end_offset
            segment = data[seg_start:seg_end, :]

            for w_start in range(0, segment.shape[0] - window_size + 1, step_size):
                if current_label == 0:
                    continue

                w_end = w_start + window_size
                window = segment[w_start:w_end, :]

                ch1_features = extract_window_features(window[:, 0])
                ch2_features = extract_window_features(window[:, 1])
                combined_features = np.concatenate([ch1_features, ch2_features])

                X_features.append(combined_features)
                y_labels.append(current_label - 1)

    return np.array(X_features), np.array(y_labels)

# ==============================================================================
# 3. HYPERPARAMETER TUNING & THRESHOLDING
# ==============================================================================

def tune_rda_shrinkage(X_train, y_train, X_val, y_val):
    shrinkage_values = np.arange(0, 1.025, 0.025)
    best_loss = float('inf')
    best_shrinkage = 0.0

    for gamma in shrinkage_values:
        model = OneVsRestClassifier(LinearDiscriminantAnalysis(solver='lsqr', shrinkage=gamma))
        model.fit(X_train, y_train)
        probas = model.predict_proba(X_val)
        loss = log_loss(y_val, probas)

        if loss < best_loss:
            best_loss = loss
            best_shrinkage = gamma

    return best_shrinkage

def find_rejection_thresholds(y_true, y_proba_matrix, target_fpr=5e-4):
    n_classes = y_proba_matrix.shape[1]
    thresholds = {}

    for c in range(n_classes):
        y_binary = (y_true == c).astype(int)
        probas = y_proba_matrix[:, c]
        fpr, tpr, thresh = roc_curve(y_binary, probas)
        valid_idx = np.where(fpr <= target_fpr)[0]

        if len(valid_idx) > 0:
            best_idx = valid_idx[np.argmax(tpr[valid_idx])]
            selected_threshold = thresh[best_idx]
        else:
            selected_threshold = 0.995

        thresholds[c] = selected_threshold

    return thresholds

# --- Export to TFLite ---

def export_to_tflite(final_model, n_features, filename="gesture_rda_model.tflite"):
    """
    Extracts weights from a scikit-learn OneVsRest(LDA) model, maps them to an
    equivalent Keras architecture (unnormalized sigmoids), and exports to a TFLite file.
    Normalization must be handled in C++.
    """
    print("\nConverting scikit-learn RDA model to TensorFlow Lite...")

    n_classes = len(final_model.estimators_)
    kernel = np.zeros((n_features, n_classes))
    bias = np.zeros((n_classes,))

    for i, estimator in enumerate(final_model.estimators_):
        kernel[:, i] = estimator.coef_[0]
        bias[i] = estimator.intercept_[0]

    # Build equivalent TensorFlow network without Sum/Divide ops to ensure ESP32 compatibility
    inputs = tf.keras.Input(shape=(n_features,))
    outputs = tf.keras.layers.Dense(n_classes, activation='sigmoid')(inputs)

    keras_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    keras_model.set_weights([kernel, bias])

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(filename, 'wb') as f:
        f.write(tflite_model)

    print(f"Model successfully saved as {filename}. (Note: Outputs are unnormalized sigmoids).")

def export_to_tflite2(final_model, n_features, filename="gesture_rda_model.tflite"):
    """
    Extracts weights from a scikit-learn OneVsRest(LDA) model, maps them to an
    equivalent Keras architecture, and exports to a TFLite file.
    """
    print("\nConverting scikit-learn RDA model to TensorFlow Lite...")

    n_classes = len(final_model.estimators_)
    kernel = np.zeros((n_features, n_classes))
    bias = np.zeros((n_classes,))

    # Extract linear boundaries from scikit-learn
    for i, estimator in enumerate(final_model.estimators_):
        kernel[:, i] = estimator.coef_[0]
        bias[i] = estimator.intercept_[0]

    # Build equivalent TensorFlow network
    inputs = tf.keras.Input(shape=(n_features,))
    # The sigmoid activation mimics binary probabilities of LDA
    x = tf.keras.layers.Dense(n_classes, activation='sigmoid')(inputs)
    # Replicate scikit-learn's OneVsRest probability normalization (probabilities sum to 1)
    # ---NEW---
    # sum_probs = tf.keras.layers.Lambda(lambda t: tf.reduce_sum(t, axis=1, keepdims=True))(x)
    sigmoids = tf.keras.layers.Dense(n_classes, activation='sigmoid', name="rda_sigmoids")(inputs)
    # outputs = tf.keras.layers.Divide()([x, sum_probs])
    outputs = tf.keras.layers.Lambda(
        lambda x: x / tf.reduce_sum(x, axis=1, keepdims=True),
        name="probability_normalization"
    )(sigmoids)

    keras_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    keras_model.set_weights([kernel, bias])

    # Convert and optimize for edge devices
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(filename, 'wb') as f:
        f.write(tflite_model)

    print(f"Model successfully saved as {filename}")

# ==============================================================================
# 5. EXECUTION
# ==============================================================================

print("Loading, balancing, and extracting features...")
X, y = load_and_segment_data(DIRECTORY_PATH)

print(f"Extracted feature matrix shape: {X.shape}")

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
fold = 1
all_accuracies = []

for train_index, test_index in skf.split(X, y):
    print(f"\n--- Fold {fold} ---")
    X_train_full, X_test = X[train_index], X[test_index]
    y_train_full, y_test = y[train_index], y[test_index]

    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=42)

    print("Tuning RDA shrinkage parameter...")
    best_gamma = tune_rda_shrinkage(X_train, y_train, X_val, y_val)
    print(f"Best shrinkage parameter selected: {best_gamma}")

    final_model = OneVsRestClassifier(LinearDiscriminantAnalysis(solver='lsqr', shrinkage=best_gamma))
    final_model.fit(X_train_full, y_train_full)

    y_test_proba = final_model.predict_proba(X_test)

    print("Calculating confidence-based rejection thresholds...")
    thresholds = find_rejection_thresholds(y_test, y_test_proba, target_fpr=5e-4)

    y_pred = final_model.predict(X_test)
    accepted_predictions = 0
    correct_predictions = 0

    for i in range(len(y_test)):
        predicted_class = y_pred[i]
        confidence = y_test_proba[i, predicted_class]

        if confidence >= thresholds[predicted_class]:
            accepted_predictions += 1
            if predicted_class == y_test[i]:
                correct_predictions += 1

    acceptance_rate = accepted_predictions / len(y_test)
    conditional_accuracy = correct_predictions / accepted_predictions if accepted_predictions > 0 else 0.0

    all_accuracies.append(conditional_accuracy)
    print(f"Acceptance Rate: {acceptance_rate:.4f}")
    print(f"Accuracy on Accepted Predictions: {conditional_accuracy:.4f}")
    fold += 1

    for c, t in thresholds.items():
        print(f"  Class {c} Threshold: {t:.4f}")

print(f"\nAverage Conditional Accuracy across 10 folds: {np.mean(all_accuracies):.4f}")

# --- Showcase 10 Random Samples ---
print("\n--- Final Model Showcase (10 Random Samples from Fold 10 Validation Set) ---")
#label_names = {0: "Rest", 1: "Power", 2: "Key", 3: "Pinch", 4: "Tripod"}
label_names = {0: "Power", 1: "Key", 2: "Pinch", 3: "Tripod"}
n_samples = min(10, len(X_val))

if n_samples > 0:
    sample_indices = np.random.choice(len(X_val), n_samples, replace=False)
    X_sample = X_val[sample_indices]
    y_sample_true = y_val[sample_indices]

    y_sample_pred = final_model.predict(X_sample)
    y_sample_proba = final_model.predict_proba(X_sample)

    # print(f"{'Actual':<10} | {'Predicted':<10} | {'Confidence':<12} | {'Status':<10} | {'Correct?':<10}")
    print(f"{'Actual':<10} | {'Predicted':<10} | {'Prediction':<10} | {'Confidence':<12} | {'Status':<10}")
    print("-" * 62)

    for i in range(n_samples):
        actual_class = y_sample_true[i]
        pred_class = y_sample_pred[i]
        pred_class_name = label_names[pred_class]
        confidence = y_sample_proba[i, pred_class]

        actual_name = label_names.get(actual_class, "Unknown")
        pred_name = label_names.get(pred_class, "Unknown")

        if confidence >= thresholds[pred_class]:
            status = "Accepted"
            is_correct = "Yes" if actual_class == pred_class else "No"
        else:
            status = "Rejected"
            is_correct = "N/A"
            pred_name = "N/A"

        # print(f"{actual_name:<10} | {pred_name:<10} | {confidence:.4f}       | {status:<10} | {is_correct:<10}")
        print(f"{actual_name:<10} | {pred_name:<10} | {pred_class_name:<10} | {confidence:.4f}       | {is_correct:<10}")

# --- TFLite Export ---
# Convert the model from the final fold into a format usable by the ESP32
export_to_tflite(final_model, n_features=X.shape[1])
