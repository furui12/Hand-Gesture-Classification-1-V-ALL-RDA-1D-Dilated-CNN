import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

DIRECTORY_PATH = "./emg_data_directory" # PLEASE CHANGE TO THE DIRECTORY OF CSV FILES WITH EMG DATA!

# ==============================================================================
# 1. FEATURE EXTRACTION FUNCTIONS
# ==============================================================================

def extract_window_features(window):
    """
    Extracts 10 specialized mathematical time-domain and energy features
    from a single window of sEMG data.
    """
    N = len(window)
    eps = 1e-8  # Prevents division by zero

    d1 = np.diff(window)
    d2 = np.diff(d1)

    m0 = np.mean(window**2)
    m2 = np.mean(d1**2)
    m4 = np.mean(d2**2)

    root_m0 = np.sqrt(m0)
    root_m2 = np.sqrt(m2)
    root_m4 = np.sqrt(m4)

    irregularity_factor = m2 / (np.sqrt(m0 * m4) + eps)

    norm_m0 = m0 / (m4 + eps)
    norm_m2 = m2 / (m4 + eps)

    l1_norm = np.sum(np.abs(window))
    l2_norm = np.sqrt(np.sum(window**2))
    sparseness = (np.sqrt(N) - (l1_norm / (l2_norm + eps))) / (np.sqrt(N) - 1 + eps)

    waveform_length = np.sum(np.abs(d1))
    wlr = waveform_length / (N - 1)

    abs_win = np.abs(window)
    cv = np.std(abs_win) / (np.mean(abs_win) + eps)

    tkeo_sequence = window[1:-1]**2 - window[:-2] * window[2:]
    tkeo_feat = np.mean(tkeo_sequence)

    return [root_m0, root_m2, root_m4, irregularity_factor, norm_m0, norm_m2, sparseness, wlr, cv, tkeo_feat]

# ==============================================================================
# 2. DATA LOADING & SEGMENTATION
# ==============================================================================

def load_and_process_dataset(directory_path):
    # Updated to 4 classes, indexed 0 to 3
    gesture_mapping = {"Power": 0, "Key": 1, "Pinch": 2, "Tripod": 3}

    X_sequences = []
    y_labels = []

    csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in directory: {directory_path}")

    for file_name in csv_files:
        assigned_gesture = None
        for gesture_name in gesture_mapping.keys():
            if gesture_name.lower() in file_name.lower():
                assigned_gesture = gesture_mapping[gesture_name]
                break

        if assigned_gesture is None:
            continue

        file_path = os.path.join(directory_path, file_name)

        try:
            df = pd.read_csv(file_path, usecols=[9, 15], header=0)
        except Exception as e:
            print(f"Skipping corrupt or misconfigured file {file_name}: {e}")
            continue

        if len(df) != 36000:
            print(f"File {file_name} contains {len(df)} rows instead of 36000 data rows. Skipping.")
            continue

        ch1_data = df.iloc[:, 0].to_numpy()
        ch2_data = df.iloc[:, 1].to_numpy()

        for cycle in range(3):
            cycle_offset = cycle * 12000

            # Skip the Rest phase entirely; only target the Active Gesture phase (starts at +6000)
            phase_start = cycle_offset + 6000

            # Isolate 70% central trial data (extract indices 900 to 5100 inside the phase window)
            start_idx = phase_start + 900
            end_idx = phase_start + 5100

            seg_ch1 = ch1_data[start_idx:end_idx]
            seg_ch2 = ch2_data[start_idx:end_idx]

            window_size = 400
            step_size = 10
            trial_features = []

            for w_start in range(0, len(seg_ch1) - window_size + 1, step_size):
                w_end = w_start + window_size

                f_ch1 = extract_window_features(seg_ch1[w_start:w_end])
                f_ch2 = extract_window_features(seg_ch2[w_start:w_end])

                combined_features = f_ch1 + f_ch2
                trial_features.append(combined_features)

            X_sequences.append(trial_features)
            y_labels.append(assigned_gesture)

    return np.array(X_sequences, dtype=np.float32), np.array(y_labels, dtype=np.int32)

# ==============================================================================
# 3. TRAINING ENVIRONMENT INITIALIZATION
# ==============================================================================

print("Executing extraction and balanced segmentation pipeline...")
X, y = load_and_process_dataset(DIRECTORY_PATH)
print(f"Data pipeline finished. Active Dataset Shapes - X: {X.shape}, y: {y.shape}")

unique_classes, counts = np.unique(y, return_counts=True)
for cls, count in zip(unique_classes, counts):
    print(f" Class {cls} Sample Count: {count}")

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_split=0.2, random_state=42, stratify=y
)

num_samples_train, timesteps, num_features = X_train.shape
num_samples_val, _, _ = X_val.shape

scaler = StandardScaler()
X_train_reshaped = X_train.reshape(-1, num_features)
X_val_reshaped = X_val.reshape(-1, num_features)

X_train_scaled = scaler.fit_transform(X_train_reshaped).reshape(num_samples_train, timesteps, num_features)
X_val_scaled = scaler.transform(X_val_reshaped).reshape(num_samples_val, timesteps, num_features)

# ==============================================================================
# 4. 1D DILATED CNN MODEL ARCHITECTURE
# ==============================================================================

def build_dilated_cnn(input_shape, num_classes=4):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=32, kernel_size=3, dilation_rate=1,
                               padding='causal', activation='relu', input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Conv1D(filters=64, kernel_size=3, dilation_rate=2,
                               padding='causal', activation='relu'),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Conv1D(filters=128, kernel_size=3, dilation_rate=4,
                               padding='causal', activation='relu'),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.GlobalAveragePooling1D(),

        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

input_temporal_shape = (timesteps, num_features)
# Initialized strictly for 4 classes
model = build_dilated_cnn(input_temporal_shape, num_classes=4)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
]

print("Beginning model training configuration...")
model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=50,
    batch_size=32,
    callbacks=callbacks
)

# ==============================================================================
# 5. EXECUTION
# ==============================================================================

print("Generating evaluation metrics and plotting confusion matrix...")

y_pred_probs = model.predict(X_val_scaled)
y_pred = np.argmax(y_pred_probs, axis=-1)

# Removed "Rest" from the visual labels
class_names = ["Power", "Key", "Pinch", "Tripod"]
cm = confusion_matrix(y_val, y_pred)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm_normalized, annot=True, fmt=".2f", cmap="Blues",
    xticklabels=class_names, yticklabels=class_names
)
plt.title("sEMG Gesture Recognition - Validation Confusion Matrix")
plt.ylabel("True Class Gesture")
plt.xlabel("Predicted Class Gesture")
plt.tight_layout()

output_plot_path = "confusion_matrix.png"
plt.savefig(output_plot_path, dpi=300)
plt.close()
print(f"Validation matrix visualization saved cleanly as: '{output_plot_path}'")

# ------------------------------------------------------------------------------

print("Generating Representative Dataset for INT8 calibration...")
def representative_data_gen():
    # Use 100 samples from the training set to calibrate activation ranges
    for i in range(min(100, len(X_train_scaled))):
        yield [X_train_scaled[i:i+1].astype(np.float32)]

print("Compiling model into FULL INT8 TensorFlow Lite format...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen

# Enforce integer-only operations
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

tflite_output_path = "emg_gesture_dilated_cnn_int8.tflite"
with open(tflite_output_path, "wb") as f:
    f.write(tflite_model)
print(f"Export Success. Flash asset deployed cleanly at: {tflite_output_path}")

# --- Extract parameters for C++ deployment ---
print("Extracting Scaler and Quantization parameters for C++ deployment...")

# Initialize TFLite interpreter to read I/O details
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

input_scale, input_zero_point = input_details['quantization']
output_scale, output_zero_point = output_details['quantization']

# Format StandardScaler arrays into a string
mean_str = ", ".join([f"{x:.6f}" for x in scaler.mean_])
scale_str = ", ".join([f"{x:.6f}" for x in scaler.scale_])

cpp_export_string = f"""// ESP32-S3 Sense C++ Deployment Parameters
// Auto-generated configuration for sEMG INT8 model deployment.
// Copy and paste this block into your Arduino (.ino) project.

#include <stdint.h>

// --- STANDARD SCALER PARAMETERS ---
// Apply to incoming features before quantization:
// scaled_feature[i] = (raw_feature[i] - SCALER_MEAN[i]) / SCALER_SCALE[i]
const int NUM_FEATURES = {num_features};
const float SCALER_MEAN[{num_features}] = {{ {mean_str} }};
const float SCALER_SCALE[{num_features}] = {{ {scale_str} }};

// --- MODEL QUANTIZATION PARAMETERS ---
// Convert scaled float inputs to INT8:
// int8_input = round((scaled_feature / INPUT_SCALE) + INPUT_ZERO_POINT)
const float INPUT_SCALE = {input_scale:.6f}f;
const int8_t INPUT_ZERO_POINT = {input_zero_point};

// Convert INT8 outputs back to floats (probabilities):
// float_probability = (int8_output - OUTPUT_ZERO_POINT) * OUTPUT_SCALE
const float OUTPUT_SCALE = {output_scale:.6f}f;
const int8_t OUTPUT_ZERO_POINT = {output_zero_point};
"""

txt_output_path = "esp32_deployment_params.txt"
with open(txt_output_path, "w") as text_file:
    text_file.write(cpp_export_string)

print(f"Deployment parameters saved cleanly as: '{txt_output_path}'")
