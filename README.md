# Hand-Gesture-Classification-1-V-ALL-RDA-1D-Dilated-CNN
AI-Based Control System for Hand Gesture Recognition Using 2 EMG Channels, 1-V-ALL RDA Gatekeeper Model, and 1D Dilated CNN Second Opinion Model

## Abstract
Hand gesture classification is crucial for proper control of human-assistive robotics, such as smart prostheses. Ownership of these devices by patients starts with efficient control, the most direct means for which can be realized using a robust algorithm capable of eliminating false positives and utilizing the most effective feature sets in the input data to properly distinguish between the different intended gestures, thereby optimizing the number of true positive predictions it can act upon. However, one problem that persists in AI-based control systems is the need to capture as much high quality data in order to capture all key differences pertaining to the key distinctions that the algorithm is meant to discover during training. Finding a means of capturing as much high quality data from pertinent regions in the most efficient way is necessary for the creation of smart prosthetic designs that will not require as many sensors, which can assist in reducing the overall cost of the finalized products. This research shows that a system could consist of only two EMG sensors and an ESP32 microcontroller, leading the potential input and processing costs of such a robotic system to be possible under $100 USD and can be made to be very robust using the first model's rejection thresholds to reject data it cannot guarantee pertains to the target motor intention of the individual and be made more performant through use of a second opinion 1D Dilated CNN model.

## Project
### Code
I created two Python files for the development of TFLite files and model quantization parameters for use in a C++ file. The Python scripts can be found in the `gen_tflite` directory, and the two C++ scripts pertaining to this project can be found in the `esp32_proc` directory.

To use the Python files:
1) Prepare the dual-channel EMG data in CSV files, ***ensure that pre-processing methods are done on the data before training the models*** (like using a 60 Hz notch filter and ensuring that the mid-point of the data exists at the zero line so features like "zero crossings" can be properly utilized).
2) Provide the directory path name in the global variable `DIRECTORY_PATH`.
3) Ensure that all imported modules are installed and properly working on your installation of Python or Python virtual environment.
4) Change the `load_and_segment_data()` functions will load in the correct data from the CSV files for use in training the models that will be converted into TFLite models.
5) Run the `main_rda.py` script on a smaller CSV dataset (typically considered a "calibration" dataset or dataset particular to an individual) and run the `main_cnn.py` script on a larger, more generalized CSV dataset, making sure to set aside a small dataset for final model validation.
6) Run `xxd -i [name_of_tflite_model].tflite > [name_of_tflite_model].h` to create the C++ arrays that can be used by the ESP32 microcontroller in the C++ scripts.

To use the C++ files:
1) The `esp32_csv.ino` was added to allow the user to collect data directly from the sensors utilized in their setup so that all incoming data and data acquisition methods could remain consistent.
2) ...
