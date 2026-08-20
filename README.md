# Hand-Gesture-Classification-1-V-ALL-RDA-1D-Dilated-CNN
AI-Based Control System for Hand Gesture Recognition Using 2 EMG Channels, 1-V-ALL RDA Gatekeeper Model, and 1D Dilated CNN Second Opinion Model

## Abstract
Hand gesture classification is crucial for proper control of human-assistive robotics, such as smart prostheses. Ownership of these devices by patients starts with efficient control, the most direct means for which can be realized using a robust algorithm capable of eliminating false positives and utilizing the most effective feature sets in the input data to properly distinguish between the different intended gestures, thereby optimizing the number of true positive predictions it can act upon. However, one problem that persists in AI-based control systems is the need to capture high-density high quality data in order to capture all key differences pertaining to the key distinctions that the algorithm is meant to discover during training. Finding a means of capturing as much high quality data only from pertinent regions in the most efficient way is necessary for the creation of smart prosthetic designs that will not require as many sensors, which can assist in reducing the overall cost of the finalized products. This research shows that a system could consist of only two EMG sensors and an ESP32 microcontroller, leading the potential input and processing costs of such a robotic system to be possible under $100 USD and can be made to be very robust using the first model's rejection thresholds to reject data it cannot guarantee pertains to the target motor intention of the individual and be made more performant through use of a second opinion 1D Dilated CNN model.

## Project
### Code
The two Python files are for the development of TFLite files and model quantization parameters for use in a C++ file that has been confirmed to run on a SEEED Xiao ESP32-S3 Sense when all data input and parameters are provided. The Python scripts can be found in the `gen_tflite` directory, and the two C++ scripts pertaining to this project can be found in the `esp32_proc` directory.

To use the Python files:
1) Prepare the dual-channel EMG training data in CSV files, ***ensure that pre-processing methods are done on the data before training the models*** (ex. using a 60 Hz notch filter and ensuring that the mid-point of the data exists at the zero line so features like "zero crossings" can be properly utilized, using blind source separation methods to select for source signal, etc.).
2) Provide the directory path name in the global variable `DIRECTORY_PATH`.
3) Ensure that all imported modules are installed and properly working on your installation of Python or Python virtual environment.
4) Change the `load_and_process_data()` and `load_and_segment_data()` functions from both Python files to load in the correct data from the CSV files for use in training the models that will be converted into TFLite models. ***(Please note that the current functions are set up to find all .csv files in the directory and specifically select the EEG data from the 10th and 16th columns, as these were the columns pertaining to the two EMG sensors' data that were used in the original experimentation. Please change accordingly to have it select the columns of data contained in your .csv files.)***.
5) Run the `main_rda.py` script on a smaller CSV dataset (typically considered a "calibration" dataset or dataset particular to an individual) and run the `main_cnn.py` script on a larger, more generalized CSV dataset, making sure to set aside a small dataset for final model validation before 
6) Run `xxd -i [name_of_tflite_model].tflite > [name_of_tflite_model].h` to create the C++ arrays that can be used by the ESP32 microcontroller in the C++ scripts.

To use the C++ files:
1) The `esp32_csv.ino` was added to allow the user to collect data directly from the sensors utilized in their setup so that all incoming data and data acquisition methods could remain consistent.
2) Place the two `[name_of_tflite_model].h` files in the working directory where you have the `esp32_emg.ino` file. (The advised method is to use the Arduino software to create a working directory for your project, copy all of the code from `esp32_emg.ino` and paste it into the main file in the newly created directory, and place the two .h files into this working directory.)
3) Change the "model headers" section's file names to ensure that they have the same names as the `[name_of_tflite_model].h` files you have in your working directory.
4) Verify that all sensor pin declarations match the proper naming methods that are laid out in your ESP32's documentation for your current wiring setup.
5) Implement the preprocessing methods into the `preprocess_emg_signals()` function to ensure that the data matches the quality of the data provided to the Python scripts for training the machine learning algorithms.
6) Finally, ensure that the gesture commands actuate the fingers to the correct degree (the implementation and degree of motion for the motors is almost certainly necessary to adjust, and may require independent testing to determine what degree measurements are necessary for each gesture), then compile and execute the code to be run on the connected ESP32, move the ESP32 to your setup to be powered by a battery or independent DC power source, and test on a user for validation.

### Setup
This project will require, at minimum:
1) An ESP32 or similar small, affordable, and capable microcontroller.
2) Two EMG sensors (the ones used during original testing were Myoware 2.0 Muscle Sensors).
3) A battery to power the EMG sensors and ESP32 circuit (which can eliminate the need for a notch filter in preprocessing steps).
4) A smart prosthetic arm with motors for independent control (flexion/extension) of finger actuation (the one used during original testing is the InMoov robot hand with six MG995 motors for finger and wrist actuation).
5) An independent power source and circuit for control of the smart prosthetic arm's motors that can be controlled by the ESP32 either via controlling a relay or a verified circuit that will allow enough current to be supplied to the motors without frying the ESP32's board and has all circuits connected to the ESP32 sharing a common ground.
6) Jumper wires for all connections and resistors to drop down the voltage readouts of the EMG sensors so that the range of possible voltage readouts fall within the voltage range that powers the ESP32 and capacitors for cleaning the EMG signal, if your sensors require them.

## Notes
This project was first verified on EMG data collected by the UC Davis laboratories, and the dataset can be found at: https://zenodo.org/records/15420178

## Citations
1) Young PR, Hong K, Winslow EJ, Sagastume GK, Battraw MA, Whittle RS, Schofield JS. (2025) The effects of limb position and grasped load on hand gesture classification using electromyography, force myography, and their combination. PLoS ONE 20(4): e0321319. https://doi.org/10.1371/journal.pone.0321319
2) Young, Peyton R., Kihun Hong, Eden J. Winslow, Giancarlo K. Sagastume, Marcus A. Battraw, Richard S. Whittle, and Jonathon S. Schofield. 2025. “The Effects of Limb Position and Grasped Load on Hand Gesture Classification Using Electromyography, Force Myography, and Their Combination.” PLOS ONE 20(4):e0321319. doi:10.1371/journal.pone.0321319
3) Krasoulis, A., Vijayakumar, S., & Nazarpour, K. (2020). Multi-Grip Classification-Based Prosthesis Control With Two EMG-IMU Sensors. IEEE Transactions on Neural Systems and Rehabilitation Engineering, 28(2), 508–518. https://doi.org/10.1109/TNSRE.2019.2959243
4) Jaramillo-Yánez, Andrés, Marco E. Benalcázar, and Elisa Mena-Maldonado. 2020. “Real-Time Hand Gesture Recognition Using Surface Electromyography and Machine Learning: A Systematic Literature Review.” Sensors 20(9):2467. doi:10.3390/s20092467
5) Aarotale, Parshuram N., and Ajita Rattani. 2024. “Machine Learning-Based sEMG Signal Classification for Hand Gesture Recognition.” doi: 10.48550/arXiv.2411.15655
