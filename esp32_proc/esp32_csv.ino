#include <WiFi.h>
#include <WebServer.h>
#include <SPI.h>
#include <SD.h>

// ==============================================================================
// 1. PIN DEFINITIONS & HARDWARE SETUP
// ==============================================================================

// MyoWare 2.0 Sensor Pins (Analog ADC Pins)
// Connect MyoWare SIG to these pins. GND to GND, and + to 3V3.
const int MYOWARE_CH1_PIN = 1; // GPIO 1 (A0 on XIAO ESP32-S3)
const int MYOWARE_CH2_PIN = 2; // GPIO 2 (A1 on XIAO ESP32-S3)

// MicroSD Card Pins for Seeed XIAO ESP32-S3 Sense
// const int SD_SCK_PIN  = 39;
// const int SD_MISO_PIN = 47;
// const int SD_MOSI_PIN = 38;
const int SD_SCK_PIN  = 7;
const int SD_MISO_PIN = 8;
const int SD_MOSI_PIN = 9;
const int SD_CS_PIN   = 21;

/* 
 * POWER NOTE: To power this standalone via battery, you do not need a specific 
 * GPIO pin. The XIAO ESP32-S3 Sense has dedicated BAT+ and BAT- pads on the 
 * bottom of the expansion board. Solder a 3.7V LiPo battery directly to these 
 * pads. The onboard power management IC handles switching from USB to battery 
 * and handles recharging. 
 */

// ==============================================================================
// 2. GLOBAL VARIABLES
// ==============================================================================

const char* AP_SSID = "Prosthetic_Calibrator";
const char* AP_PASS = "12345678"; // PLEASE CHANGE FOR BETTER SECURITY

WebServer server(80);

// Arrays for 3 seconds of data at 1000 Hz (3000 samples)
const int NUM_SAMPLES = 3000;
uint16_t emg1_data[NUM_SAMPLES];
uint16_t emg2_data[NUM_SAMPLES];

// ==============================================================================
// 3. SINGLE PAGE APPLICATION (HTML/CSS/JS)
// ==============================================================================
// R"rawliteral(...)rawliteral" allows us to write standard HTML without escaping quotes.

const char index_html[] PROGMEM = R"rawliteral(
<!DOCTYPE html>
<html>
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prosthetic Calibration</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #121212; color: #ffffff; display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100vh; margin: 0; text-align: center; }
        h1 { font-size: 2rem; color: #4facfe; margin-bottom: 10px; }
        h2 { font-size: 1.5rem; color: #a8a8a8; min-height: 40px; }
        .timer { font-size: 5rem; font-weight: bold; margin: 20px 0; min-height: 100px; color: #00f2fe; }
        button { background: linear-gradient(to right, #4facfe 0%, #00f2fe 100%); border: none; border-radius: 50px; color: #121212; padding: 20px 40px; font-size: 1.5rem; font-weight: bold; cursor: pointer; transition: transform 0.2s; box-shadow: 0 4px 15px rgba(0, 242, 254, 0.4); }
        button:active { transform: scale(0.95); }
    </style>
</head>
<body>
    <h1 id="title">Prosthetic Calibration</h1>
    <h2 id="instruction">Connect sensors and secure the socket.</h2>
    <div id="timer" class="timer"></div>
    <button id="startBtn" onclick="startCalibration()">Calibrate</button>

    <script>
        const gestures = ["Cylindrical Wrap", "Key Pinch", "Pulp Pinch", "Tripod Pinch"];
        const REPS = 3;
        const PHASE_TIME = 3; // 3 seconds for relax and 3 seconds for task

        let gestureIdx = 0;
        let repIdx = 0;

        async function countdown(seconds) {
            const timerEl = document.getElementById("timer");
            for(let i = seconds; i > 0; i--) {
                timerEl.innerText = i;
                await new Promise(r => setTimeout(r, 1000));
            }
            timerEl.innerText = "";
        }

        async function processCycle() {
            if (gestureIdx >= gestures.length) {
                document.getElementById("title").innerText = "Calibration finished!";
                document.getElementById("title").style.color = "#00ff88";
                document.getElementById("instruction").innerText = "Data safely saved to microSD.";
                return;
            }

            let currentGesture = gestures[gestureIdx];

            // --- RELAX PHASE ---
            document.getElementById("title").innerText = `${currentGesture} (Rep ${repIdx + 1}/${REPS})`;
            document.getElementById("instruction").innerText = "Relax and prepare...";
            await countdown(PHASE_TIME);

            // --- TASK PHASE ---
            document.getElementById("instruction").innerText = "PERFORM GESTURE NOW!";
            
            // Start visual countdown and ESP32 data collection concurrently
            let timerPromise = countdown(PHASE_TIME);
            let fetchPromise = fetch('/record');
            
            // Wait for both 3 seconds to pass AND the ESP32 to finish saving
            await Promise.all([timerPromise, fetchPromise]);

            // --- SAVING/TRANSITION PHASE ---
            document.getElementById("instruction").innerText = "Saving...";
            await new Promise(r => setTimeout(r, 1000)); // Brief visual pause

            // Increment logic
            repIdx++;
            if (repIdx >= REPS) {
                repIdx = 0;
                gestureIdx++;
            }
            
            processCycle();
        }

        function startCalibration() {
            document.getElementById("startBtn").style.display = "none";
            processCycle();
        }
    </script>
</body>
</html>
)rawliteral";


// ==============================================================================
// 4. SERVER ROUTING & DATA COLLECTION
// ==============================================================================

void handleRoot() {
    server.send(200, "text/html", index_html);
}

void handleRecord() {
    // 1. Data Collection (1000 Hz for 3 seconds)
    // Using micros() ensures precision regardless of analogRead execution time.
    unsigned long nextSampleTime = micros();
    
    for (int i = 0; i < NUM_SAMPLES; i++) {
        emg1_data[i] = analogRead(MYOWARE_CH1_PIN);
        emg2_data[i] = analogRead(MYOWARE_CH2_PIN);
        
        nextSampleTime += 1000; // 1000 microseconds = 1 millisecond
        while (micros() < nextSampleTime) {
            // Tight loop to enforce exact 1000Hz sampling
            // Using a blocking wait here is safe because this request runs on Core 1
            // while the background WiFi stack runs on Core 0.
        }
    }

    // 2. Determine next available filename (e.g., /1.csv, /2.csv...)
    int fileIndex = 1;
    String filename;
    while (true) {
        filename = "/" + String(fileIndex) + ".csv";
        if (!SD.exists(filename)) {
            break; // Found an unused filename
        }
        fileIndex++;
    }

    // 3. Write arrays to microSD card
    File file = SD.open(filename, FILE_WRITE);
    if (!file) {
        server.send(500, "text/plain", "Failed to open file for writing");
        return;
    }

    file.println("EMG1,EMG2");
    for (int i = 0; i < NUM_SAMPLES; i++) {
        file.print(emg1_data[i]);
        file.print(",");
        file.println(emg2_data[i]);
    }
    file.close();

    // 4. Release the JS Promise by sending success response
    server.send(200, "text/plain", "Data saved as " + filename);
}

// ==============================================================================
// 5. MAIN SETUP & LOOP
// ==============================================================================

void setup() {
    Serial.begin(115200);
    
    // Configure ADC precision (12-bit is standard for ESP32)
    analogReadResolution(12);
    
    // Initialize SPI for Seeed ESP32-S3 Sense SD Card
    SPI.begin(SD_SCK_PIN, SD_MISO_PIN, SD_MOSI_PIN, SD_CS_PIN);
    
    if (!SD.begin(SD_CS_PIN, SPI)) {
        Serial.println("Card Mount Failed. Check SD card insertion and pins.");
    } else {
        Serial.println("SD Card initialized.");
    }

    // Setup Access Point
    WiFi.softAP(AP_SSID, AP_PASS);
    IPAddress IP = WiFi.softAPIP();
    Serial.print("AP IP address: ");
    Serial.println(IP);

    // Setup Server Routes
    server.on("/", HTTP_GET, handleRoot);
    server.on("/record", HTTP_GET, handleRecord);
    
    server.begin();
    Serial.println("HTTP server started");
}

void loop() {
    server.handleClient();
}
