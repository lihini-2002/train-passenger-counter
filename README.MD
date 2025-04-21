# Train Passenger Counter System 🚆

An AI-powered system for counting train passengers entering and exiting using ESP32-CAM, IR sensors, and MediaPipe BlazePose.

## 🔧 Hardware Used

- ESP32-CAM
- 2x IR Beam Sensors (TCRT5000)
- Power Bank
- 3D Printed Mount

## 🧠 Software Overview

- Images captured by ESP32-CAM
- Sent to backend via MQTT
- MediaPipe BlazePose detects people
- IR sensors determine direction
- SQLite logs entry/exit count per station

## ✅ Goal

Count how many people enter or exit at each train station, even if they don't carry RFID cards.
