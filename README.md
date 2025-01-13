# Gestus – Echtzeit-Gestenerkennungssystem

Gestus (Latein für „Geste“) ist ein Python-basiertes Echtzeit-Gestenerkennungssystem, das Computer Vision und maschinelles Lernen verwendet, um Handgesten zu erkennen und zu interpretieren. Das System nutzt MediaPipe für die Pose-Schätzung und TensorFlow/Keras für die Gestenklassifikation.

## Voraussetzungen

### Python-Installation
- Python 3.8 oder höher muss auf Ihrem System installiert sein
- Sie können Python von der [offiziellen Website](https://www.python.org/downloads/) herunterladen
- Überprüfen Sie Ihre Python-Installation mit folgendem Befehl:
```bash
python --version
```

### Erforderliche Bibliotheken
Installieren Sie die folgenden Abhängigkeiten mit pip:
```bash
pip install mediapipe numpy opencv-python tensorflow scikit-learn
```

## Funktionen

- Echtzeit-Gestenerkennung: Erkennt und zeigt Gesten in Echtzeit an
- LSTM-basiertes neuronales Netzwerk: Sorgt für eine präzise Gestenvorhersage
- MediaPipe-Integration: Extrahiert effizient Posen und Handmarkierungen
- Visuelles Feedback: Zeigt Keypoints und erkannte Gesten auf dem Bildschirm an.

## Projektstruktur

```
gestus/
├── gestus.py          # Hauptanwendungsdatei
├── gesten1.h5         # Vorgefertigtes Modell
├── datensatz1.zip     # Komprimierte Datensatzdatei
└── datensatz1/        # Datensatzordner (nach dem Entpacken)
```

## Konfiguration

Das System kann mit den folgenden Parametern konfiguriert werden:

- `model_path`: Pfad zur Datei mit den trainierten Modellgewichten (*.h5)
- `dataset_path`: Verzeichnis, das die Gestenklassen enthält
- `threshold`: Schwellenwert für die Konfidenz bei der Gestenerkennung (Standard: 0,6)

## Verwendung

1. Klonen Sie das Repository:
```bash
git clone git clone https://github.com/zazabuzaladze/gestus.git
cd gestus
```
2. Entpacken Sie den Datensatz:
   - Extrahieren Sie die Datei datensatz1.zip im Projektverzeichnis
   - Stellen Sie sicher, dass der extrahierte Ordner datensatz1 heißt
   - Die endgültige Struktur sollte datensatz1 als Verzeichnis mit den Gestenklassen anzeigen

3. Vergewissern Sie sich, dass das trainierte Modell (`gesten1.h5`) und das Datensatzverzeichnis (`datensatz1`) vorhanden sind.

4. Starten Sie die Anwendung:
```bash
python gestus.py
```

5. Verwenden Sie die Webcam:
   - Führen Sie Gesten vor der Kamera aus
   - Die erkannte Geste wird oben im Fenster angezeigt
   - Drücken Sie 's', um die Anwendung zu beenden
![X-ezgif com-video-to-gif-converter (1)](https://github.com/user-attachments/assets/fd2d7b3f-f0c1-430c-8d1c-e44dfe0625a0)


## Technische Details

Das System verwendet eine mehrschichtige Architektur:

- **Eingabeverarbeitung**: OpenCV für Videokamera-Aufnahmen und Frame-Verarbeitung
- **Pose-Schätzung**: MediaPipe Holistic zur Extraktion von Körper-Keypoints
- **Gestenerkennung**: LSTM-basiertes neuronales Netzwerk mit der folgenden Architektur:
  - LSTM-Schichten (64 → 128 → 64 Einheiten)
  - Dense-Schichten (64 → 32 → Ausgabe)
  - Softmax-Aktivierung für die finale Klassifikation

## Modellarchitektur

```
Sequential([
    LSTM(64, return_sequences=True, activation='relu', input_shape=(20, 258))
    LSTM(128, return_sequences=True, activation='relu')
    LSTM(64, return_sequences=False, activation='relu')
    Dense(64, activation='relu')
    Dense(32, activation='relu')
    Dense(n_classes, activation='softmax')
])
```

## Inspiration und Credits

Dieses Projekt wurde von Nicholas Renottes Arbeit zur Erkennung von Gebärdensprache inspiriert. Die Implementierung folgt ähnlichen Prinzipien, wurde jedoch angepasst und modifiziert, um ein allgemeineres Gestenerkennungssystem zu schaffen. Die Original-Tutorial-Serie finden Sie hier:
- [Sign Language Detection Tutorial](https://www.youtube.com/watch?v=doDUihpj6ro)

---

# Gestus - Real-Time Gesture Recognition System

Gestus (Latin for "gesture") is a Python-based real-time gesture recognition system that uses computer vision and machine learning to detect and interpret hand gestures. The system utilizes MediaPipe for pose estimation and TensorFlow/Keras for gesture classification.

## Prerequisites

### Python Installation
- Python 3.8 or higher must be installed on your system
- You can download Python from the [official website](https://www.python.org/downloads/)
- Verify your Python installation by running:
```bash
python --version
```

### Required Libraries
Install the following dependencies using pip:
```bash
pip install mediapipe numpy opencv-python tensorflow scikit-learn
```

## Features

- Real-time Gesture Recognition: Detects and displays gestures live
- LSTM-based Neural Network: Ensures accurate prediction of gestures
- Mediapipe Integration: Efficiently extracts pose and hand landmarks
- Visual Feedback: Displays keypoints and recognized gestures on-screen

## Project Structure

```
gestus/
├── gestus.py          # Main application file
├── gesten1.h5         # Pre-trained model
├── datensatz1.zip     # Compressed dataset file
└── datensatz1/        # Dataset folder (after unzipping)
```

## Configuration

The system can be configured with the following parameters:

- `model_path`: Path to the trained model weights file (*.h5)
- `dataset_path`: Directory containing the gesture classes
- `threshold`: Confidence threshold for gesture detection (default: 0.6)

## Usage

1. Clone the repository:
```bash
git clone git clone https://github.com/zazabuzaladze/gestus.git
cd gestus
```
2. Unzip the dataset:
   - Extract the `datensatz1.zip` file in the project directory
   - Make sure the extracted folder is named `datensatz1`
   - The final structure should show `datensatz1` as a directory containing the gesture classes

3. Ensure you have the trained model (`gesten1.h5`) and dataset directory (`datensatz1`) in place.

4. Run the application:
```bash
python gestus.py
```

5. Use the webcam interface:
   - Perform gestures in front of the camera
   - The recognized gesture will be displayed at the top of the window
   - Press 's' to stop the application
![X-ezgif com-video-to-gif-converter (1)](https://github.com/user-attachments/assets/fd2d7b3f-f0c1-430c-8d1c-e44dfe0625a0)


## Technical Details

The system uses a multi-layer architecture:

- **Input Processing**: OpenCV for video capture and frame processing
- **Pose Estimation**: MediaPipe Holistic for extracting body keypoints
- **Gesture Recognition**: LSTM-based neural network with the following architecture:
  - LSTM layers (64 → 128 → 64 units)
  - Dense layers (64 → 32 → output)
  - Softmax activation for final classification

## Model Architecture

```
Sequential([
    LSTM(64, return_sequences=True, activation='relu', input_shape=(20, 258))
    LSTM(128, return_sequences=True, activation='relu')
    LSTM(64, return_sequences=False, activation='relu')
    Dense(64, activation='relu')
    Dense(32, activation='relu')
    Dense(n_classes, activation='softmax')
])
```

## Inspiration and Credits

This project is inspired by Nicholas Renotte's work on sign language detection. The implementation follows similar principles but has been adapted and modified to create a more generalized gesture recognition system. You can find the original tutorial series here:
- [Sign Language Detection Tutorial](https://www.youtube.com/watch?v=doDUihpj6ro)
