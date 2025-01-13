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
└── datensatz1/        # Dataset folder
```

## Configuration

The system can be configured with the following parameters:

- `model_path`: Path to the trained model weights file (*.h5)
- `dataset_path`: Directory containing the gesture classes
- `threshold`: Confidence threshold for gesture detection (default: 0.6)

## Usage

1. Clone the repository:
```bash
git clone <repository-url>
cd gestus
```

2. Ensure you have the trained model (`gesten1.h5`) and dataset directory (`datensatz1`) in place.

3. Run the application:
```bash
python gestus.py
```

4. Use the webcam interface:
   - Perform gestures in front of the camera
   - The recognized gesture will be displayed at the top of the window
   - Press 's' to stop the application
![X-ezgif com-video-to-gif-converter (1)](https://github.com/user-attachments/assets/2887a202-ed41-428e-9420-18c2b9ed3b2f)

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
