import mediapipe as mp    #model erstellen
import numpy as np        #mit arrays arbeiten
import os                 #mit dem Dateisystem interagieren
from sklearn.model_selection import train_test_split        #datensatz teilen für testen und trainieren
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential
"""Die sequentielle ist nützlicher als die funktionale, 
da wir eine Dateneingabe und eine Datenausgabe haben.
Es leitet die Daten weiter und fließt in sequentieller Reihenfolge von oben nach unten, 
bis die Daten am Ende des Modells ankommen.
https://www.educba.com/keras-sequential/
https://www.youtube.com/watch?v=8uC-WT1LYnU&ab_channel=Simplilearn
"""

from tensorflow.keras.layers import LSTM, Dense
"""LSTM Layer: ermöglicht uns die Erkennung von Aktionen.
Keras Dense-Schicht ist die Schicht, 
die alle Neuronen enthält, die in sich selbst tief verbunden sind. 
Das bedeutet, dass jedes Neuron in der dichten Schicht 
die Eingabe von allen anderen Neuronen der vorherigen Schicht erhält. 
Wir können so viele dichte Schichten wie nötig hinzufügen. 
Sie ist eine der am häufigsten verwendeten Schichten.
https://www.educba.com/keras-dense/
https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
"""

import cv2 as cv    #zugriff auf kamera
import tensorflow as tf
from typing import Tuple, List, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GestureRecognizer:
    def __init__(self, model_path: str, dataset_path: str, threshold: float = 0.6):
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.threshold = threshold
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.gestures = self._load_gestures()
        self.model = self._build_and_load_model()
        self.sentence: List[str] = []
        self.frames_buffer: List[np.ndarray] = []
        
    def _load_gestures(self) -> np.ndarray:
        #Liste aller Gesten abrufen
        gesture_list = os.listdir(self.dataset_path)
        return np.array(gesture_list)

    #model bauen und laden
    def _build_and_load_model(self) -> Sequential:
        model = Sequential([
            LSTM(64, return_sequences=True, activation='relu', input_shape=(20, 258)),
            #64 LSTM Units     
            #return sequences = True: weil die nächste Schicht dies braucht
            #activation = relu: Funktion, die die Eingabe direkt ausgibt, wenn sie positiv ist, andernfalls gibt sie Null aus.

            LSTM(128, return_sequences=True, activation='relu'),
            LSTM(64, return_sequences=False, activation='relu'),
            #return sequences = False: weil die nächste Schicht ist Dense layer/schicht


            Dense(64, activation='relu'),
            Dense(32, activation='relu'),

            #letzte Schicht
            Dense(len(self.gestures), activation='softmax')
            #Softmax ist eine Aktivierungsfunktion, die Zahlen/Logits in Wahrscheinlichkeiten umwandelt. 
            #Die Ausgabe einer Softmax-Funktion ist ein Vektor (z. B. v) mit Wahrscheinlichkeiten für jedes mögliche Ergebnis. 
            #Die Wahrscheinlichkeiten im Vektor v summieren sich für alle möglichen Ergebnisse oder Klassen zu eins.
            #https://towardsdatascience.com/softmax-activation-function-how-it-actually-works-d292d335bd78
        ])
        
        try:
            model.load_weights(self.model_path)
            logging.info("Model weights loaded successfully")
        except Exception as e:
            logging.error(f"Error loading model weights: {e}")
            raise
            
        return model

    #farben umwandlung + model hinzufügen
    def conversion(self, frame: np.ndarray, model: mp.solutions.holistic.Holistic) -> Tuple[np.ndarray, mp.solutions.holistic.Holistic]:
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        #hier umwandeln wir frame in rgb weil mediapipe braucht rgb format
        
        frame_rgb.flags.writeable = False
        results = model.process(frame_rgb)
        #hier hinzufügen wir mediapipe model/"eine Vorhersage treffen"

        frame_rgb.flags.writeable = True
        frame_bgr = cv.cvtColor(frame_rgb, cv.COLOR_RGB2BGR)
        #hier zurückkehren wir in BGR damit cv2 es zeigen kann

        return frame_bgr, results

    #orientierungspunkte zeigen lassen an cv
    def show_keypoints(self, frame: np.ndarray, results: mp.solutions.holistic.Holistic) -> None:

        #wir werden gesichtspunkte nicht gebrauchen bei gebärdensprache
        #mp_drawing.draw_landmarks(main_frame, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)
    
        #                         image       orientierungspunkte     verbindungskarte
        self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS)
        self.mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
        self.mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)

    #k-points extrahieren
    def extract_keypoints(self, results: mp.solutions.holistic.Holistic) -> np.ndarray:
        def extract_or_zeros(landmarks, points: int, dimensions: int) -> np.ndarray:
            #wir verwenden die "try, except"-Methode, falls nicht alle Schlüsselpunkte in der Kamera angezeigt werden, so dass es nicht zu einem Fehler kommt
            try:
                if dimensions == 4:
                    return np.array([[res.x, res.y, res.z, res.visibility] for res in landmarks.landmark]).flatten()
                return np.array([[res.x, res.y, res.z] for res in landmarks.landmark]).flatten()
            except:
                return np.zeros(points * dimensions)

        pose = extract_or_zeros(results.pose_landmarks, 33, 4)
        left_hand = extract_or_zeros(results.left_hand_landmarks, 21, 3)
        right_hand = extract_or_zeros(results.right_hand_landmarks, 21, 3)
        
        return np.concatenate([pose, left_hand, right_hand])

    def process_frame(self, frame: np.ndarray, results: mp.solutions.holistic.Holistic) -> Optional[str]:
        keypoints = self.extract_keypoints(results)
        self.frames_buffer.append(keypoints)
        self.frames_buffer = self.frames_buffer[-30:]  #wir führen die Vorhersage nur aus, wenn es 30 Frames sind
        
        #wir führen die Vorhersage nur aus, wenn es 30 Frames sind
        if len(self.frames_buffer) == 30:
            prediction = self.model.predict(np.expand_dims(self.frames_buffer, axis=0), verbose=0)[0]
            predicted_idx = np.argmax(prediction)
            confidence = prediction[predicted_idx]
            
            if confidence > self.threshold:
                predicted_gesture = self.gestures[predicted_idx]
                if not self.sentence or predicted_gesture != self.sentence[-1]:
                    self.sentence = [predicted_gesture]     #nur eine(letzte) wort zeigen
                    return predicted_gesture
        return None

    #programm ausführen
    def run(self) -> None:
        capture = cv.VideoCapture(0)
        # 1 = mein computer webcam
        # 0 = mein externe webcam/handy
        
        with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while capture.isOpened():
                ret, frame = capture.read()
                if not ret:
                    logging.error("Failed to grab frame")
                    break

                flipped_frame = cv.flip(frame, 1)   #bild spiegeln
                main_frame, results = self.conversion(flipped_frame, holistic)
                self.show_keypoints(main_frame, results)    #orientierungspunkte in kamera zeigen lassen
                
                predicted_gesture = self.process_frame(main_frame, results)
                if predicted_gesture:
                    logging.info(f"Detected gesture: {predicted_gesture}")

                cv.rectangle(main_frame, (0,0), (650, 50), (255, 255, 255), -1)
                cv.putText(main_frame, ' '.join(self.sentence), (240,45), 
                          cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3, cv.LINE_AA)
                cv.putText(main_frame, "to stop, press 's'", (450,460), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
                
                cv.imshow("Kamera - Gestus", main_frame)
                
                if cv.waitKey(20) & 0xFF == ord("s"):
                    break

        capture.release()
        cv.destroyAllWindows()

if __name__ == "__main__":
    gestus = GestureRecognizer(
        model_path="gesten1.h5",
        dataset_path="datensatz1",
        threshold=0.6
    )
    gestus.run()