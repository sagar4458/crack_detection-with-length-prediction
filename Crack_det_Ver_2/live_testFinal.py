import cv2
from tensorflow.keras.models import load_model
import numpy as np
import os


class CrackDetector:
    def __init__(self, model_path, camera_index=0):  # Corrected init method
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.model = load_model(model_path)
        self.cap = cv2.VideoCapture(camera_index)

        if not self.cap.isOpened():
            raise RuntimeError("Error opening video stream or file")

        self.pixel_to_cm_ratio = 0.1  # Example conversion ratio
        self.frame_resizing_dim = (224, 224)

    def preprocess_frame(self, frame):
        resized_frame = cv2.resize(frame, self.frame_resizing_dim)
        img_array = np.expand_dims(resized_frame, axis=0)
        img_array = img_array / 255.0
        return img_array

    def detect_crack(self, frame):
        img_array = self.preprocess_frame(frame)
        prediction = self.model.predict(img_array)
        return prediction[0][0] > 0.5

    def process_frame(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_frame, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        crack_length = 0
        for contour in contours:
            crack_length += cv2.arcLength(contour, True)
            cv2.drawContours(frame, [contour], -1, (0, 0, 255), 2)  # Draw contours for visualization

        crack_length_cm = crack_length * self.pixel_to_cm_ratio
        return frame, crack_length_cm

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            if self.detect_crack(frame):
                frame, crack_length_cm = self.process_frame(frame)
                cv2.putText(frame, f"Crack Length: {crack_length_cm:.2f} cm", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 2)
                cv2.putText(frame, "Crack Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No Crack Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Display the frame using OpenCV's imshow function
            cv2.imshow("Crack Detection", frame)

            # Exit mechanism: press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":  # Corrected condition
    model_path = 'models/crack_detector.h5'  # Ensure this path is correct
    detector = CrackDetector(model_path)
    detector.run()
