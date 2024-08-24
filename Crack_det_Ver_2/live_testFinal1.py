import cv2
from tensorflow.keras.models import load_model
import numpy as np
import os


class CrackDetector:
    def __init__(self, model_path, camera_index=0):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.model = load_model(model_path)
        self.cap = cv2.VideoCapture(camera_index)

        if not self.cap.isOpened():
            raise RuntimeError("Error opening video stream or file")

        self.pixel_to_cm_ratio = 0.1  # Example conversion ratio
        self.frame_resizing_dim = (224, 224)

        # Perform camera calibration
        self.camera_matrix, self.dist_coeffs = self.calibrate_camera()

    def calibrate_camera(self, checkerboard_size=(7, 7), square_size=1.0):
        # Define the criteria for termination of the algorithm
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Prepare object points based on the checkerboard size
        objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
        objp *= square_size

        # Arrays to store object points and image points from all images
        objpoints = []  # 3d points in real-world space
        imgpoints = []  # 2d points in image plane

        # Capture multiple frames to find checkerboard corners
        for i in range(20):  # Try capturing more frames
            ret, frame = self.cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

            print(f"Frame {i}: Checkerboard {'found' if ret else 'not found'}")
            if ret:
                objpoints.append(objp)
                cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners)

                cv2.drawChessboardCorners(frame, checkerboard_size, corners, ret)
                cv2.imshow('Checkerboard', frame)
                cv2.waitKey(500)

        cv2.destroyAllWindows()

        if len(objpoints) > 0 and len(imgpoints) > 0:
            ret, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None,
                                                                        None)
            return camera_matrix, dist_coeffs
        else:
            raise RuntimeError(
                "Camera calibration failed. Make sure the checkerboard pattern is visible in multiple frames.")

    def preprocess_frame(self, frame):
        # Adjust brightness and contrast
        frame = self.adjust_brightness_contrast_auto(frame)

        # Resize the frame for the model
        resized_frame = cv2.resize(frame, self.frame_resizing_dim)
        img_array = np.expand_dims(resized_frame, axis=0)
        img_array = img_array / 255.0
        return img_array

    def adjust_brightness_contrast_auto(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)

        # Adjust based on mean brightness
        if mean_brightness < 100:
            frame = self.adjust_brightness_contrast(frame, brightness=30, contrast=40)
        elif mean_brightness > 150:
            frame = self.adjust_brightness_contrast(frame, brightness=-30, contrast=40)

        return frame

    def adjust_brightness_contrast(self, frame, brightness=0, contrast=0):
        if brightness != 0:
            if brightness > 0:
                shadow = brightness
                highlight = 255
            else:
                shadow = 0
                highlight = 255 + brightness
            alpha_b = (highlight - shadow) / 255
            gamma_b = shadow

            frame = cv2.addWeighted(frame, alpha_b, frame, 0, gamma_b)

        if contrast != 0:
            f = 131 * (contrast + 127) / (127 * (131 - contrast))
            alpha_c = f
            gamma_c = 127 * (1 - f)

            frame = cv2.addWeighted(frame, alpha_c, frame, 0, gamma_c)

        return frame

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
            cv2.drawContours(frame, [contour], -1, (0, 0, 255), 2)

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


if __name__ == "__main__":
    model_path = 'models/crack_detector.h5'  # Ensure this path is correct
    detector = CrackDetector(model_path)
    detector.run()
