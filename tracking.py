import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import time

class EyeTracker:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.LEFT_IRIS = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]
        self.LEFT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154,
                                 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373,
                                  390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        self.LEFT_EYE_CORNERS = [33, 133]
        self.RIGHT_EYE_CORNERS = [362, 263]
        
        self.LEFT_EYE_VERTICAL = [159, 145]
        self.RIGHT_EYE_VERTICAL = [386, 374]
        
        self.blink_threshold = 0.25
        self.blink_frames = 0
        self.blink_counter = 0
        self.eye_aspect_ratios = deque(maxlen=10)
        
        self.gaze_history = deque(maxlen=5)
        self.last_blink_time = time.time()

    def calculate_eye_aspect_ratio(self, eye_landmarks, landmarks):
        h, w = 480, 640
        
        points = []
        for idx in eye_landmarks:
            x = landmarks[idx].x * w
            y = landmarks[idx].y * h
            points.append([x, y])
        
        if len(points) >= 6:
            vertical_1 = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
            vertical_2 = np.linalg.norm(np.array(points[2]) - np.array(points[4]))
            
            horizontal = np.linalg.norm(np.array(points[0]) - np.array(points[3]))
            
            ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
            return ear
        return 0.3

    def calculate_simple_ear(self, top_point, bottom_point, left_point, right_point, landmarks, h, w):
        top = np.array([landmarks[top_point].x * w, landmarks[top_point].y * h])
        bottom = np.array([landmarks[bottom_point].x * w, landmarks[bottom_point].y * h])
        left = np.array([landmarks[left_point].x * w, landmarks[left_point].y * h])
        right = np.array([landmarks[right_point].x * w, landmarks[right_point].y * h])
        
        vertical_dist = np.linalg.norm(top - bottom)
        horizontal_dist = np.linalg.norm(left - right)
        
        if horizontal_dist > 0:
            ear = vertical_dist / horizontal_dist
        else:
            ear = 0.3
            
        return ear

    def detect_blink(self, landmarks, h, w):
        left_ear = self.calculate_simple_ear(159, 145, 33, 133, landmarks, h, w)
        right_ear = self.calculate_simple_ear(386, 374, 362, 263, landmarks, h, w)
        
        avg_ear = (left_ear + right_ear) / 2.0
        self.eye_aspect_ratios.append(avg_ear)
        
        if avg_ear < self.blink_threshold:
            self.blink_frames += 1
        else:
            if self.blink_frames >= 2:
                self.blink_counter += 1
                self.last_blink_time = time.time()
            self.blink_frames = 0
        
        is_blinking = avg_ear < self.blink_threshold
        
        return is_blinking, avg_ear

    def calculate_gaze_direction(self, landmarks, h, w):
        directions = []
        
        for iris_indices, eye_corners in [(self.LEFT_IRIS, self.LEFT_EYE_CORNERS),
                                         (self.RIGHT_IRIS, self.RIGHT_EYE_CORNERS)]:
            
            iris_center, _ = self.detect_iris_center(landmarks, iris_indices, h, w)
            
            inner_corner = np.array([landmarks[eye_corners[0]].x * w, 
                                   landmarks[eye_corners[0]].y * h])
            outer_corner = np.array([landmarks[eye_corners[1]].x * w, 
                                   landmarks[eye_corners[1]].y * h])
            
            eye_center = (inner_corner + outer_corner) / 2
            
            relative_pos = iris_center - eye_center
            
            eye_width = np.linalg.norm(outer_corner - inner_corner)
            if eye_width > 0:
                relative_pos = relative_pos / eye_width
            
            directions.append(relative_pos)
        
        if len(directions) == 2:
            avg_direction = (directions[0] + directions[1]) / 2
        else:
            avg_direction = directions[0] if directions else np.array([0, 0])
        
        self.gaze_history.append(avg_direction)
        
        if len(self.gaze_history) > 0:
            smoothed_direction = np.mean(self.gaze_history, axis=0)
        else:
            smoothed_direction = np.array([0, 0])
        
        return smoothed_direction

    def get_gaze_direction_text(self, direction):
        x, y = direction
        
        horizontal_threshold = 0.15
        vertical_threshold = 0.1
        
        directions = []
        
        if x > horizontal_threshold:
            directions.append("RIGHT")
        elif x < -horizontal_threshold:
            directions.append("LEFT")
        else:
            directions.append("CENTER")
        
        if y > vertical_threshold:
            directions.append("DOWN")
        elif y < -vertical_threshold:
            directions.append("UP")
        else:
            directions.append("CENTER")
        
        if directions[0] == "CENTER" and directions[1] == "CENTER":
            return "CENTER"
        elif directions[0] == "CENTER":
            return directions[1]
        elif directions[1] == "CENTER":
            return directions[0]
        else:
            return f"{directions[1]} {directions[0]}"

    def detect_iris_center(self, landmarks, iris_indices, h, w):
        iris_points = []
        for idx in iris_indices:
            x = int(landmarks[idx].x * w)
            y = int(landmarks[idx].y * h)
            iris_points.append([x, y])
        
        iris_points = np.array(iris_points)
        center = np.mean(iris_points, axis=0).astype(int)
        radius = int(np.mean(np.linalg.norm(iris_points - center, axis=1)))
        
        return center, radius

    def detect_iris(self, img, landmarks, iris_indices):
        h, w = img.shape[:2]
        return self.detect_iris_center(landmarks, iris_indices, h, w)

    def detect_pupil(self, img, iris_center, iris_radius):
        x, y = iris_center
        r = iris_radius
        roi_size = int(r * 2.5)
        x1 = max(0, x - roi_size)
        y1 = max(0, y - roi_size)
        x2 = min(img.shape[1], x + roi_size)
        y2 = min(img.shape[0], y + roi_size)
        roi = img[y1:y2, x1:x2]
        
        if roi.size == 0:
            return iris_center, int(iris_radius * 0.3)
            
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_roi = cv2.bilateralFilter(gray_roi, 9, 75, 75)
        circles = cv2.HoughCircles(
            gray_roi,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=30,
            param1=50,
            param2=30,
            minRadius=int(r * 0.2),
            maxRadius=int(r * 0.6)
        )
        if circles is not None:
            circles = np.uint16(np.around(circles))
            pupil_x = circles[0, 0, 0] + x1
            pupil_y = circles[0, 0, 1] + y1
            pupil_r = circles[0, 0, 2]
            return (pupil_x, pupil_y), pupil_r
        return iris_center, int(iris_radius * 0.3)

    def draw_eye_parts(self, img, landmarks):
        h, w = img.shape[:2]
        overlay = img.copy()
        
        is_blinking, ear_value = self.detect_blink(landmarks, h, w)
        gaze_direction = self.calculate_gaze_direction(landmarks, h, w)
        gaze_text = self.get_gaze_direction_text(gaze_direction)
        
        if all(idx < len(landmarks) for idx in self.LEFT_IRIS):
            left_iris_center, left_iris_radius = \
                self.detect_iris(img, landmarks, self.LEFT_IRIS)
            cv2.circle(overlay, tuple(left_iris_center),
                       left_iris_radius, (255, 100, 0), -1)
            left_pupil_center, left_pupil_radius = \
                self.detect_pupil(img, left_iris_center, left_iris_radius)
            cv2.circle(overlay, tuple(left_pupil_center),
                       left_pupil_radius, (0, 0, 0), -1)
            eye_points = []
            for idx in self.LEFT_EYE_CONTOUR:
                x = int(landmarks[idx].x * w)
                y = int(landmarks[idx].y * h)
                eye_points.append([x, y])
            eye_points = np.array(eye_points, dtype=np.int32)
            cv2.polylines(overlay, [eye_points], True, (0, 255, 0), 2)
            
        if all(idx < len(landmarks) for idx in self.RIGHT_IRIS):
            right_iris_center, right_iris_radius = \
                self.detect_iris(img, landmarks, self.RIGHT_IRIS)
            cv2.circle(overlay, tuple(right_iris_center),
                       right_iris_radius, (255, 100, 0), -1)
            right_pupil_center, right_pupil_radius = \
                self.detect_pupil(img, right_iris_center, right_iris_radius)
            cv2.circle(overlay, tuple(right_pupil_center),
                       right_pupil_radius, (0, 0, 0), -1)
            eye_points = []
            for idx in self.RIGHT_EYE_CONTOUR:
                x = int(landmarks[idx].x * w)
                y = int(landmarks[idx].y * h)
                eye_points.append([x, y])
            eye_points = np.array(eye_points, dtype=np.int32)
            cv2.polylines(overlay, [eye_points], True, (0, 255, 0), 2)
        
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        
        y_offset = 30
        
        gaze_color = (0, 255, 255)
        cv2.putText(img, f"Gaze: {gaze_text}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, gaze_color, 2)
        y_offset += 25
        
        if is_blinking:
            cv2.putText(img, "BLINKING", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(img, "Eyes Open", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_offset += 25
        
        cv2.putText(img, f"Blinks: {self.blink_counter}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
        
        cv2.putText(img, f"EAR: {ear_value:.3f}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 30
        
        cv2.putText(img, "Legend:", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 20
        cv2.putText(img, "Green - Eye Contour", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        y_offset += 20
        cv2.putText(img, "Blue - Iris", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 2)
        y_offset += 20
        cv2.putText(img, "Black - Pupil", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return img

    def run(self):
        cap = cv2.VideoCapture(0)
        print("Enhanced Eye Tracking Started!")
        print("Features: Gaze Direction + Blink Detection")
        print("Press 'q' to quit")
        print("Press 's' to save a screenshot")
        print("Press 'r' to reset blink counter")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                frame = self.draw_eye_parts(frame, face_landmarks.landmark)
            else:
                cv2.putText(frame, "No face detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow('Enhanced Eye Tracking', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite('enhanced_eye_tracking_screenshot.png', frame)
                print("Screenshot saved!")
            elif key == ord('r'):
                self.blink_counter = 0
                print("Blink counter reset!")
                
        cap.release()
        cv2.destroyAllWindows()

def main():
    try:
        import cv2
        import mediapipe
        import numpy
    except ImportError:
        print("Please install required packages:")
        print("pip install opencv-python mediapipe numpy")
        return
    
    tracker = EyeTracker()
    tracker.run()

if __name__ == "__main__":
    main()
