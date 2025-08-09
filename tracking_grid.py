import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import time
import pygame
import threading

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
        
        self.grid_x = 0
        self.grid_y = 2
        self.grid_positions = [
            ["up left", "up center", "up right"],
            ["center left", "center", "center right"],
            ["down left", "down center", "down right"]
        ]
        

        self.movement_cooldown = 0.8
        self.last_move_time = 0
        self.gaze_hold_time = 0.8
        self.gaze_direction_start = None
        self.current_gaze_direction = None
        self.gaze_confidence_threshold = 0.25
        self.movement_history = deque(maxlen=15)
        

        self.long_blink_threshold = 1.0
        self.blink_start_time = None
        self.is_long_blinking = False
        

        pygame.mixer.init()
        self.audio_queue = []
        self.audio_thread_running = True
        self.audio_thread = threading.Thread(target=self._audio_worker)
        self.audio_thread.daemon = True
        self.audio_thread.start()
        
        print("Enhanced Eye Tracker with 3x3 Grid initialized!")
        print(f"Starting position: {self.grid_positions[self.grid_y][self.grid_x]}")
        self._play_position_sound()

    def _audio_worker(self):
        while self.audio_thread_running:
            if self.audio_queue:
                sound_type = self.audio_queue.pop(0)
                self._play_sound(sound_type)
            time.sleep(0.1)

    def _play_sound(self, sound_type):
        try:
            duration = 0.3
            sample_rate = 22050
            frames = int(duration * sample_rate)
            
            if sound_type == "move":
                frequency = 800
            elif sound_type == "select":
                frequency = 400
            else:
                frequency = 600
                
            arr = np.zeros(frames)
            for i in range(frames):
                arr[i] = np.sin(2 * np.pi * frequency * i / sample_rate)
            
            arr = (arr * 32767).astype(np.int16)
            stereo_arr = np.zeros((frames, 2), dtype=np.int16)
            stereo_arr[:, 0] = arr
            stereo_arr[:, 1] = arr
            
            sound = pygame.sndarray.make_sound(stereo_arr)
            sound.play()
            
            if sound_type == "move":
                print(f"Audio: Moved to {self.grid_positions[self.grid_y][self.grid_x]}")
            elif sound_type == "select":
                print(f"Audio: Selected {self.grid_positions[self.grid_y][self.grid_x]}")
                
        except Exception as e:
            print(f"Audio error: {e}")

    def _play_position_sound(self):
        self.audio_queue.append("move")

    def _play_selection_sound(self):
        self.audio_queue.append("select")

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
        
        if is_blinking:
            if self.blink_start_time is None:
                self.blink_start_time = time.time()
            else:
                blink_duration = time.time() - self.blink_start_time
                if blink_duration >= self.long_blink_threshold and not self.is_long_blinking:
                    self.is_long_blinking = True
                    self._confirm_selection()
        else:
            self.blink_start_time = None
            self.is_long_blinking = False
        
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

    def process_gaze_movement(self, gaze_direction):
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_move_time < self.movement_cooldown:
            return False
        
        # Add to movement history
        self.movement_history.append(gaze_direction)
        
        if len(self.movement_history) >= 10:
            recent_directions = list(self.movement_history)[-10:]
            direction_counts = {}
            for d in recent_directions:
                if d != "CENTER":
                    direction_counts[d] = direction_counts.get(d, 0) + 1
            
            if direction_counts:
                dominant_direction = max(direction_counts, key=direction_counts.get)
                consistency_ratio = direction_counts[dominant_direction] / 10.0
                
                if consistency_ratio >= 0.7:
                    if self.current_gaze_direction != dominant_direction:
                        self.current_gaze_direction = dominant_direction
                        self.gaze_direction_start = current_time
                        return False
                    elif current_time - self.gaze_direction_start >= self.gaze_hold_time:
                        new_x = self.grid_x
                        new_y = self.grid_y
                        moved = False
                        
                        if dominant_direction == "LEFT" and self.grid_x > 0:
                            new_x = self.grid_x - 1
                            moved = True
                        elif dominant_direction == "RIGHT" and self.grid_x < 2:
                            new_x = self.grid_x + 1
                            moved = True
                        elif dominant_direction == "UP" and self.grid_y > 0:
                            new_y = self.grid_y - 1
                            moved = True
                        elif dominant_direction == "DOWN" and self.grid_y < 2:
                            new_y = self.grid_y + 1
                            moved = True
                        
                        if moved:
                            self.grid_x = new_x
                            self.grid_y = new_y
                            self.last_move_time = current_time
                            self.current_gaze_direction = None
                            self.gaze_direction_start = None
                            self.movement_history.clear()
                            self._play_position_sound()
                            return True
                else:
                    self.current_gaze_direction = None
                    self.gaze_direction_start = None
            else:
                self.current_gaze_direction = None
                self.gaze_direction_start = None
                
        return False

    def _confirm_selection(self):
        self._play_selection_sound()
        print(f"SELECTION CONFIRMED: {self.grid_positions[self.grid_y][self.grid_x]} at grid position ({self.grid_x}, {self.grid_y})")
        
        if self.grid_x == 2 and self.grid_y == 0:
            print("CANCEL selected! Returning to start position...")
            self.grid_x = 0
            self.grid_y = 2
            self._play_position_sound()

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
        self.process_gaze_movement(gaze_text)
        
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
        
        current_pos = self.grid_positions[self.grid_y][self.grid_x]
        cv2.putText(img, f"Position: {current_pos}", (10, y_offset),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 3)
        y_offset += 35
        
        cv2.putText(img, f"Grid: ({self.grid_x}, {self.grid_y})", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        y_offset += 25
        
        gaze_color = (0, 255, 255)
        cv2.putText(img, f"Gaze: {gaze_text}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, gaze_color, 2)
        y_offset += 25
        
        if self.current_gaze_direction and self.gaze_direction_start:
            hold_progress = min(1.0, (time.time() - self.gaze_direction_start) / self.gaze_hold_time)
            progress_text = f"Hold {self.current_gaze_direction}: {hold_progress*100:.0f}%"
            cv2.putText(img, progress_text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_offset += 25
        
        if self.is_long_blinking:
            cv2.putText(img, "LONG BLINK - SELECTING!", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3)
        elif is_blinking:
            cv2.putText(img, "BLINKING", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(img, "Eyes Open", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_offset += 25
        
        time_since_move = time.time() - self.last_move_time
        if time_since_move < self.movement_cooldown:
            remaining = self.movement_cooldown - time_since_move
            cv2.putText(img, f"Cooldown: {remaining:.1f}s", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            y_offset += 25
        
        cv2.putText(img, f"Blinks: {self.blink_counter}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
        
        cv2.putText(img, f"EAR: {ear_value:.3f}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 30
        
        cv2.putText(img, "3x3 Grid:", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 20
        
        for row in range(3):
            grid_line = ""
            for col in range(3):
                if row == self.grid_y and col == self.grid_x:
                    grid_line += "[X] "
                else:
                    grid_line += "[ ] "
            cv2.putText(img, grid_line, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            y_offset += 20
        
        y_offset += 10
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
        print("Features: Gaze Direction + Blink Detection + 3x3 Grid Navigation")
        print("Controls:")
        print("- Look around to navigate the 3x3 grid")
        print("- Long blink (1.0s) to select current position")
        print("- Press 'q' to quit")
        print("- Press 's' to save a screenshot")
        print("- Press 'r' to reset blink counter and grid position")
        
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
                cv2.putText(frame, f"Position: {self.grid_positions[self.grid_y][self.grid_x]}", 
                           (10, 60), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 255), 2)

            cv2.imshow('Enhanced Eye Tracking', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite('enhanced_eye_tracking_screenshot.png', frame)
                print("Screenshot saved!")
            elif key == ord('r'):
                self.blink_counter = 0
                self.grid_x = 0
                self.grid_y = 2
                self._play_position_sound()
                print("Blink counter and grid position reset!")
                
        cap.release()
        cv2.destroyAllWindows()
        self.audio_thread_running = False
        pygame.mixer.quit()

def main():
    try:
        import cv2
        import mediapipe
        import numpy
        import pygame
    except ImportError as e:
        print("Please install required packages:")
        print("pip install opencv-python mediapipe numpy pygame")
        print(f"Missing: {e}")
        return
    
    tracker = EyeTracker()
    tracker.run()

if __name__ == "__main__":
    main()
