import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import time
import pygame
import threading

class EyeTrackerGrid:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.4,  # Lower for Pi 3
            min_tracking_confidence=0.4    # Lower for Pi 3
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
        self.long_blink_threshold = 1.0
        self.blink_start_time = None
        self.is_long_blinking = False
        
        self.gaze_history = deque(maxlen=5)
        self.last_blink_time = time.time()
        
        self.grid_x = 0
        self.grid_y = 2
        self.grid_positions = [
            ["up left", "up center", "up right"],
            ["center left", "center", "center right"],
            ["down left", "down center", "down right"]
        ]
        
        self.movement_cooldown = 0.6
        self.last_move_time = 0
        
        # Initialize audio - simplified for Pi
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            self.audio_enabled = True
        except:
            print("Audio not available, continuing without sound")
            self.audio_enabled = False
            
        self.audio_queue = []
        self.audio_thread_running = True
        if self.audio_enabled:
            self.audio_thread = threading.Thread(target=self._audio_worker)
            self.audio_thread.daemon = True
            self.audio_thread.start()
        
        print("Enhanced Eye Tracker with 3x3 Grid initialized!")
        print(f"Starting position: {self.grid_positions[self.grid_y][self.grid_x]}")
        if self.audio_enabled:
            self._play_position_sound()

    def _audio_worker(self):
        while self.audio_thread_running and self.audio_enabled:
            if self.audio_queue:
                sound_type = self.audio_queue.pop(0)
                self._play_sound(sound_type)
            time.sleep(0.1)

    def _play_sound(self, sound_type):
        if not self.audio_enabled:
            return
            
        try:
            duration = 0.2  # Shorter for Pi
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
            
            arr = (arr * 16383).astype(np.int16)  # Lower volume for Pi
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
        if self.audio_enabled:
            self.audio_queue.append("move")

    def _play_selection_sound(self):
        if self.audio_enabled:
            self.audio_queue.append("select")

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
            if self.blink_start_time is not None:
                blink_duration = time.time() - self.blink_start_time
                if blink_duration >= 0.1 and blink_duration < self.long_blink_threshold:
                    self.blink_counter += 1
                    self.last_blink_time = time.time()
                    
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
        
        if current_time - self.last_move_time < self.movement_cooldown:
            return False
            
        new_x = self.grid_x
        new_y = self.grid_y
        moved = False
        
        if gaze_direction == "LEFT" and self.grid_x > 0:
            new_x = self.grid_x - 1
            moved = True
        elif gaze_direction == "RIGHT" and self.grid_x < 2:
            new_x = self.grid_x + 1
            moved = True
        elif gaze_direction == "UP" and self.grid_y > 0:
            new_y = self.grid_y - 1
            moved = True
        
        if moved:
            self.grid_x = new_x
            self.grid_y = new_y
            self.last_move_time = current_time
            self._play_position_sound()
            return True
            
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

    def draw_eye_parts(self, img, landmarks):
        h, w = img.shape[:2]
        
        # Skip overlay for better performance on Pi
        is_blinking, ear_value = self.detect_blink(landmarks, h, w)
        gaze_direction = self.calculate_gaze_direction(landmarks, h, w)
        gaze_text = self.get_gaze_direction_text(gaze_direction)
        
        self.process_gaze_movement(gaze_text)
        
        # Draw simple eye tracking points - less intensive
        if all(idx < len(landmarks) for idx in self.LEFT_IRIS):
            left_iris_center, left_iris_radius = self.detect_iris(img, landmarks, self.LEFT_IRIS)
            cv2.circle(img, tuple(left_iris_center), left_iris_radius, (255, 100, 0), 2)
            
        if all(idx < len(landmarks) for idx in self.RIGHT_IRIS):
            right_iris_center, right_iris_radius = self.detect_iris(img, landmarks, self.RIGHT_IRIS)
            cv2.circle(img, tuple(right_iris_center), right_iris_radius, (255, 100, 0), 2)
        
        # Simplified UI for Pi performance
        y_offset = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        current_pos = self.grid_positions[self.grid_y][self.grid_x]
        cv2.putText(img, f"Position: {current_pos}", (10, y_offset),
                    font, 1.0, (0, 255, 255), 2)
        y_offset += 35
        
        cv2.putText(img, f"Grid: ({self.grid_x}, {self.grid_y})", (10, y_offset),
                    font, 0.7, (255, 255, 255), 2)
        y_offset += 30
        
        cv2.putText(img, f"Gaze: {gaze_text}", (10, y_offset),
                    font, 0.7, (0, 255, 255), 2)
        y_offset += 25
        
        if self.is_long_blinking:
            cv2.putText(img, "SELECTING!", (10, y_offset),
                        font, 0.7, (0, 0, 255), 2)
        elif is_blinking:
            cv2.putText(img, "BLINKING", (10, y_offset),
                        font, 0.7, (255, 0, 0), 2)
        else:
            cv2.putText(img, "Eyes Open", (10, y_offset),
                        font, 0.7, (0, 255, 0), 2)
        y_offset += 25
        
        time_since_move = time.time() - self.last_move_time
        if time_since_move < self.movement_cooldown:
            remaining = self.movement_cooldown - time_since_move
            cv2.putText(img, f"Cooldown: {remaining:.1f}s", (10, y_offset),
                        font, 0.6, (255, 255, 0), 2)
        y_offset += 25
        
        cv2.putText(img, f"Blinks: {self.blink_counter}", (10, y_offset),
                    font, 0.6, (255, 255, 255), 2)
        y_offset += 20
        
        cv2.putText(img, f"EAR: {ear_value:.3f}", (10, y_offset),
                    font, 0.6, (255, 255, 255), 2)
        y_offset += 25
        
        # Simple grid display
        cv2.putText(img, "Grid:", (10, y_offset), font, 0.6, (255, 255, 255), 2)
        y_offset += 20
        
        for row in range(3):
            grid_line = ""
            for col in range(3):
                if row == self.grid_y and col == self.grid_x:
                    grid_line += "[X] "
                else:
                    grid_line += "[ ] "
            cv2.putText(img, grid_line, (10, y_offset),
                        font, 0.5, (255, 255, 255), 1)
            y_offset += 18
        
        return img

    def run(self):
        # Try different camera indices for USB camera
        cap = None
        for i in range(3):  # Try video0, video1, video2
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f"Camera found at index {i}")
                break
            cap.release()
        
        if not cap or not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        # Pi 3 optimized settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)   # Smaller for Pi 3
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)  # Smaller for Pi 3
        cap.set(cv2.CAP_PROP_FPS, 15)            # Lower FPS for Pi 3
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)      # Reduce buffer lag
        
        print("Enhanced Eye Tracking Started on Raspberry Pi!")
        print("Optimized for Pi 3 performance")
        print("Controls:")
        print("- Look around to navigate 3x3 grid")
        print("- Long blink (1.0s) to select")
        print("- Top-right corner = cancel")
        print("- Press 'q' to quit, 'r' to reset")
        
        frame_count = 0
        process_every_n_frames = 2  # Process every 2nd frame for performance
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            frame = cv2.flip(frame, 1)
            frame_count += 1
            
            # Process every nth frame to improve performance
            if frame_count % process_every_n_frames == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb_frame)

                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]
                    frame = self.draw_eye_parts(frame, face_landmarks.landmark)
                else:
                    cv2.putText(frame, "No face detected", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, f"Position: {self.grid_positions[self.grid_y][self.grid_x]}", 
                               (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow('Eye Tracking - Pi3', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite('pi_eye_tracking_screenshot.png', frame)
                print("Screenshot saved!")
            elif key == ord('r'):
                self.grid_x = 0
                self.grid_y = 2
                self.blink_counter = 0
                self._play_position_sound()
                print("Reset to start position!")
                
        cap.release()
        cv2.destroyAllWindows()
        self.audio_thread_running = False
        if self.audio_enabled:
            pygame.mixer.quit()

def main():
    try:
        import cv2
        import mediapipe
        import numpy
        import pygame
        print("All required packages imported successfully!")
    except ImportError as e:
        print("Please install required packages:")
        print("pip3 install mediapipe numpy pygame")
        print(f"Missing: {e}")
        return
    
    tracker = EyeTrackerGrid()
    tracker.run()

if __name__ == "__main__":
    main()
