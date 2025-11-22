import cv2
import mediapipe as mp
import numpy as np
import random
import math
import time


# -----------------------------
# Hand tracking: index finger tip
# -----------------------------
class HandTracker:
    def __init__(self,
                 max_num_hands=1,
                 detection_confidence=0.6,
                 tracking_confidence=0.6):

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=1,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self.mp_draw = mp.solutions.drawing_utils

    def get_index_tip(self, frame, draw=True):
        """
        Returns (x, y) of index finger tip in image coordinates,
        or None if no hand detected.
        """
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        h, w, _ = frame.shape
        index_tip = None

        if results.multi_hand_landmarks:
            # Use first detected hand
            hand_landmarks = results.multi_hand_landmarks[0]

            if draw:
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )

            # MediaPipe index finger tip id = 8
            lm = hand_landmarks.landmark[8]
            x, y = int(lm.x * w), int(lm.y * h)
            index_tip = (x, y)

        return index_tip


# -----------------------------
# Snake game logic
# -----------------------------
class SnakeGame:
    def __init__(self, frame_width, frame_height):
        self.frame_w = frame_width
        self.frame_h = frame_height
        self.food_radius = 12
        self.self_collision_distance = 10
        self.reset()

    def reset(self):
        # Snake represented by a sequence of points
        self.points = []          # [(x1, y1), (x2, y2), ...]
        self.lengths = []         # segment distances between points
        self.current_length = 0.0
        self.target_length = 150  # snake length (grows with food)
        self.prev_head = None

        self.food_pos = self._random_food_position()
        self.score = 0
        self.game_over = False

    def _random_food_position(self):
        margin = 50
        x = random.randint(margin, self.frame_w - margin)
        y = random.randint(margin, self.frame_h - margin)
        return (x, y)

    @staticmethod
    def _distance(p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def update(self, head_pos):
        """
        Update the snake given the new head position (from finger).
        head_pos is (x, y) in pixel coordinates.
        """
        if self.game_over:
            return

        if head_pos is None:
            # No finger detected: keep last state
            return

        if self.prev_head is None:
            self.prev_head = head_pos

        # Add new head
        self.points.append(head_pos)
        segment_len = self._distance(head_pos, self.prev_head)
        self.lengths.append(segment_len)
        self.current_length += segment_len
        self.prev_head = head_pos

        # Trim tail to keep snake at target_length
        while self.current_length > self.target_length and len(self.lengths) > 0:
            self.current_length -= self.lengths.pop(0)
            self.points.pop(0)

        # Check food collision
        if self._distance(head_pos, self.food_pos) < self.food_radius + 5:
            self.score += 1
            self.target_length += 30      # grow the snake
            self.food_pos = self._random_food_position()

        # Check self collision (standard Snake rule)
        if len(self.points) > 20:
            head = np.array(head_pos)
            body = np.array(self.points[:-10])  # ignore last few near head
            if len(body) > 0:
                dists = np.linalg.norm(body - head, axis=1)
                if np.min(dists) < self.self_collision_distance:
                    self.game_over = True

        # Check boundary collision
        x, y = head_pos
        if x <= 0 or x >= self.frame_w or y <= 0 or y >= self.frame_h:
            self.game_over = True

    def draw(self, frame):
        """
        Draws snake, food, score, and game-over text on the frame.
        """
        # Draw snake body
        if len(self.points) > 1:
            for i in range(1, len(self.points)):
                cv2.line(
                    frame,
                    self.points[i - 1],
                    self.points[i],
                    (0, 255, 0),
                    10,
                )

        # Draw snake head
        if self.points:
            cv2.circle(frame, self.points[-1], 10, (0, 0, 255), cv2.FILLED)

        # Draw food
        cv2.circle(frame, self.food_pos, self.food_radius, (255, 0, 0), cv2.FILLED)

        # Draw score
        cv2.putText(
            frame,
            f"Score: {self.score}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        # Game over overlay
        if self.game_over:
            cv2.putText(
                frame,
                "GAME OVER",
                (self.frame_w // 2 - 170, self.frame_h // 2 - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 0, 255),
                5,
            )
            cv2.putText(
                frame,
                "Press R to Restart, Q to Quit",
                (self.frame_w // 2 - 260, self.frame_h // 2 + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

        return frame


# -----------------------------
# Main loop
# -----------------------------
def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Set a reasonable size for smooth performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Read one frame to get real dimensions
    success, frame = cap.read()
    if not success:
        print("Error: Could not read from webcam.")
        cap.release()
        return

    frame_h, frame_w, _ = frame.shape

    hand_tracker = HandTracker()
    game = SnakeGame(frame_w, frame_h)

    prev_time = time.time()

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Flip for mirror-like experience
        frame = cv2.flip(frame, 1)

        # 1. Gesture recognition: get index finger tip
        index_tip = hand_tracker.get_index_tip(frame, draw=True)

        # 2 & 3. Update snake game with the finger-controlled head
        game.update(index_tip)

        # 4. Draw everything on top of the webcam feed
        frame = game.draw(frame)

        # FPS display for performance check
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if curr_time != prev_time else 0
        prev_time = curr_time

        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (10, frame_h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )

        cv2.imshow("Gesture-Controlled Snake (Index Finger)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            game.reset()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
