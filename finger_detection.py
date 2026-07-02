"""
Finger Detection & Counting using Webcam
------------------------------------------
Detects a hand via webcam and counts the number of raised fingers in real time.

Tech stack:
- OpenCV        -> webcam capture + drawing
- MediaPipe     -> hand landmark detection (21 landmarks per hand)

Run:
    python finger_detection.py

Press 'q' to quit.
"""

import cv2
import mediapipe as mp
import time


class FingerDetector:
    def __init__(self, max_hands=1, detection_conf=0.7, tracking_conf=0.7):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Landmark indices for fingertips: thumb, index, middle, ring, pinky
        self.tip_ids = [4, 8, 12, 16, 20]

    def find_hands(self, frame, draw=True):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb)

        if self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(
                        frame, hand_lms, self.mp_hands.HAND_CONNECTIONS
                    )
        return frame

    def find_positions(self, frame, hand_no=0):
        """Return list of [id, x, y] for each of the 21 landmarks of one hand."""
        lm_list = []
        if self.results.multi_hand_landmarks:
            if hand_no < len(self.results.multi_hand_landmarks):
                hand = self.results.multi_hand_landmarks[hand_no]
                h, w, _ = frame.shape
                for idx, lm in enumerate(hand.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lm_list.append([idx, cx, cy])
        return lm_list

    def count_fingers(self, lm_list, hand_label="Right"):
        """Return number of fingers up (0-5) given a landmark list."""
        if not lm_list:
            return 0, []

        fingers_up = []

        # Thumb: compare x-coordinates (works differently since thumb moves sideways)
        # For a right hand, tip.x > joint.x means thumb is open; mirrored for left hand.
        if hand_label == "Right":
            fingers_up.append(1 if lm_list[self.tip_ids[0]][1] > lm_list[self.tip_ids[0] - 1][1] else 0)
        else:
            fingers_up.append(1 if lm_list[self.tip_ids[0]][1] < lm_list[self.tip_ids[0] - 1][1] else 0)

        # Other 4 fingers: tip.y < pip_joint.y means finger is raised (y grows downward in image coords)
        for tip_id in self.tip_ids[1:]:
            fingers_up.append(1 if lm_list[tip_id][2] < lm_list[tip_id - 2][2] else 0)

        return sum(fingers_up), fingers_up


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    detector = FingerDetector(max_hands=1)
    prev_time = 0

    if not cap.isOpened():
        print("Error: could not open webcam. Check camera index/permissions.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame.")
            break

        frame = cv2.flip(frame, 1)  # mirror view
        frame = detector.find_hands(frame)
        lm_list = detector.find_positions(frame)

        # Determine hand label (Left/Right) from mediapipe results
        hand_label = "Right"
        if detector.results.multi_handedness:
            hand_label = detector.results.multi_handedness[0].classification[0].label

        count, fingers_up = detector.count_fingers(lm_list, hand_label)

        # UI: finger count box
        cv2.rectangle(frame, (20, 20), (170, 140), (50, 50, 50), cv2.FILLED)
        cv2.putText(frame, str(count), (55, 115), cv2.FONT_HERSHEY_SIMPLEX,
                    3, (0, 215, 255), 6)

        # FPS counter
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 170), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

        cv2.putText(frame, f"Hand: {hand_label}", (20, 200), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 0), 2)

        cv2.imshow("Finger Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
