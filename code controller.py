import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize MediaPipe Hand model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Variables to track swipe direction and action
start_position_index = None
start_position_middle = None
swipe_threshold = 60  # Minimum distance for a valid swipe
swipe_completed = False
cooldown = False  # Cooldown flag to prevent rapid slide changes
cooldown_time = 1  # Cooldown duration in seconds
last_swipe_time = time.time()  # Timestamp of the last swipe action

def check_swipe_completed(start, end):
    """Check if a complete to-and-fro swipe has been done."""
    if abs(start - end) > swipe_threshold:
        return True
    return False

def get_gesture(landmarks):
    global start_position_index, start_position_middle, swipe_completed, cooldown

    # Get current X-coordinates of index finger (landmark 8) and middle finger (landmark 12)
    current_x_index = landmarks[8].x * 1000  # Multiply by 1000 for better precision
    current_x_middle = landmarks[12].x * 1000

    # If start positions are not initialized, set them
    if start_position_index is None and start_position_middle is None:
        start_position_index = current_x_index
        start_position_middle = current_x_middle
        return None

    # Check if a swipe is completed
    if not swipe_completed and not cooldown:
        if check_swipe_completed(start_position_index, current_x_index) and \
           check_swipe_completed(start_position_middle, current_x_middle):
            swipe_completed = True  # Mark the swipe as completed
            if current_x_index > start_position_index and current_x_middle > start_position_middle:
                return "right"  # Swipe right completed
            elif current_x_index < start_position_index and current_x_middle < start_position_middle:
                return "left"  # Swipe left completed

    # Once a swipe is completed, reset the start positions
    if swipe_completed:
        start_position_index = current_x_index
        start_position_middle = current_x_middle
        swipe_completed = False

    return None

while cap.isOpened():
    ret, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    # Only proceed if at least one hand is detected
    if result.multi_hand_landmarks:

        # Check if only one hand is detected (ignore if two hands are detected)
        if len(result.multi_hand_landmarks) == 1:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Ensure that the full hand is detected by checking for all 21 landmarks
                if len(hand_landmarks.landmark) == 21:
                    gesture = get_gesture(hand_landmarks.landmark)

                    # Check if enough time has passed since the last swipe (cooldown)
                    current_time = time.time()
                    if gesture and (current_time - last_swipe_time) > cooldown_time:
                        if gesture == "left":
                            pyautogui.press('left')  # Move to the previous slide
                            print("Previous Slide")
                        elif gesture == "right":
                            pyautogui.press('right')  # Move to the next slide
                            print("Next Slide")

                        # Start cooldown and update the last swipe time
                        cooldown = True
                        last_swipe_time = current_time

        else:
            # Ignore if two hands are detected
            print("Two hands detected. Ignoring gestures.")

    # Reset cooldown after the duration has passed
    if cooldown and (time.time() - last_swipe_time) > cooldown_time:
        cooldown = False

    cv2.imshow("Hand Gesture Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()