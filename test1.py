import cv2
import mediapipe as mp


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils


def is_hello_sign(landmarks):
    if landmarks:
      
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]

        if (thumb_tip.y < landmarks[2].y and
            index_tip.y < landmarks[6].y and
            middle_tip.y < landmarks[10].y and
            ring_tip.y < landmarks[14].y and
            pinky_tip.y < landmarks[18].y):
            return True
    return False


def is_yes_sign(landmarks):
    if landmarks:
       
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]

        if (thumb_tip.y < landmarks[2].y and
            index_tip.y > landmarks[6].y and
            middle_tip.y > landmarks[10].y and
            ring_tip.y > landmarks[14].y and
            pinky_tip.y > landmarks[18].y):
            return True
    return False

# Function to check for "No" sign (index and middle fingers extended, others folded)
# def is_no_sign(landmarks):
#     if landmarks:
#         # Check if the index and middle fingers are extended and others are folded
#         index_tip = landmarks[8]
#         middle_tip = landmarks[12]
#         ring_tip = landmarks[16]
#         pinky_tip = landmarks[20]
#         thumb_tip = landmarks[4]

#         if (index_tip.y < landmarks[6].y and
#             middle_tip.y < landmarks[10].y and
#             ring_tip.y > landmarks[14].y and
#             pinky_tip.y > landmarks[18].y and
#             thumb_tip.y > landmarks[2].y):
#             return True
#     return False
# Function to check for "No" sign (index and middle fingers extended, others folded)
def is_no_sign(landmarks):
    if landmarks:
       
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        thumb_tip = landmarks[4]

       
        if (index_tip.y < landmarks[6].y and   # Index extended (tip is higher than the second joint)
            middle_tip.y < landmarks[10].y and # Middle extended (tip is higher than the second joint)
            ring_tip.y > landmarks[14].y and   # Ring folded (tip is lower than the second joint)
            pinky_tip.y > landmarks[18].y and  # Pinky folded (tip is lower than the second joint)
            thumb_tip.y > landmarks[2].y):    # Thumb folded (tip is lower than the second joint)
            return True
    return False


# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip the image horizontally for a later selfie-view display, and convert the BGR image to RGB.
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and find hands
    results = hands.process(image_rgb)

    # Draw hand annotations on the image
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2))

            # Check for "Hello", "Yes", or "No" signs
            if is_hello_sign(hand_landmarks.landmark):
                cv2.putText(image, 'Hello Sign Detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            elif is_yes_sign(hand_landmarks.landmark):
                cv2.putText(image, 'Yes Sign Detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            elif is_no_sign(hand_landmarks.landmark):
                cv2.putText(image, 'No Sign Detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the image
    cv2.imshow('Sign Language Detection', image)

    # Break loop on 'q' key press
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

