import cv2
import mediapipe as mp
import os
import urllib.request


# --------------------------------------------------
# Download MediaPipe hand model only if needed
# --------------------------------------------------

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

# Get the folder where this Python file is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Save the model in the same folder as this Python file
MODEL_PATH = os.path.join(SCRIPT_DIR, "hand_landmarker.task")


# Download only if the model does not already exist
if not os.path.exists(MODEL_PATH):

    print("Downloading MediaPipe hand model...")

    urllib.request.urlretrieve(
        MODEL_URL,
        MODEL_PATH
    )

    print("Model downloaded successfully!")

else:

    print("MediaPipe hand model already exists.")
    print("No download needed.")


# --------------------------------------------------
# Create MediaPipe Hand Landmarker
# --------------------------------------------------

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)


hands = HandLandmarker.create_from_options(options)


# --------------------------------------------------
# Hand connections for drawing
# --------------------------------------------------

HAND_CONNECTIONS = [

    # Thumb
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    # Index finger
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    # Middle finger
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    # Ring finger
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    # Pinky
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),

    # Palm
    (5, 9),
    (9, 13),
    (13, 17)
]


# --------------------------------------------------
# Gesture detection function
# --------------------------------------------------

def detect_gesture(hand_landmarks):

    landmarks = hand_landmarks

    tip_ids = [4, 8, 12, 16, 20]
    pip_ids = [2, 6, 10, 14, 18]

    extended = 0


    # Check thumb
    if abs(
        landmarks[tip_ids[0]].x -
        landmarks[pip_ids[0]].x
    ) > 0.04:

        extended += 1


    # Check other four fingers
    for i in range(1, 5):

        if landmarks[tip_ids[i]].y < landmarks[pip_ids[i]].y:

            extended += 1


    # Classify gesture
    if extended >= 4:

        return "Open"

    elif extended <= 1:

        return "Closed Fist"

    else:

        return "Partial"


# --------------------------------------------------
# Open webcam
# --------------------------------------------------

cap = cv2.VideoCapture(0)


if not cap.isOpened():

    print("Error: Could not access the webcam.")

    exit()


print("Hand Tracking Started!")
print("Press 'q' to quit.")


# --------------------------------------------------
# Start webcam loop
# --------------------------------------------------

frame_timestamp_ms = 0


while True:

    # Read frame from webcam
    success, frame = cap.read()


    if not success:

        break


    # Flip image like a mirror
    frame = cv2.flip(frame, 1)


    # Get frame dimensions
    h, w, _ = frame.shape


    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # Convert OpenCV image to MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame_rgb
    )


    # Increase timestamp
    frame_timestamp_ms += 33


    # Detect hands
    results = hands.detect_for_video(
        mp_image,
        frame_timestamp_ms
    )


    # Default gesture
    gesture = "No hand detected"


    # --------------------------------------------------
    # Process detected hands
    # --------------------------------------------------

    if results.hand_landmarks:

        for idx, hand_landmarks in enumerate(
            results.hand_landmarks
        ):

            # Detect gesture
            gesture = detect_gesture(
                hand_landmarks
            )


            # --------------------------------------------------
            # Get left/right hand
            # --------------------------------------------------

            if idx < len(results.handedness):

                hand_label = (
                    results
                    .handedness[idx][0]
                    .category_name
                )
                if hand_label == "Left":
                    hand_label = "Right"
                else:
                    hand_label = "Left"
            else:

                hand_label = "Unknown"


            # --------------------------------------------------
            # Draw hand connections
            # --------------------------------------------------

            for connection in HAND_CONNECTIONS:

                start = connection[0]
                end = connection[1]


                x1 = int(
                    hand_landmarks[start].x * w
                )

                y1 = int(
                    hand_landmarks[start].y * h
                )


                x2 = int(
                    hand_landmarks[end].x * w
                )

                y2 = int(
                    hand_landmarks[end].y * h
                )


                cv2.line(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 255, 255),
                    2
                )


            # --------------------------------------------------
            # Draw fingertips
            # --------------------------------------------------

            fingertip_ids = [4, 8, 12, 16, 20]


            for tip_id in fingertip_ids:

                lm = hand_landmarks[tip_id]


                x = int(
                    lm.x * w
                )

                y = int(
                    lm.y * h
                )


                # Draw circle
                cv2.circle(
                    frame,
                    (x, y),
                    10,
                    (255, 0, 255),
                    cv2.FILLED
                )


                # Display landmark number
                cv2.putText(
                    frame,
                    str(tip_id),
                    (x - 5, y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2
                )


            # --------------------------------------------------
            # Display left/right hand label
            # --------------------------------------------------

            wrist = hand_landmarks[0]


            wrist_x = int(
                wrist.x * w
            )

            wrist_y = int(
                wrist.y * h
            )


            cv2.putText(
                frame,
                f"{hand_label} Hand",
                (wrist_x - 40, wrist_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )


    # --------------------------------------------------
    # Choose gesture text color
    # --------------------------------------------------

    if gesture in ["Open", "Closed Fist"]:

        status_color = (0, 255, 0)

    else:

        status_color = (0, 165, 255)


    # --------------------------------------------------
    # Display gesture
    # --------------------------------------------------

    cv2.putText(
        frame,
        f"Gesture: {gesture}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        status_color,
        2
    )


    # --------------------------------------------------
    # Show webcam
    # --------------------------------------------------

    cv2.imshow(
        "Hand Gesture Detection",
        frame
    )


    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# --------------------------------------------------
# Cleanup
# --------------------------------------------------

cap.release()

hands.close()

cv2.destroyAllWindows()