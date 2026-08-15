import cv2, mediapipe as mp, numpy as np
from pycaw.pycaw import AudioUtilities
import screen_brightness_control as sbc

Hands = mp.solutions.hands
hands = Hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
draw = mp.solutions.drawing_utils
TH, IX = Hands.HandLandmark.THUMB_TIP, Hands.HandLandmark.INDEX_FINGER_TIP

try:
    dev = AudioUtilities.GetSpeakers()
    volctl = dev.EndpointVolume
    minv, maxv = volctl.GetVolumeRange()[:2]

    print("Audio device:", dev.FriendlyName)
    print("Volume range:", minv, "to", maxv)

except Exception as e:
    print(f"Pycaw error: {e}")
    exit()

cap = cv2.VideoCapture(0)
if not cap.isOpened(): print("Error: Webcam not accessible."); exit()

WIN = "Hand Gesture Control"; cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

while True:
    ok, img = cap.read()
    if not ok: break
    img = cv2.flip(img, 1); h, w = img.shape[:2]
    res = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if res.multi_hand_landmarks and res.multi_handedness:
        for i, hand in enumerate(res.multi_hand_landmarks):
            label = res.multi_handedness[i].classification[0].label
            draw.draw_landmarks(img, hand, Hands.HAND_CONNECTIONS)
            lm = hand.landmark
            tp = (int(lm[TH].x*w), int(lm[TH].y*h)); ip = (int(lm[IX].x*w), int(lm[IX].y*h))
            cv2.circle(img, tp, 10, (255,0,0), cv2.FILLED); cv2.circle(img, ip, 10, (255,0,0), cv2.FILLED)
            cv2.line(img, tp, ip, (0,255,0), 3)
            dist = float(np.hypot(ip[0]-tp[0], ip[1]-tp[1]))

            if label == "Left":  # real RIGHT hand -> volume (frame is flipped)
                v = np.interp(dist, [30,200], [minv,maxv])
                try: volctl.SetMasterVolumeLevel(v, None)
                except Exception as e: print(f"Volume error: {e}")
                bar = int(np.interp(dist, [30,200], [400,150])); pct = int(np.interp(dist, [30,200], [0,100]))
                cv2.rectangle(img, (50,150), (85,400), (255,0,0), 2); cv2.rectangle(img, (50,bar), (85,400), (255,0,0), cv2.FILLED)
                cv2.putText(img, f"{pct}%", (40,450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 3)

            elif label == "Right":  # real LEFT hand -> brightness
                b = int(np.interp(dist, [30,200], [0,100]))
                try: sbc.set_brightness(b)
                except Exception as e: print(f"Brightness error: {e}")
                bar = int(np.interp(dist, [30,200], [400,150])); x1, x2 = w-85, w-50
                cv2.rectangle(img, (x1,150), (x2,400), (0,255,0), 2); cv2.rectangle(img, (x1,bar), (x2,400), (0,255,0), cv2.FILLED)
                cv2.putText(img, f"{b}%", (w-110,450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)

    cv2.imshow(WIN, img)
    k = cv2.waitKey(1) & 0xFF
    if k in (27, ord("q")): break
    try:
        if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1: break
    except cv2.error:
        break

cap.release(); cv2.destroyAllWindows()