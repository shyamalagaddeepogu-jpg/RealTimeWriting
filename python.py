import cv2, mediapipe as mp, numpy as np, math, os
from tensorflow.keras.models import load_model

# ========== SETTINGS ==========
MODEL_PATH = "handwriting_model.h5"
IMG_SIZE = 100  # must match your training input
MIN_PINCH_DIST = 35  # lower value → easier writing

# ========== LOAD MODEL ==========
model = load_model(MODEL_PATH)
labels = [chr(i) for i in range(65, 91)]  # A–Z

# ========== INITIALIZE ==========
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
canvas = np.zeros((480, 640, 3), np.uint8)
sentence = ""
writing = False
prev_x = prev_y = None
xmin = ymin = 9999
xmax = ymax = 0

# ========== FUNCTIONS ==========
def dist(a, b): return math.hypot(a[0] - b[0], a[1] - b[1])

def predict_letter(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    gray = gray / 255.0
    pred = model.predict(gray.reshape(1, IMG_SIZE, IMG_SIZE, 1), verbose=0)
    return labels[np.argmax(pred)]

# ========== MAIN LOOP ==========
with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6) as hands:
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0].landmark
            x_i, y_i = int(lm[8].x * w), int(lm[8].y * h)
            x_t, y_t = int(lm[4].x * w), int(lm[4].y * h)
            d = dist((x_i, y_i), (x_t, y_t))

            # pinch gesture start
            if d < MIN_PINCH_DIST and not writing:
                writing = True
                prev_x, prev_y = x_i, y_i
                xmin, ymin, xmax, ymax = w, h, 0, 0

            # writing gesture
            if writing:
                if prev_x is not None:
                    if dist((prev_x, prev_y), (x_i, y_i)) < 50:
                        cv2.line(canvas, (prev_x, prev_y), (x_i, y_i), (255, 255, 255), 5)
                        xmin, ymin = min(xmin, x_i), min(ymin, y_i)
                        xmax, ymax = max(xmax, x_i), max(ymax, y_i)
                prev_x, prev_y = x_i, y_i

            # release pinch → recognize letter
            if d >= MIN_PINCH_DIST and writing:
                writing = False
                prev_x = prev_y = None
                if xmax > xmin and ymax > ymin:
                    letter_img = canvas[ymin:ymax, xmin:xmax]
                    if letter_img.size > 0:
                        letter = predict_letter(letter_img)
                        sentence += letter
                        print(f"Predicted: {letter}")
                        cv2.putText(canvas, letter, (xmin, ymin - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

            mp_draw.draw_landmarks(frame, res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

        view = cv2.addWeighted(frame, 0.7, canvas, 0.3, 0)
        cv2.putText(view, "Pinch=Draw | Release=Recognize | C=Clear | Q=Quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        cv2.putText(view, f"Output: {sentence}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        cv2.imshow("AI Air Writing", view)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('c'):
            canvas[:] = 0
            sentence = ""
        elif key in [27, ord('q')]:
            break

cap.release()
cv2.destroyAllWindows()
