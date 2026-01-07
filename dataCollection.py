import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

# ================== SETUP ==================
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300

folder = "data/B"
os.makedirs(folder, exist_ok=True)
counter = 0

# ================== UI FUNCTION ==================
def drawUI(frame, counter):
    ui = np.zeros((120, frame.shape[1], 3), dtype=np.uint8)

    cv2.putText(ui, "HAND SIGN DATA COLLECTION", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    cv2.putText(ui, f"Saved Images : {counter}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.putText(ui, "[S] Save   [Q] Quit", (frame.shape[1] - 260, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return ui

# ================== LOOP ==================
while True:
    success, img = cap.read()
    if not success:
        continue

    hands, img = detector.findHands(img)

    # ================== HAND PROCESS ==================
    imgWhite = None

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        hImg, wImg, _ = img.shape
        x1 = max(0, x - offset)
        y1 = max(0, y - offset)
        x2 = min(wImg, x + w + offset)
        y2 = min(hImg, y + h + offset)

        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.size != 0:
            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wGap + wCal] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hGap + hCal, :] = imgResize

            # Tampilkan preview kecil di frame utama
            preview = cv2.resize(imgWhite, (150, 150))
            img[20:170, img.shape[1]-170:img.shape[1]-20] = preview

    # ================== COMPOSE UI ==================
    uiPanel = drawUI(img, counter)
    finalView = np.vstack((uiPanel, img))

    cv2.imshow("Hand Sign Collector", finalView)

    key = cv2.waitKey(1)

    # ================== SAVE ==================
    if key == ord("s") and imgWhite is not None:
        counter += 1
        cv2.imwrite(f"{folder}/Image_{time.time()}.jpg", imgWhite)
        print("Saved:", counter)

    # ================== EXIT ==================
    if key == ord("q"):
        break

# ================== CLEANUP ==================
cap.release()
cv2.destroyAllWindows()
