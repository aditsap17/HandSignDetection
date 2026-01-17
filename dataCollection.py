import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300

folder = "data/S"
os.makedirs(folder, exist_ok=True)
counter = 0

def drawUI(frame, counter):
    ui = np.zeros((120, frame.shape[1], 3), dtype=np.uint8)

    cv2.putText(ui, "HAND SIGN DATA COLLECTION", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    cv2.putText(ui, f"Saved Images : {counter}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.putText(ui, "[S] Save   [Q] Quit", (frame.shape[1] - 260, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return ui

while True:
    success, img = cap.read()
    if not success:
        continue

    hands, _ = detector.findHands(img, draw=False)

    imgWhite = None

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        cv2.rectangle(img, (x - offset, y - offset),
                      (x + w + offset, y + h + offset),
                      (0, 255, 0), 2)

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

            preview = cv2.resize(imgWhite, (150, 150))
            img[20:170, img.shape[1]-170:img.shape[1]-20] = preview

    uiPanel = drawUI(img, counter)
    finalView = np.vstack((uiPanel, img))

    cv2.imshow("Hand Sign Collector", finalView)

    key = cv2.waitKey(1)

    if key == ord("s") and imgWhite is not None:
        counter += 1
        cv2.imwrite(f"{folder}/Image_{time.time()}.jpg", imgWhite)
        print("Saved:", counter)

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
