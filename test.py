import cv2
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

# ================== SETUP ==================
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imgSize = 300

# Kalimat hasil input
sentence = ""

# ================== LOOP ==================
while True:
    success, img = cap.read()
    if not success:
        continue

    hands, img = detector.findHands(img)

    imgWhite = None
    predictedChar = ""

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

            # ================== PREDIKSI ==================
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            predictedChar = chr(65 + index)  # A-Z

            # Preview hasil
            cv2.putText(img, f"Detected: {predictedChar}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            preview = cv2.resize(imgWhite, (150, 150))
            img[20:170, img.shape[1]-170:img.shape[1]-20] = preview

    # ================== TEXT BOX ==================
    cv2.rectangle(img, (20, img.shape[0]-80),
                  (img.shape[1]-20, img.shape[0]-20),
                  (0, 0, 0), -1)

    cv2.putText(img, sentence, (30, img.shape[0]-35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Hand Sign Test", img)

    # ================== KEYBOARD CONTROL ==================
    key = cv2.waitKey(1)

    # Tambah huruf HANYA saat ditekan
    if key == ord(" ") :
        sentence += " "

    elif key == 8:  # BACKSPACE
        sentence = sentence[:-1]

    elif key == ord("q"):
        break

    elif predictedChar != "" and 97 <= key <= 122:
        # a-z ditekan → ambil hasil deteksi
        sentence += predictedChar

# ================== CLEANUP ==================
cap.release()
cv2.destroyAllWindows()
