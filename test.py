import cv2
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imgSize = 300


def detect_hand_sign(img):
    hands, _ = detector.findHands(img, draw=False)

    if not hands:
        return img, "", 0

    hand = hands[0]
    x, y, w, h = hand['bbox']

    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

    hImg, wImg, _ = img.shape
    x1, y1 = max(0, x-offset), max(0, y-offset)
    x2, y2 = min(wImg, x+w+offset), min(hImg, y+h+offset)

    imgCrop = img[y1:y2, x1:x2]
    if imgCrop.size == 0:
        return img, "", 0

    aspectRatio = h / w

    if aspectRatio > 1:
        k = imgSize / h
        wCal = math.ceil(k * w)
        imgResize = cv2.resize(imgCrop, (wCal, imgSize))
        wGap = (imgSize - wCal) // 2
        imgWhite[:, wGap:wGap+wCal] = imgResize
    else:
        k = imgSize / w
        hCal = math.ceil(k * h)
        imgResize = cv2.resize(imgCrop, (imgSize, hCal))
        hGap = (imgSize - hCal) // 2
        imgWhite[hGap:hGap+hCal, :] = imgResize

    predictions, index = classifier.getPrediction(imgWhite, draw=False)
    confidence = round(predictions[index] * 100, 2)
    predictedChar = chr(65 + index)

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return img, predictedChar, confidence
