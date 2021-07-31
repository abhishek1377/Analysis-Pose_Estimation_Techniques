import cv2
import numpy as np
import matplotlib as plt
import time
import PoseModule as pm
import ClassifierModule as cm
import os
pTime = 0
detector = pm.poseDetector()

landmarks = pm.assign_landmarks()
detector = pm.poseDetector()
category = None
img = cv2.imread(os.path.join("yoga poses/Train/goddess","00000397.jpg"))
cv2.imshow("Image", img)

img = detector.findPose(img)
lmList = detector.findPosition(img)
print(len(lmList))
if len(lmList) != 0:
    print("here")
    for i in range(len(lmList)):
        lmList[i][4] = 1 - lmList[i][4]
    category = cm.classifier(lmList, landmarks)
    if category == "goddess":
        # categories[category] += 1
        # successful_count += 1
        pass

root = "yoga poses/Train/goddess"
fileName = "00000400.jpg"
img = cv2.imread(os.path.join(root,fileName))
img = detector.findPose(img)
lmList = detector.findPosition(img)
if len(lmList) != 0:
    for i in range(len(lmList)):
        lmList[i][4] = 1 - lmList[i][4]
    category = cm.classifier(lmList, landmarks)
print(lmList)
cv2.imshow("Image", img)
cv2.waitKey(1000)
print(category)