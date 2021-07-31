import cv2
import mediapipe as mp
import time
import numpy as np
import PoseModule as pm


mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()
cap = cv2.VideoCapture(0)
fourcc= cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc,20.0,(640,480))
detector = pm.poseDetector()
pTime = 0

while True:
    success, img = cap.read()
    if success:

        cv2.imshow("Img", img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)
        img = detector.findPose(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList)
        cTime = time.time()
        # print("cTime:",cTime,"pTime:", pTime)
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        out.write(img)
        # cv2.imshow('gray', gray)
        cv2.imshow("Image", img)
        # cv2.waitKey(1000)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
#
    else:
        break

# while True:
#
#     success,img = cap.read()
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     cv2.imshow('frame',img)
#     cv2.imshow('gray',gray)
#
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

cap.release()
out.release()
cv2.destroyAllWindows()

