import cv2
import time
import PoseModule as pm

cap = cv2.VideoCapture('yoga poses/Train/warrior2/00000254.jpg')
# cap = cv2.VideoCapture('Videos/2.mp4')
pTime = 0
detector = pm.poseDetector()
while True:
    success, img = cap.read()
    if success:
        img = detector.findPose(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[14])
            cv2.circle(img, (lmList[14][1], lmList[14][2]), 10, (0, 0, 255), cv2.FILLED)
        cTime = time.time()
        # print("cTime:",cTime,"pTime:", pTime)
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img,str(int(fps)),(70,50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
        cv2.imshow("Image",img)
        cv2.waitKey(10000)
    else:
        break
