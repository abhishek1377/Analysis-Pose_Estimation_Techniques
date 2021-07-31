import cv2
import mediapipe as mp
import time

class poseDetector():

    def __init__(self,mode = True, upBody = False, smooth = True,
                 detectionCon = 0.5, trackCon = 0.5):

        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose()
        # self.pose = self.mpPose.Pose(self.mode,self.upBody,self.smooth,self.detectionCon,self.trackCon)

    def findPose(self, img, draw = True):

        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

	print("Testing5")
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img,self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h,w,c = img.shape
                cx,cy = int(lm.x * w), int(lm.y*h)
                lmList.append([id,cx,cy,lm.x,lm.y,lm.z])
                # if draw:
                #     cv2.circle(img,(cx,cy),5,(255,0,0),cv2.FILLED)
        return lmList

def assign_landmarks():
    landmarks = {}
    lmDescriptions = {}
    landmarks["nose"] = 0
    landmarks["left_eye"] = 2
    landmarks["right_eye"] = 5
    landmarks["left_ear"] = 7
    landmarks["right_ear"] = 8
    landmarks["mouth_left"] = 9
    landmarks["mouth_right"] = 10
    landmarks["left_shoulder"] = 11
    landmarks["right_shoulder"] = 12
    landmarks["left_elbow"] = 13
    landmarks["right_elbow"] = 14
    landmarks["left_wrist"] = 15
    landmarks["right_wrist"] = 16
    landmarks["left_pinky"] = 17
    landmarks["right_pinky"] = 18
    landmarks["left_index"] = 19
    landmarks["right_index"] = 20
    landmarks["left_thumb"] = 21
    landmarks["right_thumb"] = 22
    landmarks["left_hip"] = 23
    landmarks["right_hip"] = 24
    landmarks["left_knee"] = 25
    landmarks["right_knee"] = 26
    landmarks["left_ankle"] = 27
    landmarks["right_ankle"] = 28
    landmarks["left_heel"] = 29
    landmarks["right_heel"] = 30
    landmarks["left_foot_index"] = 31
    landmarks["right_foot_index"] = 32
    for key,value in landmarks.items():
        lmDescriptions[value] = key
    return landmarks

def main():
    cap = cv2.VideoCapture('Videos/3.mp4')
    pTime = 0
    detector = poseDetector()
    count = 0
    while True:
        success, img = cap.read()
        if success:
            img = detector.findPose(img)
            lmList = detector.findPosition(img)
            landmark_descriptions = assign_landmarks()
            print(lmList)
            if len(lmList) !=0:
                cv2.circle(img, (lmList[14][1], lmList[14][2]), 5, (0, 0, 255), cv2.FILLED)

            cTime = time.time()
            fps = 1/(cTime-pTime)
            pTime = cTime
            cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
            cv2.imshow("Image",img)
            cv2.waitKey(10)
        else:
            break

if __name__ == "__main__":
    main()
