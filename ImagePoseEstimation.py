import cv2
import time
import PoseModule as pm
import ClassifierModule as cm
import os
pTime = 0
detector = pm.poseDetector()
landmarks = pm.assign_landmarks()

def single_image_classification(imgPath):
    pTime = 0
    category = "unknown"
    cap = cv2.VideoCapture(imgPath)
    success, img = cap.read()
    if success:
        img = detector.findPose(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            for i in range(len(lmList)):
                lmList[i][4] = 1 - lmList[i][4]
            category = cm.classifier(lmList, landmarks)
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img,category,(70,50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
        cv2.imshow("Image",img)
        cv2.waitKey(10000)


def image_classifaction_through_folder(directory):
    categories = {}
    categories["downdog"] = 0
    categories["goddess"] = 0
    categories["plank"] = 0
    categories["tree"] = 0
    categories["warrior2"] = 0
    total_count = 0
    successful_count = 0
    unsuccessful_classified = []

    for root,_,files in os.walk(directory):

        for fileName in files:
            total_count += 1
            category = "unknown"
            current_pose_directory = root.split("\\")[-1]
            try:
                cap = cv2.VideoCapture(os.path.join(root,fileName))
                while True:
                    success, img = cap.read()
                    if success:
                        img = detector.findPose(img)
                        lmList = detector.findPosition(img)
                        if len(lmList) != 0:
                            for i in range(len(lmList)):
                                lmList[i][4] = 1 - lmList[i][4]
                            category = cm.classifier(lmList, landmarks)
                            if category == current_pose_directory:
                                categories[category] += 1
                                successful_count += 1
                            else:
                                unsuccessful_classified.append(os.path.join(root,fileName))

                    else:
                        break
                    break
            except:
                img = cv2.imread(os.path.join(root, fileName))
                img = detector.findPose(img)
                lmList = detector.findPosition(img)
                if len(lmList) != 0:
                    for i in range(len(lmList)):
                        lmList[i][4] = 1 - lmList[i][4]
                    category = cm.classifier(lmList, landmarks)
                    if category == current_pose_directory:
                        categories[category] += 1
                        successful_count += 1
                    else:
                        unsuccessful_classified.append(os.path.join(root, fileName))


            print("after classification:", "root: ",root," filename:",fileName,"current pose being identified: ", current_pose_directory," identified pose: ", category, successful_count, total_count)
    print(categories,successful_count,total_count)
    return unsuccessful_classified

def main():
    directory = 'yoga poses/Train'
    print("git test")
    failed = image_classifaction_through_folder(directory)
    # for img in failed:
    #     print(img)
    #     single_image_classification(img)

if __name__ == "__main__":
    main()

