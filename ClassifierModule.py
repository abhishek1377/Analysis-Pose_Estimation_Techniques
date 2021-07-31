import cv2
import mediapipe as mp
import time

def classifier(lmList,landmarks):
    # for key,value in landmarks.items():
    #    print(key, " coordinates:","id:",value, "x: ", lmList[value][3],"y: ",lmList[value][4])

    if lmList[landmarks["right_hip"]][4] >  lmList[landmarks["right_shoulder"]][4]:
        return "downdog"

    elif lmList[landmarks["right_wrist"]][4] < lmList[landmarks["right_hip"]][4] or lmList[landmarks["left_wrist"]][4] < lmList[landmarks["right_hip"]][4]:
        return "plank"

    elif (lmList[landmarks["left_hip"]][3] < lmList[landmarks["left_heel"]][3] < lmList[landmarks["right_hip"]][3]) or (lmList[landmarks["left_hip"]][3] > lmList[landmarks["left_heel"]][3] > lmList[landmarks["right_hip"]][3]):
        return "tree"

    elif (lmList[landmarks["left_ear"]][3] < lmList[landmarks["nose"]][3] < lmList[landmarks["right_ear"]][3]) or (lmList[landmarks["left_ear"]][3] > lmList[landmarks["nose"]][3] > lmList[landmarks["right_ear"]][3]):
        return "goddess"
    else:
        return "warrior2"

def main():
   pass

if __name__ == "__main__":
    main()