import mediapipe as mp
import cv2
import math

import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def pose_estimate(pose_res, image):
    ##left_body_indices = [7,9,11,13,15,17,19,21,23,25,27,29,31]
    ##right_body_indices = [8,10,12,14,16,18,20,22,24,26,28,30,32]
    left_body_indices = [1, 2, 3, 7, 9, 11, 13, 15, 23, 25, 27, 29, 31]
    right_body_indices = [4, 5, 6, 8, 10, 12, 14, 16, 24, 26, 28, 30, 32]

    font = cv2.FONT_HERSHEY_SIMPLEX
    if pose_res != None:
        x = pose_res.landmark
        # print(x)
        count = 0
        red = (0, 0, 255)
        green = (0, 128, 0)

        left_avg = 0
        right_avg = 0
        left_body_cnt = 0
        right_body_cnt = 0
        face_cnt = 0

        lsa = math.sqrt((x[11].x - x[27].x) ** 2 + (x[11].y - x[27].y) ** 2)
        rsa = math.sqrt((x[12].x - x[28].x) ** 2 + (x[12].y - x[28].y) ** 2)

        for k in range(10):
            if (x[k].visibility > 0.5):
                face_cnt += 1

        for l in left_body_indices:
            left_avg += x[l].visibility
            if (x[l].visibility > 0.7):
                left_body_cnt += 1

        for r in right_body_indices:
            right_avg += x[r].visibility
            if (x[r].visibility > 0.7):
                right_body_cnt += 1

        left_avg /= len(left_body_indices)
        right_avg /= len(right_body_indices)

        for y in x:
            # print("this particular landmark: ", y)
            if y.visibility > 0.50:
                count += 1

        variation_avg = abs(left_avg - right_avg)
        variation_cnt = abs(left_body_cnt - right_body_cnt)

        pose = ""
        face = ""
        angle_toe_gnd = 0
        full_frame = 0

        if ((variation_avg <= 0.005) and (
                variation_cnt <= 2)):  ## Change made here (less than equal was modified) and threshold also changed
            if (face_cnt > 7):
                face = "Front"
            else:
                face = "Back"

            if count > 30:
                full_frame = 1
                if (abs(x[11].x - x[27].x) <= 0.9 * lsa) and (abs(x[12].x - x[28].x) <= 0.9 * rsa):
                    pose = "Straight"
            else:
                full_frame = 0

        else:
            if (right_body_cnt > left_body_cnt) or (right_avg > left_avg):  ##checking left or right
                face = "Right"
                if right_body_cnt >= 10:  ##Checking full body or not
                    full_frame = 1
                    ang_to_check = math.degrees(math.atan(abs(x[12].y - x[30].y) / abs(x[12].x - x[30].x)))
                    if ang_to_check > 85:
                        pose = "Straight"
                        angle_toe_gnd = ang_to_check
                    elif ang_to_check < 15:
                        pose = "Down"
                        angle_toe_gnd = math.degrees(math.atan(abs(x[12].y - x[32].y) / abs(x[12].x - x[32].x)))
                    else:
                        pose = "Inclined"
                        angle_toe_gnd = math.degrees(math.atan(abs(x[12].y - x[32].y) / abs(x[12].x - x[32].x)))
                else:
                    full_frame = 0

            elif (left_body_cnt > right_body_cnt) or (right_avg < left_avg):  # checking left or right
                face = "Left"
                if left_body_cnt >= 10:  ##Checking full body or not
                    full_frame = 1
                    ang_to_check = math.degrees(math.atan(abs(x[11].y - x[29].y) / abs(x[11].x - x[29].x)))
                    if ang_to_check > 85:
                        pose = "Straight"
                        angle_toe_gnd = ang_to_check
                    elif ang_to_check < 15:
                        pose = "Down"
                        angle_toe_gnd = math.degrees(math.atan(abs(x[11].y - x[31].y) / abs(x[11].x - x[31].x)))
                    else:
                        pose = "Inclined"
                        angle_toe_gnd = math.degrees(math.atan(abs(x[11].y - x[31].y) / abs(x[11].x - x[31].x)))
                else:
                    full_frame = 0

    # ang_knee_hip = math.degrees(math.atan(abs(x[23].y - x[25].y) / abs(x[23].x - x[25].x)))
    # ang_wrist_shoulder = math.degrees(math.atan(abs(x[11].y - x[15].y) / abs(x[11].x - x[15].x)))
    #
    # hand_straight_angle = math.degrees(math.atan(abs(x[11].y - x[15].y) / abs(x[11].x - x[15].x)))
    # elbow_straight_angle = math.degrees(math.atan(abs(x[14].y - x[16].y) / abs(x[14].x - x[16].x)))
    # elbow_flat_angle = math.degrees(math.atan(abs(x[12].y - x[14].y) / abs(x[12].x - x[14].x)))

    return face, pose, full_frame

def calculate_angle(a,b,c):
    a = np.array(a) #first
    b = np.array(b) #mid
    c = np.array(c) #End

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

dumbell_fly_cnt = 0
dumbell_fly_pose = True

def get_x_y_visibility(landmark):

    arr =[landmark.x,landmark.y,landmark.visibility]
    return arr

def dumbell_fly_rep(landmarks, facing_pose):
    global dumbell_fly_cnt
    global dumbell_fly_pose
    print("here:", landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value])

    # print("checkpoint1")
    if facing_pose == "Left" or facing_pose == "Right":
        # print("checkpoint2")
        if facing_pose == "Left":

            shoulder = get_x_y_visibility(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value])
            elbow = get_x_y_visibility(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value])
            wrist = get_x_y_visibility(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
            hip = get_x_y_visibility(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])

        elif facing_pose == "Right":
            print("checkpoint3")
            shoulder = get_x_y_visibility(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])
            elbow = get_x_y_visibility(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])
            wrist = get_x_y_visibility(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])
            hip = get_x_y_visibility(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value])

        left_wrist = get_x_y_visibility(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
        right_wrist = get_x_y_visibility(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])

        visibilty_shoulders = shoulder[2]
        visibilty_elbow = elbow[2]
        visibilty_wrist = wrist[2]
        visibilty_hip = hip[2]
        visibilty_left_wrist = left_wrist[2]
        visibilty_right_wrist = right_wrist[2]

        if visibilty_hip > .75 and visibilty_shoulders > .75:
            print("checkpoint4")
            radians = np.arctan2( hip[1] - shoulder[1],  hip[0] - shoulder[0])
            angle_of_upper_body_from_ground = np.abs(radians * 180.0 / np.pi)
            if angle_of_upper_body_from_ground > 180:
                angle_of_upper_body_from_ground = 360 - angle_of_upper_body_from_ground
            print("angle: ", angle_of_upper_body_from_ground)
            if angle_of_upper_body_from_ground < 40:
                print("checkpoint5")
                if  visibilty_elbow > .75:
                    print("checkpoint6")
                    length_of_arm = ((wrist[1]-elbow[1])**2 + (wrist[0]-elbow[0])**2)**0.5 +((shoulder[1]-elbow[1])**2 + (shoulder[0]-elbow[0])**2)**0.5
                    hip_to_shoulder = ((hip[1]-shoulder[1])**2 + (hip[0]-shoulder[0])**2)**0.5
                    curl_angle = calculate_angle(shoulder, elbow, wrist)
                    if curl_angle > 120:
                        print("checkpoint7")
                        if dumbell_fly_pose and wrist[1] > shoulder[1]-0.15:
                            print("checkpoint8")
                            dumbell_fly_pose = False

                        elif not dumbell_fly_pose and calculate_angle(elbow,shoulder,hip) > 75 and abs((wrist[1]-shoulder[1])) + 0.05 > length_of_arm and (abs(wrist[1]- shoulder[1])/hip_to_shoulder) > 0.75:
                            print("checkpoint9")
                            dumbell_fly_pose = True
                            dumbell_fly_cnt += 1

    return dumbell_fly_cnt

# cap = cv2.VideoCapture('Videos/Df.mp4')
cap = cv2.VideoCapture(0)
# Initiate holistic model
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        try:
            width = int(frame.shape[1] * 70 / 100)
            height = int(frame.shape[0] * 70 / 100)
            dim = (width, height)
            frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        except:
            break
        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Make Detections
        results = pose.process(image)
        # Recolor image back to BGR for rendering
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # Pose Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        # print(results.pose_landmarks.landmark)
        font = cv2.FONT_HERSHEY_SIMPLEX
        pose_res = ""
        face_res = ""

        if results.pose_landmarks != None:
            face_res, pose_res, full_frame_res = pose_estimate(results.pose_landmarks, image)
            landmarks = results.pose_landmarks.landmark

            if face_res != "":
                cv2.putText(image, face_res, (50, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4)
            if pose_res != "":
                cv2.putText(image, pose_res, (200, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4)

            curr_dumbell_fly_cnt = dumbell_fly_rep(landmarks, face_res)
            cv2.putText(image, "dumbell_fly : " + str(curr_dumbell_fly_cnt), (50, 150), font, 1, (0, 255, 255), 2, cv2.LINE_4)
            # cv2.putText(image, "ratio : " + str(ratio), (50, 300), font, 1, (0, 255, 255), 2, cv2.LINE_4)
            # cv2.putText(image, "angle : " + str(angle), (50, 300), font, 1, (0, 255, 255), 2, cv2.LINE_4)

        cv2.imshow('video', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
