#imports
import cv2
import mediapipe as mp

#setups
mp_drawing=mp.solutions.drawing_utils
mp_drawing_styles=mp.solutions.drawing_styles
mp_pose=mp.solutions.pose


#webcam input
cap=cv2.VideoCapture(0)

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success,image=cap.read()
        image= cv2.resize(image, (1100, 600))
        if not success:
            print("Ignoring ")
            continue


        image.flags.writeable = False
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        results=pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            # landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )

        cv2.imshow("push-up", cv2.flip(image, 1))
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

cap.release()


