import cv2
import numpy as np
import mediapipe as mp
from math import sqrt

mp_pose = mp.solutions.pose


class DetectPose():
    pose_image = mp_pose.Pose(static_image_mode=False, smooth_landmarks=True, min_detection_confidence=0.5,
                              model_complexity=1)



    def __init__(self, mode=False, model_com=1, smooth_lm=True, detectioncon=0.5, trackcon=0.5):
        self.mode = mode
        self.model_com = model_com
        self.smooth_lm = smooth_lm
        self.detectioncon = detectioncon
        self.trackcon = trackcon
        # self.pose=mp.solutions.pose.Pose(self.mode,self.model_com,self.smooth_lm,self.detectioncon,self.trackcon)
        self.mp_draw = mp.solutions.drawing_utils  # for drawing the skeleton
        self.mp_pose = mp.solutions.pose

        self.dict_features = {}
        self.left_features = {
            'shoulder': 11,
            'elbow': 13,
            'wrist': 15,
            'hip': 23,
            'knee': 25,
            'ankle': 27,
            'foot': 31,
            'eye_inner': 1,
            'eye': 2,
            'eye_outer': 3,
            'ear': 7,
            'mouth': 9,
            'pinky': 17,
            'index': 19,
            'thumb': 21,
            'heel': 29,
            'foot_index': 31
        }

        self.right_features = {
            'shoulder': 12,
            'elbow': 14,
            'wrist': 16,
            'hip': 24,
            'knee': 26,
            'ankle': 28,
            'foot': 32,
            'eye_inner': 4,
            'eye': 5,
            'eye_outer': 6,
            'ear': 8,
            'mouth': 10,
            'pinky': 18,
            'index': 20,
            'thumb': 22,
            'heel': 30,
            'foot_index': 32
        }

        self.dict_features['left'] = self.left_features
        self.dict_features['right'] = self.right_features
        self.dict_features['nose'] = 0

    def detectPose(self, image, pose=pose_image,draw=False, display=False):
        output_image = image.copy()
        h, w, _ = image.shape
        frame_Dimension = [h, w]
        output_image = cv2.resize(output_image, (600, 600))

        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = pose.process(imageRGB)
        self.results = self.results.pose_landmarks

        if self.results and draw:
            self.mp_draw.draw_landmarks(image=output_image, landmark_list=self.results,
                                        connections=mp_pose.POSE_CONNECTIONS,
                                        landmark_drawing_spec=self.mp_draw.DrawingSpec(color=(255, 255, 255),
                                                                                       thickness=2, circle_radius=2),
                                        connection_drawing_spec=self.mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2,
                                                                                         circle_radius=0))

        if display:

            cv2.imshow("original image", image)
            cv2.imshow("output image", output_image)
        else:
            return output_image, self.results, frame_Dimension

    def get_positions(self, img,tar_angle,cnd_angle, draw=True):
        lmlist = []
        red = (0, 0, 255)
        green = (0, 255, 0)
        yellow = (0, 255, 255)
        max = 20
        min = 35



        # avg = 10

        if self.results:
            for idx, lm in enumerate(self.results.landmark):

                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([idx, cx, cy])
            if draw:
                circle_list=[11,12,13,14,23,24,25,26]

                for i  in range(len(circle_list)):
                        x,y=lmlist[circle_list[i]][1],lmlist[circle_list[i]][2]
                        self.mp_draw.draw_landmarks(image=img, landmark_list=self.results,
                                                    connections=mp_pose.POSE_CONNECTIONS,
                                                    landmark_drawing_spec=self.mp_draw.DrawingSpec(
                                                        color=(255, 255, 255),
                                                        thickness=2, circle_radius=2),
                                                    connection_drawing_spec=self.mp_draw.DrawingSpec(
                                                        color=(255, 255, 255), thickness=2,
                                                        circle_radius=0))
                        # print(x,y)
                        # cv2.circle(img, (x, y), 30, (255, 0, 255), 2)



                        if(tar_angle[i]-max<cnd_angle[i] and tar_angle[i]+min>cnd_angle[i] ):
                                # print("threshold",(tar_angle[i]-max , tar_angle[i]+min))
                                cv2.circle(img, (x, y), 20, green,2)
                        elif(tar_angle[i]-min<cnd_angle[i] and tar_angle[i]+min>cnd_angle[i] ):
                                cv2.circle(img, (x, y), 20, yellow, 2)
                        else:
                                cv2.circle(img, (x, y), 20, red, 2)



        return lmlist,img

    def get_landmark_array(self, pose_landmark, key, frame_width, frame_height):

        denorm_x = int(pose_landmark[key].x * frame_width)
        denorm_y = int(pose_landmark[key].y * frame_height)
        coordinates = [denorm_x, denorm_y]

        # return np.array([denorm_x, denorm_y])
        return coordinates

    def get_angle(self, img,draw=True):
        img=cv2.resize(img,(600,600))
        frame_height, frame_width, _ = img.shape
        feature = 'left'
        angle = []
        while (len(angle) != 8):
            elbow_angle = int(calculate_angle(
                self.get_landmark_array(self.results.landmark, self.dict_features[feature]['shoulder'], frame_width, frame_height),
                self.get_landmark_array(self.results.landmark, self.dict_features[feature]['elbow'], frame_width, frame_height),
                self.get_landmark_array(self.results.landmark, self.dict_features[feature]['wrist'], frame_width, frame_height)))
            angle.append(elbow_angle)

            shld_angle = int(calculate_angle(
                self.get_landmark_array(self.results.landmark, self.dict_features[feature]['elbow'], frame_width, frame_height),
                self.get_landmark_array(self.results.landmark, self.dict_features[feature]['shoulder'], frame_width, frame_height),
                self.get_landmark_array(self.results.landmark, self.dict_features[feature]['hip'], frame_width, frame_height)))
            angle.append(shld_angle)

            hip_angle = int(calculate_angle(
                self.get_landmark_array(self.results.landmark, self.dict_features[feature]['shoulder'], frame_width, frame_height),
                self.get_landmark_array(self.results.landmark, self.dict_features[feature]['hip'], frame_width, frame_height),
                self.get_landmark_array(self.results.landmark, self.dict_features[feature]['knee'], frame_width, frame_height)))
            angle.append(hip_angle)

            knee_angle = int(calculate_angle(
                self.get_landmark_array(self.results.landmark, self.dict_features[feature]['hip'], frame_width, frame_height),
                self.get_landmark_array(self.results.landmark, self.dict_features[feature]['knee'], frame_width, frame_height),
                self.get_landmark_array(self.results.landmark, self.dict_features[feature]['foot'], frame_width, frame_height)))
            angle.append(knee_angle)

            feature = 'right'

            if draw:
                if self.results and draw:
                    self.mp_draw.draw_landmarks(img, self.results, self.mp_pose.POSE_CONNECTIONS,
                                           self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                                           self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))

                    cv2.rectangle(img, (0, 0), (150, 300), (255, 255, 255), -1)
                    flag = 1
                    while (flag != 3):
                        j = 2
                        y = 30
                        if (flag == 1):
                            heading = 'ID'
                            x = 18
                        else:
                            heading = 'Angle'
                            x = 78
                        cv2.putText(img, heading, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .75, [0, 0, 255], 2, cv2.LINE_AA)
                        for idx, val in enumerate(angle):
                            if (flag == 1):
                                var = idx + 1
                            else:
                                var = val
                            cv2.putText(img, str(var), (x, y * j), cv2.FONT_HERSHEY_SIMPLEX, .75, [0, 0, 255], 2,
                                        cv2.LINE_AA)
                            j += 1
                        flag += 1

                # return image

        return angle,img

    def get_landmark(self, results, dmlist):
        # results=results.landmark
        keypoints = []

        # if self.results:
        if results:
            # print("landmarks:")
            # print(self.results.landmark)
            for point in results.landmark:
                keypoints.append({
                    'X': point.x * dmlist[1],
                    'Y': point.y * dmlist[0],
                    'Z': point.z,
                })
                # print("keypoints")
                # print(keypoints)

                return keypoints

    def cosine_sim(self, x, y):
        sum = 0;
        sumx = 0;
        sumy = 0;
        for i, j in zip(x, y):
            sum += i * j
            sumx += i * i
            sumy += j * j
        return sum / ((sqrt(sumx)) * (sqrt(sumy)))

    def diff_compare_points(self, cndlist, targlist, cnd_dim, tar_dim):
        average = []
        cndlist = self.get_landmark(cndlist, cnd_dim)
        targlist = self.get_landmark(targlist, tar_dim)


        for i, j in zip(range(len(list(cndlist))), range(len(list(targlist)))):
            s1 = self.cosine_sim(list(cndlist[i].values()), list(targlist[j].values()))
            print("==================================================:  ",list(cndlist[i].values()))

            average.append(s1)
        p_score = sqrt(2 * (1 - round(self.Average(average), 2)))

        return 1 - p_score
    def diff_compare_anlges(self,target_angle,cnd_angle,draw=False):
        new_x = []
        for i, j in zip(range(len(target_angle)), range(len(cnd_angle))):
            z = np.abs(target_angle[i] - cnd_angle[j]) / (target_angle[i] + cnd_angle[j] / 2)
            new_x.append(z)
        a_score =self.Average(new_x)

        return 1-a_score

    def Average(self, lst):
        return sum(lst) / len(lst)






def calculate_angle(a, b, c):

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180:
        angle = 360 - angle
    return angle



def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
    decImg = DetectPose()
    decWeb = DetectPose()
    # img = "C://Users//Admin//PycharmProjects//EIPSEM4//images//ab.png"
    img = 'C://Users//prera//OneDrive//Desktop//MCT//MCT 2.0//man//standing_pose.jpg'
    black_screen= cv2.imread("C://Users//prera//OneDrive//Desktop//MCT//IvxTk.png")
    black_screen=cv2.resize(black_screen, (750, 685))
    # img=cv2.flip(img,1)
    img = cv2.imread(img)

    target_img1, target_results, tar_dim = decImg.detectPose(img, draw=True)
    # l1,target_img = decImg.get_positions(img)
    target_angle,target_img=decImg.get_angle(img)
    # print("angle target: ",target_angle)
    # target_img=cv2.resize(target_img,(600,600))
    # target_list=decImg.get_landmark(img)
    # h,w,c=img.shape

    while cap.isOpened():
        success, frame = cap.read()
        frame = cv2.flip(frame, 1)
        web_img1, cnd_results, cnd_dim = decWeb.detectPose(frame,draw=True)




        if not success:
            continue
        output=None
        pre_score = 0
        msg="Image is not found"
        if cnd_results:
            web_angle, _ = decWeb.get_angle(frame)
            l1, web_img = decWeb.get_positions(frame, target_angle, web_angle)



            # print("webangle  ", web_angle)

            score = decWeb.diff_compare_points(cnd_results, target_results, cnd_dim, tar_dim)
            # score=decWeb.diff_compare_anlges(target_angle,web_angle)
            pre_score = score * 100
            if (pre_score < 0 and pre_score < (-50)):
                pre_score = 0
            else:
                pre_score = np.abs(pre_score)
            # msg=int(pre_score)
            cv2.putText(web_img, str(int(pre_score)) + '%', (130, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
            output=web_img


        else:
            cv2.putText(black_screen, str(msg), (130, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
            output=black_screen



        output = cv2.resize(output, (750, 685))
        cv2.imshow("output_result", output)
        cv2.imshow("target", target_img)



        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()  # close camera
    cv2.destroyAllWindows()  # cl


if __name__ == "__main__":
    main()
    print("hello")



