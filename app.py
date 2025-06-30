import cv2
import math
import numpy as np
import mediapipe as mp

class PoseDetection():

    def __init__(self,  mode=False, upBody=False, smooth=True, detectionCon=0.5, trackCon=0.5):
        self.upBody = upBody
        self.smooth = smooth
        self.mode = mode
        self.detection = detectionCon
        self.trackCon = trackCon

        self.counter = 0 
        self.down_thresh = 100.0
        self.up_thresh = 50.0
        self.down = False

        self.mpPose = mp.solutions.pose
        self.mpDraw = mp.solutions.drawing_utils
        self.pose = self.mpPose.Pose(
                        static_image_mode=self.mode,
                        model_complexity=1,
                        smooth_landmarks=self.smooth,
                        enable_segmentation=False,
                        min_detection_confidence=self.detection,
                        min_tracking_confidence=self.trackCon
                    )

    def find_position(self, frame, draw = True):
        self.results = self.pose.process(frame)

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(frame, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return frame
    
    def find_landmarks(self, frame):
        self.LM_points = []
        self.elbow_points = []
        h, w, c = frame.shape        
        
        if self.results.pose_landmarks:
            for idx, landmarks in enumerate(self.results.pose_landmarks.landmark):
                self.LM_points.append([int(landmarks.x * w), int(landmarks.y * h)])
                if idx == 14:
                    self.elbow_points.append([landmarks.x, landmarks.y])
                cv2.circle(frame, (int(landmarks.x * w),int(landmarks.y*h)), 5, (255, 255, 0), cv2.FILLED)  
        return self.LM_points

    def angle(self, img, right_shoulder, right_elbow, right_wrist, left_shoulder, left_elbow, left_wrist,  draw = True):
        x1, y1 = self.LM_points[right_shoulder]
        x2, y2 = self.LM_points[right_elbow]
        x3, y3 = self.LM_points[right_wrist]

        x1_, y1_ = self.LM_points[left_shoulder]
        x2_, y2_ = self.LM_points[left_elbow]
        x3_, y3_ = self.LM_points[left_wrist]


        angle_right_hand = math.degrees(math.atan2(y2-y1, x2-x1) - math.atan2(y3-y2, x3-x2))
        angle_left_hand = math.degrees(math.atan2(y2_-y1_, x2_-x1_) - math.atan2(y3_-y2_, x3_-x2_))
        
        if angle_right_hand < 0:
            angle_right_hand += 360
        
        if angle_left_hand < 0:
              angle_left_hand = np.abs(angle_left_hand)

        if img is not None:
            if draw:
                cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
                cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
                cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
                cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
                cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
                cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
                cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
                cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
                cv2.putText(img, str(int(angle_right_hand)), (x2 - 50, y2 + 50),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)


                cv2.line(img, (x1_, y1_), (x2_, y2_), (0, 255, 255), 3)
                cv2.line(img, (x3_, y3_), (x2_, y2_), (0, 255, 255), 3)
                cv2.circle(img, (x1_, y1_), 10, (0, 255, 0), cv2.FILLED)
                cv2.circle(img, (x2_, y2_), 10, (0, 255, 0), cv2.FILLED)
                cv2.circle(img, (x3_, y3_), 10, (0, 255, 0), cv2.FILLED)
                cv2.putText(img, str(int(angle_left_hand)), (x2_ - 50, y2_ + 50),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)

                return angle_right_hand, angle_left_hand            
        
        return None
    
    def count_reps(self, frame, angle_right_hand, angle_left_hand):  

        # while up
        if angle_right_hand < self.up_thresh and angle_left_hand < self.up_thresh:
            if not self.down:
                self.down = True
        
        # while down
        if angle_right_hand > self.down_thresh and angle_left_hand > self.down_thresh:
            if self.down:
                self.counter += 1
                self.down = False   

        cv2.putText(frame, f'Reps: {self.counter}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        return self.counter   

def main():

    pd = PoseDetection()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frames = pd.find_position(frame_rgb)    
        print(pd.find_landmarks(frame=frames))

        frames_bgr = cv2.cvtColor(frames, cv2.COLOR_RGB2BGR)

        angle_right, angle_left = pd.angle(frames_bgr, 11, 13, 15, 12, 14, 16, draw=True)


        pd.count_reps(frames_bgr, angle_left_hand=angle_left, angle_right_hand=angle_right)
        cv2.imshow("Frames", frames_bgr)
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()
    
        
if __name__ == '__main__':
    main()