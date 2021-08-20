#!/usr/bin/env python
# coding: utf-8

# Importing Packages
import numpy as np
import mediapipe as mp
import cv2

# Video Capturion and making Detections
# reps ctr keeps track how many repetitions they have completed while stage shows the state of the rep they are in (up/down)
reps_ctr = 0 
stage = None

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
cap = cv2.VideoCapture(0)

# mediapipe instance setup
with mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        # BGR -> RGB
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img.flags.writeable = False
      
        # make detections
        res = pose.process(img)
        
        # RGB -> BGR
        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # drawing out the landmarks
        try:
            landmarks = res.pose_landmarks.landmark
            
            # obtaining the coordinates of the arm (wrist, elbow, shoulder)
            wrist = np.array([landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y])
            elbow = np.array([landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y])
            shoulder = np.array([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y])
            
            # calculate angles & rad -> deg
            angle_rad = np.arctan2(wrist[1]-elbow[1], wrist[0]-elbow[0]) - np.arctan2(shoulder[1]-elbow[1], shoulder[0]-elbow[0])
            angle_deg = np.abs(angle_rad*180.0/np.pi)
            
            if angle_deg > 180.0:
                angle_deg = 360-angle_deg
                
            # logic for a proper curl
            if angle_deg > 160:
                stage = "down"
            if angle_deg < 30 and stage =='down':
                stage="up"
                reps_ctr +=1
        except:
            pass
    
        # rendering reps number and status along with labels for each
        cv2.putText(img, 'Reps', (15,400), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
        cv2.putText(img, str(reps_ctr), 
                    (15,440), 
                    cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
        cv2.putText(img,'Stage', (65,400), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
        cv2.putText(img, stage, 
                    (65,440), 
                    cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
        # rendering detections
        mp_drawing.draw_landmarks(img, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )               
        
        cv2.imshow('Workout Frame', img)
        
        # exit
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()





