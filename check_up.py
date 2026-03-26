import cv2 #imports OpenCV to the environment for face recognition
import numpy as np #for maths
import mediapipe as mp #for detecting any facial movements

print(f"OpenCV version: {cv2.__version__}") #to get version of opencv

cap = cv2.VideoCapture(0) 
if cap.isOpened():
    print("Webcam: WORKING")
    cap.release()
else:
    print("Webcam: NOT FOUND")

    
