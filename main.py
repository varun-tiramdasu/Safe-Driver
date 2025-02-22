import cv2
import mediapipe as mp
import numpy as np
import time
from scipy.spatial import distance
import face_recognition
from ultralytics import YOLO
import pygame

# Initialize pygame for sound
pygame.mixer.init()

# Load the alert sound (make sure you have a sound file named "alert.wav" in the same directory)
alert_sound = pygame.mixer.Sound("music2.mp3")

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

model = YOLO("yolov8n-seg.pt")

# Eye Aspect Ratio (EAR)
