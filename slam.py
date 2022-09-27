#!/usr/bin/env python3
import cv2
from display import Display

WIDTH = 1920 // 2
HEIGHT = 1080 // 2
MAX_FEATURES = 700

disp = Display(WIDTH, HEIGHT)
orb = cv2.ORB_create(MAX_FEATURES)
# features track point from one image to the next image
def process_frame(img):
    keypoints, descriptors = orb.detectAndCompute(img, None)

    for point in keypoints:
        u, v = map(lambda x: int(round(x)), point.pt)
        cv2.circle(img, (u, v), color=(0, 255, 0), radius=3)

    img = cv2.resize(img, (WIDTH, HEIGHT))
    disp.paint(img)

if __name__ == "__main__":
    cap = cv2.VideoCapture("videos/test_countryroad.mp4")

    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            process_frame(frame)
        else:
            break