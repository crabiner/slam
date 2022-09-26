#!/usr/bin/env python3
import cv2
from display import Display

WIDTH = 1920 // 2
HEIGHT = 1080 // 2

disp = Display(WIDTH, HEIGHT)

def process_frame(img):
    img = cv2.resize(img, (WIDTH, HEIGHT))
    disp.paint(img)
    # print(img.shape)



if __name__ == "__main__":
    cap = cv2.VideoCapture("videos/test_countryroad.mp4")

    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            process_frame(frame)
        else:
            break