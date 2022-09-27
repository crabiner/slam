#!/usr/bin/env python3
import cv2
from display import Display
import numpy as np
from extractor import Extractor

WIDTH = 1920 // 2
HEIGHT = 1080 // 2

disp = Display(WIDTH, HEIGHT)


fe = Extractor()


def process_frame(img):
    img = cv2.resize(img, (WIDTH, HEIGHT))
    keypoints, descriptors, matches = fe.extract(img)

    if matches is None:
        return

    for point in keypoints:
        u, v = map(lambda x: int(round(x)), point.pt)
        cv2.circle(img, (u, v), color=(0, 255, 0), radius=3)

    for match in matches:
        print(match)

    disp.paint(img)


if __name__ == "__main__":
    cap = cv2.VideoCapture("videos/test_countryroad.mp4")

    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            process_frame(frame)
        else:
            break
