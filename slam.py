#!/usr/bin/env python3
import cv2
from display import Display
import numpy as np
# np.set_printoptions(supress=True)
from extractor import Extractor

WIDTH = 1920 // 2
HEIGHT = 1080 // 2

disp = Display(WIDTH, HEIGHT)

F = 1
K = np.array([[F, 0, WIDTH // 2], [0, F, HEIGHT // 2], [0, 0, 1]])

fe = Extractor(WIDTH, HEIGHT)


def process_frame(img):
    img = cv2.resize(img, (WIDTH, HEIGHT))
    matches = fe.extract(img)

    if matches is None:
        return

    print("%d matches" % (len(matches)))

    for pt1, pt2 in matches:
        u1, v1 = fe.denormalize(pt1)
        u2, v2 = fe.denormalize(pt2)

        cv2.circle(img, (u1, v1), color=(0, 255, 0), radius=3)
        cv2.line(img, (u1, v1), (u2, v2), color=(255, 0, 0))

    disp.paint(img)


if __name__ == "__main__":
    cap = cv2.VideoCapture("videos/test_countryroad.mp4")

    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            process_frame(frame)
        else:
            break
