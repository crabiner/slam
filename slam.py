#!/usr/bin/env python3
import cv2
from display import Display
import numpy as np
from frame import Frame, denormalize, match
import g2o

# camera intrinsics
WIDTH = 1920 // 2
HEIGHT = 1080 // 2

cx = WIDTH // 2
cy = HEIGHT // 2
F = 270
K = np.array([[F, 0, cx], [0, F, cy], [0, 0, 1]])
# principal point that is usually at the image center

# main classes
disp = Display(WIDTH, HEIGHT)

frames = []
def process_frame(img):
    img = cv2.resize(img, (WIDTH, HEIGHT))

    frame = Frame(img, K)
    frames.append(frame)
    if len(frames) <= 1:
        return

    ret, Rt = match(frames[-1], frames[-2])

    # print("%d matches" % (len(matches)))
    # print(pose)

    for pt1, pt2 in ret:
        u1, v1 = denormalize(K, pt1)
        u2, v2 = denormalize(K, pt2)

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
