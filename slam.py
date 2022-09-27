#!/usr/bin/env python3
import cv2
from display import Display
import numpy as np

WIDTH = 1920 // 2
HEIGHT = 1080 // 2
MAX_FEATURES = 100

disp = Display(WIDTH, HEIGHT)


class FeatureExtractor(object):
    def __init__(self):
        self.orb = cv2.ORB_create(MAX_FEATURES)

    def extract(self, img):
        # use opencv good features to track
        features = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=3)
        keypoints = [cv2.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in features]
        descriptors = self.orb.compute(img, keypoints)
        # self.orb.compute()
        return keypoints, descriptors

fe = FeatureExtractor()

def process_frame(img):
    img = cv2.resize(img, (WIDTH, HEIGHT))
    keypoints, descriptors = fe.extract(img)

    for point in keypoints:
        u, v = map(lambda x: int(round(x)), point.pt)
        cv2.circle(img, (u, v), color=(0, 255, 0), radius=3)

    disp.paint(img)

if __name__ == "__main__":
    cap = cv2.VideoCapture("videos/test_countryroad.mp4")

    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            process_frame(frame)
        else:
            break