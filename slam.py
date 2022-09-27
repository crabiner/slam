#!/usr/bin/env python3
import cv2
from display import Display
import numpy as np

WIDTH = 1920 // 2
HEIGHT = 1080 // 2
MAX_FEATURES = 100

disp = Display(WIDTH, HEIGHT)


class Extractor(object):
    def __init__(self):
        self.orb = cv2.ORB_create(MAX_FEATURES)

        # Brute-Force matcher is simple. It takes the descriptor of one feature in first set and is matched
        # with all other features in second set using some distance calculation. And the closest one is returned.
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.last = None

    def extract(self, img):
        # detection
        # use opencv good features to track
        features = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 3000, qualityLevel=0.01,
                                           minDistance=3)

        # extraction
        keypoints = [cv2.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in features]
        keypoints, descriptors = self.orb.compute(img, keypoints)

        # matching
        # match to previous frame using brute force matcher
        matches = None
        if self.last is not None:
            matches = self.bf.match(descriptors, self.last['descriptors'])

        self.last = {'keypoints': keypoints, 'descriptors': descriptors}

        # self.orb.compute()
        return keypoints, descriptors, matches


fe = Extractor()


def process_frame(img):
    img = cv2.resize(img, (WIDTH, HEIGHT))
    keypoints, descriptors, _ = fe.extract(img)

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
