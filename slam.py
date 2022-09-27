#!/usr/bin/env python3
import cv2
from display import Display

WIDTH = 1920 // 2
HEIGHT = 1080 // 2
MAX_FEATURES = 100

disp = Display(WIDTH, HEIGHT)


# break the image into a grid to improve distribution of detections
class FeatureExtractor(object):
    # grid 16x16
    GRIDX = 16//2
    GRIDY = 12//2

    def __init__(self):
        self.orb = cv2.ORB_create(MAX_FEATURES)

    def extract(self, img):
        # run detect in grid//
        image_block_size_y = img.shape[0] // self.GRIDY
        image_block_size_x = img.shape[1] // self.GRIDX
        akp = []

        # iterate over the pixels in the grid
        for ry in range(0, img.shape[0], image_block_size_y):
            for rx in range(0, img.shape[1], image_block_size_x):
                img_chunk = img[ry: ry + image_block_size_y, rx: rx + image_block_size_x]
                print(img_chunk.shape)

                # do only the detection part on the chunk
                # detect finds good keypoints to track
                keypoints = self.orb.detect(img_chunk, None)
                for p in keypoints:
                    # add back rx and ry to return to img original coordinates
                    p.pt = (p.pt[0] + rx, p.pt[1] + ry)
                    akp.append(p)
        return akp

fe = FeatureExtractor()

def process_frame(img):
    img = cv2.resize(img, (WIDTH, HEIGHT))
    keypoints = fe.extract(img)

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