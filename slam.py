#!/usr/bin/env python3
import cv2
from display import Display
import numpy as np
from frame import Frame, denormalize, match_frames, IRt
import g2o

# camera intrinsics
WIDTH = 1920 // 2
HEIGHT = 1080 // 2

cx = WIDTH // 2
cy = HEIGHT // 2
F = 270
K = np.array([[F, 0, cx], [0, F, cy], [0, 0, 1]])
# principal point that is usually at the image center


# global map
class Map(object):
    def __init__(self):
        self.frames = []
        self.points = []

    def display_map(self):
        for f in self.frames:
            print(f.id)
            print(f.pose)
        # for p in points:
        #     print(p.xyz)

# main classes
disp = Display(WIDTH, HEIGHT)
mapp = Map()

class Point(object):
    # A point is a 3-D point in the world
    # Each point is observed in multiple frames
    def __init__(self, mapp, loc):
        self.frames = []
        self.xyz = loc
        self.idxs = []
        self.id = len(mapp.points)
        mapp.points.append(self)

    # idx is index of which descriptor it is in the frame
    def add_observation(self, frame, idx):
        self.frames.append(frame)
        self.idxs.append(idx)
def triangulate(pose1, pose2, pts1, pts2):
    return  cv2.triangulatePoints(pose1[:3], pose2[:3], pts1.T, pts2.T).T


def process_frame(img):
    img = cv2.resize(img, (WIDTH, HEIGHT))

    frame = Frame(mapp, img, K)
    if frame.id == 0:
        return

    f1 = mapp.frames[-1]
    f2 = mapp.frames[-2]

    idx1, idx2, Rt = match_frames(f1, f2)

    # triangolate between the first and second image
    # pts4d is in homogenous coordinates
    f1.pose = np.dot(Rt, f2.pose)

    pts4d = triangulate(f1.pose, f2.pose, f1.points[idx1], f2.points[idx2])
    # homogenous 3D coords
    pts4d /= pts4d[:, 3:]

    # X is right, y is down, z is forward
    # reject points behind the camera
    # reject pts without enough parallax
    good_pts4d = (np.abs(pts4d[:, 3])> 0.005) & (pts4d[:, 2] > 0)
    print(f"sum(good_pts4d) {sum(good_pts4d)}, len(good_pts4d) {len(good_pts4d)}")
    pts4d = pts4d[good_pts4d]

    for i,p in enumerate(pts4d):
        if not good_pts4d[i]:
            continue
        pt = Point(mapp, p)
        pt.add_observation(f1, idx1[i])
        pt.add_observation(f2, idx2[i])

    # print(pts4d)
    # print("%d matches" % (len(matches)))


    for pt1, pt2 in zip(f1.points[idx1], f2.points[idx2]):
        u1, v1 = denormalize(K, pt1)
        u2, v2 = denormalize(K, pt2)

        cv2.circle(img, (u1, v1), color=(0, 255, 0), radius=3)
        cv2.line(img, (u1, v1), (u2, v2), color=(255, 0, 0))

    disp.paint(img)

    # 3D
    mapp.display_map()


if __name__ == "__main__":
    cap = cv2.VideoCapture("videos/test_countryroad.mp4")

    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            process_frame(frame)
        else:
            break
