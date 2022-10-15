#!/usr/bin/env python3
import os
import sys
import cv2
from display import Display
import numpy as np
from frame import Frame, denormalize, match_frames, IRt
import g2o
from pointmap import Map, Point

# set this!
F = int(os.getenv("F", "800"))

# camera intrinsics
WIDTH = 1920 // 2
HEIGHT = 1080 // 2

cx = WIDTH // 2
cy = HEIGHT // 2
K = np.array([[F, 0, cx], [0, F, cy], [0, 0, 1]])
Kinv = np.linalg.inv(K)

# main classes
mapp = Map()
mapp.create_viewer() if os.getenv("D3D") is not None else None
disp = Display(WIDTH, HEIGHT) if os.getenv("D2D") is not None else None


# trianglulate taken from orb-slam 2
def triangulate(pose1, pose2, pts1, pts2):
    # return cv2.triangulatePoints(pose1[:3], pose2[:3], pts1.T, pts2.T).T
    ret = np.zeros((pts1.shape[0], 4))
    pose1 = np.linalg.inv(pose1)
    pose2 = np.linalg.inv(pose2)
    for i, p in enumerate(zip(pts1, pts2)):
        A = np.zeros((4,4))
        A[0] = p[1][0] * pose1[2] - pose1[0]
        A[1] = p[1][1] * pose1[2] - pose1[1]
        A[2] = p[0][0] * pose2[2] - pose2[0]
        A[3] = p[0][1] * pose2[2] - pose2[1]

        _, _, vt = np.linalg.svd(A)
        ret[i] = vt[3]
    return ret


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

    pts4d = triangulate(f1.pose, f2.pose, f1.kps[idx1], f2.kps[idx2])
    # homogenous 3D coords
    pts4d /= pts4d[:, 3:]

    # X is right, y is down, z is forward
    # reject points behind the camera
    # reject pts without enough parallax
    unmatched_points = np.array([f1.pts[i] is None for i in idx1]).astype(np.bool)

    print(np.all(unmatched_points))
    good_pts4d = (np.abs(pts4d[:, 3]) > 0.005) & (pts4d[:, 2] > 0) & unmatched_points
    #  f"sum(good_pts4d) {sum(good_pts4d)}, len(good_pts4d) {len(good_pts4d)}")
    pts4d = pts4d[good_pts4d]

    for i, p in enumerate(pts4d):
        if not good_pts4d[i]:
            continue
        pt = Point(mapp, p)
        pt.add_observation(f1, idx1[i])
        pt.add_observation(f2, idx2[i])

    # print(pts4d)
    # print("%d matches" % (len(matches)))

    for pt1, pt2 in zip(f1.kps[idx1], f2.kps[idx2]):
        u1, v1 = denormalize(K, pt1)
        u2, v2 = denormalize(K, pt2)

        cv2.circle(img, (u1, v1), color=(0, 255, 0), radius=3)
        cv2.line(img, (u1, v1), (u2, v2), color=(255, 0, 0))

    # 2D
    if disp is not None:
        disp.paint(img)

    # 3D
    mapp.display()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("%s <video.mp4>" % sys.argv[0])
        exit(-1)

    cap = cv2.VideoCapture(sys.argv[1])

    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            process_frame(frame)
        else:
            break
