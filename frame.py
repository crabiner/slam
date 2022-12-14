#!/usr/bin/env python3
import cv2
import numpy as np
np.set_printoptions(suppress=True)

from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
from skimage.transform import EssentialMatrixTransform

# turn [[x,y]] -> [[x,y,1]]
def add_ones(x):
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

# IRt - Identity rotation translation - matrix of 4 rows and 4 columns
IRt = np.eye(4) # no rotation

def extractRt(F):
    W = np.mat([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    U, d, Vt = np.linalg.svd(F)
    assert np.linalg.det(U) > 0
    if np.linalg.det(Vt) < 0:
        Vt *= -1.0
    R = np.dot(np.dot(U, W), Vt)
    if np.sum(R.diagonal()) < 0:
        R = np.dot(np.dot(U, W.T), Vt)
    t = U[:, 2]
    ret = np.eye(4)

    # up to column 3
    ret[:3, :3] = R
    # last column (3)
    ret[:3, 3] = t

    # print(ret)

    # to calibrate F this vector should be close to [1 1 0]
    # print(d)
    return ret

def extract(img):
    # detection
    orb = cv2.ORB_create()

    # use opencv good features to track
    points = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8),
                                     500,
                                     qualityLevel=0.01,
                                     minDistance=7)

    # extraction
    keypoints = [cv2.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in points]
    keypoints, descriptors = orb.compute(img, keypoints)

    # return pts and des
    return np.array([(kp.pt[0], kp.pt[1]) for kp in keypoints]), descriptors


def normalize(Kinv, pts):
    return np.dot(Kinv, add_ones(pts).T).T[:, 0:2]

# use the projection matrix of the camera
def denormalize(K, pt):
    # https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
    # Take the homogenous coordinates and left multiply it by the K matrix
    ret = np.dot(K, np.array([pt[0], pt[1], 1.0]))
    ret /= ret[2]
    return int(round(ret[0])), int(round(ret[1]))



def match_frames(f1, f2):
    # matching

    # Brute-Force matcher is simple. It takes the descriptor of one feature in first set and is matched
    # with all other features in second set using some distance calculation. And the closest one is returned.
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    # match to previous frame using brute force matcher
    matches = bf.knnMatch(f1.des, f2.des, k=2)

    # Lowe's ratio test
    ret = []
    idx1, idx2 = [], []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            p1 = f1.kps[m.queryIdx]
            p2 = f2.kps[m.trainIdx]

            #print(m.distance)
            # travel less than 10% of frame diagonal and be within orb distance 32
            # np.linalg.norm((p1 - p2))  i s on pixel scale
            if np.linalg.norm((p1 - p2)) < 0.1 * np.linalg.norm([f1.w, f1.h]) and m.distance < 32:
                # keep around indices
                idx1.append(m.queryIdx)
                idx2.append(m.trainIdx)

                ret.append((p1, p2))

    assert len(ret) >= 8
    ret = np.array(ret)
    idx1 = np.array(idx1)
    idx2 = np.array(idx2)

    # fit matrix
    # When we have the F number (calibrated camera) we can use
    # EssentialMatrixTransform and it estimates less parameters
    model, inliers = ransac((ret[:, 0], ret[:, 1]),
                            # EssentialMatrixTransform,
                            FundamentalMatrixTransform,
                            min_samples=8,
                            residual_threshold=0.001,  # lower residual threshold to get less errors
                            max_trials=100)
    # print(f"sum(inliers) {sum(inliers)}, len(inliers) {len(inliers)}")
    print("Matches: %d -> %d -> %d -> %d" % (len(f1.des), len(matches), len(inliers), sum(inliers)))

    # ignore outliers
    # now we want just the inliers and not the noise
    ret = ret[inliers]

    # extract rotation and translation
    Rt = extractRt(model.params)

    # we only care about the matches
    return idx1[inliers], idx2[inliers], Rt


class Frame(object):
  def __init__(self, mapp, img, K):
    self.K = K
    self.Kinv = np.linalg.inv(self.K)
    self.pose = IRt
    self.h, self.w = img.shape[0:2]

    keypoints, self.des = extract(img)
    self.kps = normalize(self.Kinv, keypoints)

    #initialize with Nones of length keypoints
    self.pts = [None]*len(self.kps)

    self.id = len(mapp.frames)
    mapp.frames.append(self)