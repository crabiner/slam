#!/usr/bin/env python3
import cv2
from display import Display
import numpy as np
np.set_printoptions(suppress=True)

from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
from skimage.transform import EssentialMatrixTransform

MAX_FEATURES = 100

f_estimate_avg = []
# turn [[x,y]] -> [[x,y,1]]
def add_ones(x):
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

class Extractor(object):
    def __init__(self, K):
        self.orb = cv2.ORB_create(MAX_FEATURES)

        # Brute-Force matcher is simple. It takes the descriptor of one feature in first set and is matched
        # with all other features in second set using some distance calculation. And the closest one is returned.
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.last = None
        self.K = K
        self.Kinv = np.linalg.inv(self.K)

    def normalize(self, pts):
        return np.dot(self.Kinv, add_ones(pts).T).T[:, 0:2]

    # use the projection matrix of the camera
    def denormalize(self, pt):
        # https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
        # Take the homogenous coordinates and left multiply it by the K matrix
        ret = np.dot(self.K, np.array([pt[0], pt[1], 1.0]))
        return int(round(ret[0])), int(round(ret[1]))

    def extract(self, img):
        # detection
        # use opencv good features to track
        features = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8),
                                           3000,
                                            qualityLevel=0.01,
                                           minDistance=3)

        # extraction
        keypoints = [cv2.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in features]
        keypoints, descriptors = self.orb.compute(img, keypoints)

        # matching
        ret = []
        if self.last is not None:
            # match to previous frame using brute force matcher
            matches = self.bf.knnMatch(descriptors, self.last['descriptors'], k=2)
            for m, n in matches:
                if m.distance < 0.75*n.distance:
                    kp1 = keypoints[m.queryIdx].pt
                    kp2 = self.last['keypoints'][m.trainIdx].pt
                    ret.append((kp1, kp2))

        # filter
        if len(ret) > 0:
            ret = np.array(ret)

            # normalize coordinates: subtract to move to zero (lens image center)
            ret[:, 0, :] = self.normalize(ret[:, 0, :])
            ret[:, 1, :] = self.normalize(ret[:, 1, :])

            # When we have the F number (calibrated camera) we can use
            # EssentialMatrixTransform and it estimates less parameters
            model, inliers = ransac((ret[:, 0], ret[:, 1]),
                                        EssentialMatrixTransform,
                                        #FundamentalMatrixTransform,
                                        min_samples = 8,
                                        residual_threshold = 0.005, # lower residual threshold to get less errors
                                        max_trials = 100)

            # now we want just the inliers and not the noise
            ret = ret[inliers]

            W = np.mat([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
            # find out fx and fy assuming they are equal
            # v should be [1, 1, 0] see Hartley and zisserman chapter 6
            # Because it is a triangle with lines of [sqrt(2)/2,sqrt(2)/2, 1]
            # we can use svd to estimate rotation and translation
            u, w, vt = (np.linalg.svd(model.params))
            # print(w)
            assert np.linalg.det(u) > 0

            if np.linalg.det(vt) < 0:
                u *= -1.0

            # there two possible rotation matrices
            R = np.dot(np.dot(u, W), vt)
            if np.sum(R.diagonal()) < 0:
                R = np.dot(np.dot(u, W.T), vt)
            print(R)


        self.last = {'keypoint'
                     's': keypoints, 'descriptors': descriptors}

        # we only care about the matches
        return ret

