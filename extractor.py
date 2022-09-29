#!/usr/bin/env python3
import cv2
from display import Display
import numpy as np
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
from skimage.transform import EssentialMatrixTransform

MAX_FEATURES = 100

class Extractor(object):
    def __init__(self, w, h):
        self.orb = cv2.ORB_create(MAX_FEATURES)

        # Brute-Force matcher is simple. It takes the descriptor of one feature in first set and is matched
        # with all other features in second set using some distance calculation. And the closest one is returned.
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.last = None
        self.w = w
        self.h = h

    def denormalize(self, pt):
        # denormalize for display (add frame center)
        return int(round(pt[0] + self.h/2)), int(round(pt[1] + self.w/2))

    def extract(self, img):
        # detection
        # use opencv good features to track
        features = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 3000, qualityLevel=0.01,
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
            ret[:, :, 0] -= img.shape[0] // 2
            ret[:, :, 1] -= img.shape[1] // 2

            model, inliers = ransac((ret[:, 0], ret[:, 1]),
                                        #EssentialMatrixTransform,
                                        FundamentalMatrixTransform,
                                        min_samples = 8,
                                        residual_threshold = 1,
                                        max_trials = 100)

            # now we want just the inliers and not the noise
            ret = ret[inliers]

            s, v, d = (np.linalg.svd(model.params))
            print(v)

        self.last = {'keypoint'
                     's': keypoints, 'descriptors': descriptors}

        # we only care about the matches
        return ret

