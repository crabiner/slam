#!/usr/bin/env python3
import cv2
from display import Display
import numpy as np
from frame import Frame, denormalize, match_frames, IRt
import g2o
import OpenGL.GL as gl
import pangolin

from multiprocessing import Process, Queue

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
        self.scam = None
        self.state = None
        self.frames = []
        self.points = []
        self.q = Queue()

        p = Process(target=self.viewer_thread, args=(self.q,))
        p.daemon = True
        p.start()

    def viewer_thread(self, q):
        self.viewer_init()
        while 1:
            self.viewer_refresh(q)

    def viewer_init(self):
        pangolin.CreateWindowAndBind('Main', 640, 480)
        gl.glEnable(gl.GL_DEPTH_TEST)

        # Define Projection and initial ModelView matrix
        self.scam = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 100),
            pangolin.ModelViewLookAt(-2, 2, -2, 0, 0, 0, pangolin.AxisDirection.AxisY))
        self.handler = pangolin.Handler3D(self.scam)

        # Create Interactive View in window
        self.dcam = pangolin.CreateDisplay()
        self.dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -640.0 / 480.0)
        self.dcam.SetHandler(self.handler)

    def viewer_refresh(self, q):
        if self.state is None or not q.empty():
            self.state = q.get()

        # trun state into points
        ppts = np.array([d[:3, 3] for d in self.state[0]])
        spts = np.array(self.state[1])

        # while not pangolin.ShouldQuit():
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        self.dcam.Activate(self.scam)

        # big green points for the poses
        gl.glPointSize(10)
        gl.glColor3f(1.0, 0.0, 0.0)
        pangolin.DrawPoints(ppts)

        # little red points
        gl.glPointSize(2)
        gl.glColor3f(0.0, 1.0, 0.0)
        pangolin.DrawPoints(spts)

        pangolin.FinishFrame()

    def display(self):
        poses, pts = [], []
        for f in self.frames:
            poses.append(f.pose)
            # print(f.id)
            # print(f.pose)
        for p in self.points:
            pts.append(p.xyz)
            print(p.xyz)
        self.q.put((poses, pts))


# main classes
# disp = Display(WIDTH, HEIGHT)
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
    return cv2.triangulatePoints(pose1[:3], pose2[:3], pts1.T, pts2.T).T


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
    good_pts4d = (np.abs(pts4d[:, 3]) > 0.005) & (pts4d[:, 2] > 0)
    print(f"sum(good_pts4d) {sum(good_pts4d)}, len(good_pts4d) {len(good_pts4d)}")
    pts4d = pts4d[good_pts4d]

    for i, p in enumerate(pts4d):
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

    # disp.paint(img)
    #
    # 3D
    mapp.display()


if __name__ == "__main__":
    cap = cv2.VideoCapture("videos/test_countryroad.mp4")

    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            process_frame(frame)
        else:
            break
