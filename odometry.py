import cv2
import os
import numpy as np
import cvlib as cv
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform, EssentialMatrixTransform

K = np.array(([140, 0, 640//2], [0, 140, 480//2], [0, 0, 1]))

def add_ones(x):
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)


def getCars(img):
    bbox, _, _ = cv.detect_common_objects(frame, confidence=0.15, model='yolov3-tiny')
    return bbox


def mask(W, H, offset=25, mask=None, bbox=None):
    if mask is None:
        mask = np.zeros(shape = (H, W), dtype = np.uint8)
        mask.fill(255)
    else:
        W = mask.shape[1]
        H = mask.shape[0]

    cv2.rectangle(mask, (0, 0), (W, 3*H//4), (0, 0, 0), -1)
    if bbox is not None:
        for box in bbox:
            car = np.array([[[box[0]-offset, box[1]-offset], [box[0]-offset, box[3]+offset],\
                             [box[2]+offset, box[3]+offset], [box[2]+offset, box[1]-offset]]], dtype=np.int32)
            cv2.fillPoly(mask, car, (255, 255, 255))

    return mask


class VisualOdometer:
    def __init__(self, K):
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.last = None
        self.K = K
        self.Kinv = np.linalg.inv(self.K)

    
    def normalize(self, pts):
        return np.dot(self.Kinv, add_ones(pts).T).T[:, 0:2]
    

    def denormalize(self, pt):
        ret = np.dot(self.K, [pt[0], pt[1], 1.0])
        return int(round(ret[0])), int(round(ret[1]))


    def extractRt(self, E):
        W = np.mat([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        U, d, Vt = np.linalg.svd(E)
        if np.linalg.det(U) < 0:
            U *= -1.0
        if np.linalg.det(Vt) < 0:
            Vt *= -1.0

        R = np.dot(np.dot(U, W), Vt)
        if np.sum(R.diagonal()) < 0:
            R = np.dot(np.dot(U, W.T), Vt)
        t = U[:, 2]

        Rt = np.concatenate([R, t.reshape(3, 1)], axis=1)
        return Rt


    def extract(self, img, mask=None):
        # detection 
        pts = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 3000, 
                                      qualityLevel=0.01, 
                                      minDistance=7, 
                                      mask=mask)

        # extraction
        kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in pts]
        kps, des = self.orb.compute(img, kps)

        # matching
        ret = []
        if self.last is not None:
            matches = self.bf.knnMatch(des, self.last['des'], k=2)
            for m, n in matches:
                if m.distance < .75*n.distance:
                    kp1 = kps[m.queryIdx].pt
                    kp2 = self.last['kps'][m.trainIdx].pt
                    ret.append((kp1, kp2))

        Rt = None
        if len(ret) > 0:
            ret = np.array(ret)
            # normalize coords
            ret[:, 0, :] = self.normalize(ret[:, 0, :])
            ret[:, 1, :] = self.normalize(ret[:, 1, :])

            # filter
            model, inliers = ransac((ret[:, 0], ret[:, 1]),
                                    EssentialMatrixTransform,
                                    min_samples=8, 
                                    residual_threshold=0.008, 
                                    max_trials=1000)
            ret = ret[inliers]
            
            # extract camera pose
            # print(model.params)
            Rt = self.extractRt(model.params)

        # store previous points
        self.last = {'kps': kps, 'des': des}
        return ret, Rt


def process_frame(img, mask=None, speed=None, prev_speed=None):
    img = cv2.resize(img, (W, H))
    matches, Rt = vo.extract(img, mask=mask)

    # display keypoints
    if matches is not None:
        for pt1, pt2 in matches:
            a, b = vo.denormalize(pt1)
            c, d = vo.denormalize(pt2)
            cv2.circle(img, (a, b), color=(0, 255, 0), radius=3)
            cv2.line(img, (a, b), (c, d), color=(255, 0, 0), thickness=1)

    if speed is not None:
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(img, "Speed: {}".format(speed), (10, 35), font, 1.2, (0, 255, 0))

    cv2.imshow("", img)


if __name__ == '__main__':
    cap = cv2.VideoCapture('test.mp4')

    # get width and height
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    CNT = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # instrinsic camera properties
    F = 140
    if os.getenv("SEEK") is not None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(os.getenv("SEEK")))

    K = np.array(([F, 0, W//2], [0, F, H//2], [0, 0, 1]))
    vo = VisualOdometer(K)

    '''
    speeds = np.zeros((20400,), dtype=float)
    train = open('train.txt', 'r')
    for i, x in enumerate(train.readlines()):
        speeds[i] = float(x.strip())
    '''

    i = 1
    while cap.isOpened() and i <= 20400:
        ret, frame = cap.read()
        bbox = getCars(frame)
        mask_vis = mask(W, H, bbox=bbox)
        mask_vis = cv2.bitwise_not(mask_vis)
        if ret:
            process_frame(frame, mask_vis)
        if cv2.waitKey(1000//48) & 0xFF == ord('q'):
            break
        i += 1