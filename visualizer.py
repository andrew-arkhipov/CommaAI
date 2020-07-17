import numpy as np
import cv2
import sys
import os
import argparse
import json
import shutil
import math


class Visualizer:

    X_TOP_OFFSET = 250
    X_BTM_OFFSET = 65
    BLUE = (255, 0, 0)
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    MASK_X_OFFSET = 35
    MASK_Y_OFFSET = 130


    def __init__(self, video):
        self.video = cv2.VideoCapture(video)
        self.clahe = cv2.createCLAHE(clipLimit=100.0, tileGridSize=(8, 8))
        self.length = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.speed = []
        f = open('train.txt', 'r')
        for line in f.readlines():
            self.speed.append(line.strip())

        self.dists = np.zeros((self.length, 4))


    def transform(self, arr):
        return arr[130:350, 35:605]

    
    def impute(self, arr):
        mask = np.isnan(arr)
        idx = np.where(~mask, np.arange(mask.shape[1]), 0)
        np.maximum.accumulate(idx, axis=1, out=idx)
        arr[mask] = arr[np.nonzero(mask)[0], idx[mask]]

    
    def average(self, dist):
        if self.curr_frame < 99:
            self.curr_frame += 1

        if self.curr_frame == 99:
            self.dists = np.roll(self.dists, -1)

        self.dists[self.curr_frame] = dist
        self.avg = np.mean(self.dists)

    
    def mask(self, mask=None, factor=1):
        if mask is None:
            W = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
            H = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            mask = np.zeros(shape = (H, W), dtype = np.uint8)
            mask.fill(255)
        else:
            W = mask.shape[1]
            H = mask.shape[0]

        cv2.rectangle(mask, (0, 0), (W, H), (0, 0, 0), -1)
        polygon = np.array([[[660-self.X_TOP_OFFSET+10, 260], [self.X_TOP_OFFSET-30, 260], [self.X_BTM_OFFSET, 350], \
                             [self.X_BTM_OFFSET+100, 350], [self.X_TOP_OFFSET, 290], [660-self.X_TOP_OFFSET-30, 290], \
                             [650-self.X_BTM_OFFSET-110, 350], [650-self.X_BTM_OFFSET, 350]]], dtype=np.int32)
        cv2.fillPoly(mask, polygon, (255, 255, 255))
        
        return mask

    
    def apply_brightness_contrast(self, frame, brightness=0, contrast=0):
        if brightness != 0:
            if brightness > 0:
                shadow = brightness
                highlight = 255
            else:
                shadow = 0
                highlight = 255 + brightness
            alpha_b = (highlight - shadow)/255
            gamma_b = shadow
            frame = cv2.addWeighted(frame, alpha_b, frame, 0, gamma_b)

        if contrast != 0:
            f = 131*(contrast + 127)/(127*(131-contrast))
            alpha_c = f
            gamma_c = 127*(1-f)
            frame = cv2.addWeighted(frame, alpha_c, frame, 0, gamma_c)

        return frame

    
    def get_features(self, prev, mask):
        feature_params = {'maxCorners': 4, 'qualityLevel': 0.25, 'minDistance': 10, 'blockSize': 7}
        return cv2.goodFeaturesToTrack(prev, mask=mask, **feature_params) 

    
    def optflow(self, prev, curr, prev_kps):
        lk_params = {'winSize': (15, 15), 'maxLevel': 5, 'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)}
        nxt, status, _ = cv2.calcOpticalFlowPyrLK(prev, curr, prev_kps, None, **lk_params)
        good_old = prev_kps[status == 1]
        good_new = nxt[status == 1]

        return good_old, good_new

    
    def visualize(self, frame, mask, prev_kps, index, old_pts=None, new_pts=None, speed=None):
        self.mask(mask)                 # making sure to use correct dimensions    
        mask = cv2.bitwise_not(mask)    # inverting mask
        frame_vis = cv2.addWeighted(mask, 0.3, frame, 1, 1)

        if old_pts is not None:
            i = 0
            for old_point, new_point, in zip(old_pts, new_pts):
                a, b = new_point.ravel()
                a, b = int(a + self.MASK_X_OFFSET), int(b + self.MASK_Y_OFFSET)
                c, d = old_point.ravel()
                c, d = int(c + self.MASK_X_OFFSET), int(d + self.MASK_Y_OFFSET)

                frame_vis = cv2.circle(frame_vis, (a, b), 3, self.GREEN, 1)
                frame_vis = cv2.line(frame_vis, (a, b), (c, d), self.RED, 2)

                dist = np.sqrt((d-b)**2 + (c-a)**2)
                self.dists[index, i] = dist if dist else np.nan
                i += 1

        if speed is not None:
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame_vis, "Speed: {}".format(speed), (10, 35), font, 1.2, self.GREEN)

        cv2.imshow('', frame_vis)

    
    def get_keypoints(self, kps, offset_x=0, offset_y=0):
        if kps is None:
            return None
        return [cv2.KeyPoint(x=p[0][0] + offset_x, y=p[0][1] + offset_y, _size = 10) for p in kps]


    def player(self, fps):
        _, first = self.video.read()
        mask = self.mask()
        mask_vis = first.copy()         # getting arbitrary frame for mask creation for the visualization
        mask = self.transform(mask)     # creating mask for keypoint generation

        prev = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
        prev = self.apply_brightness_contrast(prev, 10, 35)
        prev = self.transform(prev)
        prev_kps = self.get_features(prev, mask)

        i = 1
        while self.video.isOpened() and i < self.length:
            ret, frame = self.video.read()
            if ret:
                frame = self.apply_brightness_contrast(frame, 10, 35)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)      # black and white
                gray = self.transform(gray)

                if prev_kps is not None:
                    old, new = self.optflow(prev, gray, prev_kps)   # getting optical flow keypoints for VO
                else:
                    old, new = None, None

                self.visualize(frame, mask_vis, prev_kps, i, old, new, self.speed[i])
                
                if cv2.waitKey(1000//fps) & 0xFF == ord('q'):
                    break

            prev = gray.copy()
            prev_kps = self.get_features(gray, mask)
            i += 1

        self.video.release()
        cv2.destroyAllWindows()


    def get_preds(self):
        return self.dists


if __name__ == '__main__':
    viz = Visualizer('train.mp4')
    viz.player(24)
    preds = viz.get_preds()
    viz.impute(preds)
    np.savetxt('features.txt', preds)