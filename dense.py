import cv2
import os
import numpy as np

W = 640
H = 480

def process_frame(img, prev):
    # dense optical flow
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # cv2.rectangle(gray, (0, 0), (W, H//2), (0, 0, 0), -1)
    flow = cv2.calcOpticalFlowFarneback(prev, gray, None, 0.5, 3, 17, 3, 5, 1.2, 0)

    # get magnitude and angle of dense flow vectors and use these for hsv parameters
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mask[..., 0] = angle * 180 / np.pi / 2 # direction 
    mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX) # magnitude

    # convert hsv parameters to rgb and display
    rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
    dense = cv2.addWeighted(img, 1, rgb, 2, 0)
    cv2.imshow("Dense Optical Flow", dense)

    return gray


if __name__ == "__main__":
    cap = cv2.VideoCapture('train.mp4')
    ret, first = cap.read()
    prev = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(first)
    mask[..., 1] = 255

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            prev = process_frame(frame, prev)
        if cv2.waitKey(1000//1000) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()