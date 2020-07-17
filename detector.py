import numpy as np
import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox


def visualize(img, boxes):
    for box in boxes:
        cv2.rectangle(img, (box[0] - 15, box[1] - 15), (box[2] + 15, box[3] + 15), (0, 255, 0))
    cv2.imshow("Car Detection", img)


def main():
    cap = cv2.VideoCapture('train.mp4')
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            bbox, label, conf = cv.detect_common_objects(frame, confidence=0.15, model='yolov3-tiny')
            visualize(frame, bbox)
        else:
            break

        if cv2.waitKey(1000//500) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()