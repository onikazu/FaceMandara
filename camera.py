import cv2
from multiprocessing import Process, Pipe
import numpy as np


def camera():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        # print(len(frame))
        cv2.imshow("window", frame)
        print(frame.shape)

        # conn.send(frame)
        k = cv2.waitKey(1)

        if k == 27:
            print("released!")
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    camera()
