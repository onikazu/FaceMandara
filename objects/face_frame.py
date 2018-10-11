# ライブラリインポート
from multiprocessing import Process, Manager, Value
import pickle
import math
import time
import sys
import traceback
import random

import cv2
import face_recognition
import dlib
import numpy as np
import faiss


class FaceFrame:
    def __init__(self, top, bottom, left, right):
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right

    def cal_center(self):
        y = self.top + (self.bottom - self.top) / 2
        x = self.left + (self.right - self.left) / 2
        center = [y, x]
        return center

    def draw_frame(self, frame):
        frame = cv2.rectangle(frame, (self.left, self.top), (self.right, self.bottom), (0, 255, 3), 3)
        return frame

    def setter(self, top, bottom, left, right):
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right
