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


class SimilarWindow:
    def __init__(self, distance, place, image, rect=[0,0,0,0], time=0):
        """
        distances : float
        place : list[y, x] ウィンドウを表示させたい左上の座標
        rect : list[top, bottom, left, right] 解析対象の顔のフレーム
        image : array 画像データ
        """
        self.distance = distance + random.uniform(0, 0.00000001)
        self.place_y = place[0]
        self.place_x = place[1]
        self.rect_top = rect[0]
        self.rect_bottom = rect[1]
        self.rect_left = rect[2]
        self.rect_right = rect[3]
        self.image = image
        self.time = time

    def put_on_frame(self, frame, place):
        """
        frame : array カメラフレーム
        place : list [y, x] 位置座標の更新
        """
        self.place_y = place[0]
        self.place_x = place[1]
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]
        window_height = 218
        window_width = 178
        image = self.num_ride(self.image, self.distance)

        try:
            #後ほど見切れたときの処理を書く
            frame = self.exe_put(self.place_y, self.place_y+window_height, self.place_x, self.place_x+window_width, image, frame)
            return frame
        except:
            print("something is happened in put_on_frame")
            print("image", type(image))
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_traceback,
                              limit=2, file=sys.stdout)
            # 未処理のフレームを返す
            return frame

    def exe_put(self, top, bottom, left, right, image, frame):
        part_frame = frame[top:bottom, left:right]
        # to avoid error
        # image = cv2.resize(image, (int(image.shape[1]/2), int(image.shape[0]/2)), interpolation = cv2.INTER_CUBIC)
        t = self.time
        if t > 15:
            t = 15
        try:
            image = cv2.resize(image, (int(image.shape[1]/15 * t), int(image.shape[0]/15 * t)))
        except:
            image = cv2.resize(image, (part_frame.shape[1], part_frame.shape[0]), interpolation = cv2.INTER_CUBIC)
        print("part_frame shape", part_frame.shape[0], part_frame.shape[1])
        print("image shape", image.shape[0], image.shape[1])
        blended_image = cv2.addWeighted(part_frame, 0, image, 1, 0)
        frame[top:bottom, left:right] = blended_image
        # 時間経過
        self.time += 1
        return frame

    def num_ride(self, image, distance):
        try:
            image_num = image.copy()
        except:
            image_num = image
        distance = round(distance, self.time%25)
        #print("distance", distance)
        padding = 10
        try:
            font = cv2.FONT_HERSHEY_TRIPLEX
            cv2.putText(image_num, str(distance),(0, image.shape[0]-padding), font, 0.5,(255,255,255),2,cv2.FONT_HERSHEY_TRIPLEX)
        except:
            pass
        return image_num
