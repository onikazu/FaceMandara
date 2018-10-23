"""
移動量属性を加えた
"""

# ライブラリインポート
from multiprocessing import Process, Manager, Value
import pickle
import math
import time
import sys
import traceback
import random

from PIL import Image, ImageDraw, ImageFont
import cv2
import face_recognition
import dlib
import numpy as np
import faiss


class SimilarWindow:
    def __init__(self, distance, place, image, movement_amount, rect=[0,0,0,0], time=0, similar_num=0):
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
        # 似ているほど大きい数字
        self.similar_num = similar_num
        self.image_frame = Image.open("./objects/mono_frame.png")
        self.movement_amount_x, self.movement_amount_y = movement_amount

    def put_on_frame(self, frame, place):
        """
        frame : array カメラフレーム
        place : list [y, x] 位置座標の更新
        """
        self.place_y = place[0]
        self.place_x = place[1]
        frame_height = frame.height
        frame_width = frame.width
        window_height = 218
        window_width = 178
        image = self._num_ride(self.image, self.distance)

        try:
            #後ほど見切れたときの処理を書く
            frame = self._exe_image_put(self.place_x, self.place_y, image, frame)
            return frame
        except:
            print("something is happened in put_on_frame")
            print("image", type(image))
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_traceback,
                              limit=2, file=sys.stdout)
            # 未処理のフレームを返す
            return frame

    def _exe_image_put(self, x, y, image, frame):
        window_height = 218
        window_width = 178
        # 画像の加工
        print("frame", type(frame))
        print("image", type(image))
        try:
            window_width = int(window_width * (self.similar_num/15))
            window_height = int(window_height * (self.similar_num/15))

            print("window_width", window_width)
            print("window_height", window_height)
            image = image.resize((window_width, window_height))
            print("self.time:", self.time)
            # 代入してやらないと動かない
            self.image_frame = self.image_frame.rotate(10)
            # PIL.ImageDraw.Draw.ellipse(xy, fill=None, outline=None)
            # imageを(重要)くり抜くmask
            # きれいな円でくり抜く
            padding = (window_height-window_width)/2
            self.image_frame = self.image_frame.resize((int(window_width), int(window_height-2*padding)))
            image.paste(self.image_frame, (0, int(padding), int(window_width), int(window_height-padding)), self.image_frame.split()[3])

            mask_im = Image.new("L", image.size, 0)
            draw = ImageDraw.Draw(mask_im)
            draw.ellipse((0, padding, window_width, window_height-padding), fill=255)
            radius = int(window_width/2)
            # 類似度に応じて拡大、縮小
            # window_size = (int(2*window_width*(self.similar_num/8)), int(2*window_width*(self.similar_num/8)))
            # image = image.resize(window_size)

            frame.paste(image, (x-radius, y-radius, x-radius+window_width, y-radius+window_height), mask_im)
            print("im in exe put try")
        except:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_traceback,
                              limit=2, file=sys.stdout)
            print("something happened in _exe_put")

        # 時間経過
        self.time += 1
        return frame

    def _num_ride(self, image, distance):
        try:
            image_num = image.copy()
        except:
            image_num = image
        # ドロワー
        draw = ImageDraw.Draw(image_num)
        distance_text = round(distance, self.time%25)
        w = image.width
        h = image.height
        padding = 10
        # PIL.ImageFont.truetype(font=None, size=10, index=0, encoding='')
        font = ImageFont.truetype("arial.ttf", 17)
        print("in _num_ride")
        try:
            print("im in try!")
            print("h:", type(h), h)
            print("distance_text: ", type(distance_text), distance_text)
            draw.text((0, h-17), str(distance_text), font=font, fill=(255,0,0,128))
        except:
            print("im in except")
            print("h:", type(h), h)
            print("distance_text: ", type(distance_text), distance_text)
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_traceback,
                              limit=2, file=sys.stdout)
            print("something happened in _num_ride")

        return image_num
