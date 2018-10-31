# ライブラリインポート
import multiprocessing as mp
from multiprocessing import Process, Manager, Value
import pickle
import math
import time
import sys
import traceback
import random

from PIL import Image, ImageDraw
import cv2
import face_recognition
import dlib
import numpy as np
import faiss

import easing
from objects import face_frame_v2
from objects import similar_window_v9_7_ubuntu, line_v3

# databaseの読み込み
print("start indexing")
datas = {}
with open('big_data.pickle', mode='rb') as f:
    datas = pickle.load(f)
# databese配列の作成
face_image_names = []
face_vectors = []
for k in datas:
    face_image_names.append(k)
    face_vectors.append(datas[k])
face_vectors = np.array(face_vectors).astype("float32")

# faissを用いたPQ
nlist = 100
m = 8
k = 8  # 類似顔7こほしいのでk=8
d = 128  # 顔特徴ベクトルの次元数
quantizer = faiss.IndexFlatL2(d)  # this remains the same
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
index.train(face_vectors)
index.add(face_vectors)
print("indexing is end")

with open('index.pickle', mode='wb') as f:
    pickle.dump(index, f)
