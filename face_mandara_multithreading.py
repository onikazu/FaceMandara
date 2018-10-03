"""
共有する変数、配列はframe, im0~7
"""

from multiprocessing import Process, Array
import pickle
import math
import time

import cv2
import face_recognition
import dlib
import numpy as np

# database、共有メモリ変数の読み込み
datas = {}
with open('mini_data.pickle', mode='rb') as f:
    datas = pickle.load(f)

similar_paths_memory = Array("c", range(10))
frame_memory = Array("i", range(720))


def get_distance(a, b):
    """
    画像間の類似度を測定する
    :param a: list
    :param b: list
    :return: float
    """
    distance = 0
    for i in range(len(a)):
        distance += (a[i] - b[i]) ** 2
    return math.sqrt(distance)


def concat_tile(im_list_2d):
    """
    イメージをタイル状に敷き詰める
    :param im_list_2d: list(2d)
    :return:
    """
    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])


def recommend_faces():
    """
    カメラ映像から取得した人物の類似顔を探し出す関数
    """
    while True:
        frame = frame_memory
        print(frame_memory)
        # 顔認識
        detector = dlib.get_frontal_face_detector()

        # 顔データ(人数分)
        rects = detector(frame, 1)

        # 顔認識できなかったとき
        if not rects:
            print("cant recognize faces")
            continue

        dsts = []
        for rect in rects:
            dst = frame[rect.top():rect.bottom(), rect.left():rect.right()]
            dsts.append(dst)

        # 距離測定（とりあえず一人だけ）
        # 顔情報のベクトル化　類似配列の生成
        try:
            target_image_encoded = face_recognition.face_encodings(dsts[0])[0]
        except IndexError:
            continue

        similar_vecs = []
        similar_paths = []
        similar_distances = []

        i = 0
        for k in datas:
            distance = get_distance(datas[k], list(target_image_encoded))
            # 最初
            if i == 0:
                similar_distances.append(distance)
                similar_paths.append(k)
                similar_vecs.append(datas[k])
                i += 1
            for j in range(len(similar_distances)):
                # 10個以上
                if len(similar_distances) >= 10:
                    # より近い
                    if similar_distances[j] > distance:
                        similar_distances.insert(j, distance)
                        similar_paths.insert(j, k)
                        similar_vecs.insert(j, datas[k])
                        del similar_distances[-1]
                        del similar_paths[-1]
                        del similar_vecs[-1]
                        break
                # 10個以下
                else:
                    if similar_distances[j] > distance:
                        similar_distances.insert(j, distance)
                        similar_paths.insert(j, k)
                        similar_vecs.insert(j, datas[k])
                        break
                    if j == len(similar_distances) - 1:
                        similar_distances.append(distance)
                        similar_paths.append(k)
                        similar_vecs.append(datas[k])

            # print("{0}:{1}".format(k, distance))
            # print("number{} is end".format(i))
            i += 1
        print("finish about one face")
        similar_paths_memory = similar_paths

def take_video():
    """
    入力データを生成する関数
    """
    cap = cv2.VideoCapture(0)  # 引数はカメラのデバイス番号
    while True:
        ret, frame = cap.read()
        # to break the loop by pressing esc
        frame_memory = frame
        k = cv2.waitKey(1)

        if k == 27:
            print("released!")
            break
    cap.release()
    cv2.destroyAllWindows()
    print("release camera!!!")


# 実行部分.画面の生成
def main():
    # プロセスの生成
    video_process = Process(target=take_video, name="video")
    recommend_process = Process(target=recommend_faces, name="recommend")

    # プロセスの開始
    video_process.start()
    recommend_process.start()

    while True:
        if not similar_paths_memory or frame_memory:
            continue

        # 初期状態でないかつ新たにデータを受け取った
        im0 = cv2.imread("./database/{}".format(similar_paths_memory[0]))
        im1 = cv2.imread("./database/{}".format(similar_paths_memory[1]))
        im2 = cv2.imread("./database/{}".format(similar_paths_memory[2]))
        im3 = cv2.imread("./database/{}".format(similar_paths_memory[3]))
        im4 = cv2.imread("./database/{}".format(similar_paths_memory[4]))
        im5 = cv2.imread("./database/{}".format(similar_paths_memory[5]))
        im6 = cv2.imread("./database/{}".format(similar_paths_memory[6]))
        im7 = cv2.imread("./database/{}".format(similar_paths_memory[7]))
        frame = cv2.resize(frame_memory, (178, 218))

        im_tile = concat_tile([[im0, im1, im2],
                               [im3, frame, im4],
                               [im5, im6, im7]])
        cv2.imshow('tile camera', im_tile)

        # to break the loop by pressing esc
        k = cv2.waitKey(1)
        if k == 27:
            print("released!")
            break


if __name__ == "__main__":
    main()



