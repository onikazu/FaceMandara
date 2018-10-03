import cv2
import face_recognition
import pickle
import math
import dlib
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob
import numpy as np

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

def main():
    cap = cv2.VideoCapture(0)  # 1はカメラのデバイス番号
    # データベース顔写真の読み込み
    target_image_paths = glob.glob("./target_face/face*.jpg")

    # databaseの読み込み
    with open('mini_data.pickle', mode='rb') as f:
        datas = pickle.load(f)

    # 初回の読み込みが完了したかどうか
    recog_flag = False

    im_tile = np.ndarray([])
    im0 = np.ndarray([])
    im1 = np.ndarray([])
    im2 = np.ndarray([])
    im3 = np.ndarray([])
    im4 = np.ndarray([])
    im5 = np.ndarray([])
    im6 = np.ndarray([])
    im7 = np.ndarray([])
    im8 = np.ndarray([])
    im9 = np.ndarray([])

    while True:
        ret, frame = cap.read()

        # 顔認識
        detector = dlib.get_frontal_face_detector()

        # 顔データ(人数分)
        rects = detector(frame, 1)

        # 顔認識できなかったとき
        if not rects:
            print("cant recognize faces")
            # 認識済みなら
            if recog_flag:
                print("in the if block")
                frame = cv2.resize(frame, (178, 218))
                im_tile = concat_tile([[im0, im1, im2],
                                       [im3, frame, im4],
                                       [im5, im6, im7]])

                cv2.imshow('tile camera', im_tile)
            continue

        dsts = []
        for rect in rects:
            dst = frame[rect.top():rect.bottom(), rect.left():rect.right()]
            dsts.append(dst)

        # 距離測定（とりあえず一人だけ）
        ## 顔情報のベクトル化　類似配列の生成
        try:
            target_image_encoded = face_recognition.face_encodings(dsts[0])[0]
        except IndexError:
            # 認識済みなら
            if recog_flag:
                frame = cv2.resize(frame, (178, 218))
                im_tile = concat_tile([[im0, im1, im2],
                                       [im3, frame, im4],
                                       [im5, im6, im7]])

                cv2.imshow('tile camera', im_tile)

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

            print("{0}:{1}".format(k, distance))
            print("number{} is end".format(i))
            i += 1
        print("finish about one face")

        # 結果画像の読み込み(178*218)
        im0 = cv2.imread("./database/{}".format(similar_paths[0]))
        im1 = cv2.imread("./database/{}".format(similar_paths[1]))
        im2 = cv2.imread("./database/{}".format(similar_paths[2]))
        im3 = cv2.imread("./database/{}".format(similar_paths[3]))
        im4 = cv2.imread("./database/{}".format(similar_paths[4]))
        im5 = cv2.imread("./database/{}".format(similar_paths[5]))
        im6 = cv2.imread("./database/{}".format(similar_paths[6]))
        im7 = cv2.imread("./database/{}".format(similar_paths[7]))
        # im8 = cv2.imread("./database/{}".format(similar_paths[8]))
        # im9 = cv2.imread("./database/{}".format(similar_paths[9]))
        frame = cv2.resize(frame, (178, 218))

        im_tile = concat_tile([[im0, im1, im2],
                               [im3, frame, im4],
                               [im5, im6, im7]])

        recog_flag = True
        cv2.imshow('tile camera', im_tile)


        # to break the loop by pressing esc
        k = cv2.waitKey(1)
        if k == 27:
            print("released!")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("release camera!!!")


if __name__ == '__main__':
    main()