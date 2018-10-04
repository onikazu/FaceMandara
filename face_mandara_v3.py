from multiprocessing import Process, Manager, Value
import pickle
import math
import time

import cv2
import face_recognition
import dlib
import numpy as np


# databaseの読み込み
datas = {}
with open('mini_data.pickle', mode='rb') as f:
    datas = pickle.load(f)


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


def recommend_faces(similar_paths_manager, frame_manager):
    """
    カメラ映像から取得した人物の類似顔を探し出す関数
    """
    while True:
        # frame manager が送られてきていなければcontinue
        if not frame_manager:
            continue

        # numpyへの変換
        frame = frame_manager[0]
        frame = np.array(frame)

        # 顔認識機のインスタンス
        detector = dlib.get_frontal_face_detector()

        # 顔データ(人数分)
        rects = detector(frame, 1)

        # 顔認識できなかったときcontinue
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
        similar_paths_manager[:] = []
        similar_paths_manager.append(similar_paths)


# カメラの撮影と結果表示
if __name__ == '__main__':
    with Manager() as manager:
        # マネージャーの作成
        similar_paths_manager = manager.list()
        frame_manager = manager.list()

        # プロセスの生成
        recommend_process = Process(target=recommend_faces, args=[similar_paths_manager, frame_manager], name="recommend")

        # プロセスの開始
        recommend_process.start()

        cap = cv2.VideoCapture(0)  # 引数はカメラのデバイス番号
        while True:
            ret, frame = cap.read()

            # 配列への変換/共有メモリへの代入
            frame_manager[:] = []
            frame_manager.append(list(frame))

            # まだ結果が出ていないなら
            if not similar_paths_manager:
                frame = cv2.resize(frame, (178 * 3, 218 * 3))
                cv2.imshow("tile camera", frame)
                k = cv2.waitKey(1)
                if k == 27:
                    print("released!")
                    break
                continue

            im0 = cv2.imread("./database/{}".format(similar_paths_manager[0][0]))
            im1 = cv2.imread("./database/{}".format(similar_paths_manager[0][1]))
            im2 = cv2.imread("./database/{}".format(similar_paths_manager[0][2]))
            im3 = cv2.imread("./database/{}".format(similar_paths_manager[0][3]))
            im4 = cv2.imread("./database/{}".format(similar_paths_manager[0][4]))
            im5 = cv2.imread("./database/{}".format(similar_paths_manager[0][5]))
            im6 = cv2.imread("./database/{}".format(similar_paths_manager[0][6]))
            im7 = cv2.imread("./database/{}".format(similar_paths_manager[0][7]))
            frame = cv2.resize(frame, (178, 218))

            im_tile = concat_tile([[im0, im1, im2],
                                   [im3, frame, im4],
                                   [im5, im6, im7]])
            cv2.imshow('tile camera', im_tile)
            k = cv2.waitKey(1)

            if k == 27:
                print("released!")
                break
        cap.release()
        cv2.destroyAllWindows()
        print("release camera!!!")


