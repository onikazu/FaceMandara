"""
結果表示方法の変更
"""

from multiprocessing import Process, Manager, Value
import pickle
import math
import time

import cv2
import face_recognition
import dlib
import numpy as np
import faiss

# databaseの読み込み
print("start indexing")
datas = {}
with open('mini_data.pickle', mode='rb') as f:
    datas = pickle.load(f)
# databese配列の作成
face_image_names = []
face_vectors = []
for k in datas:
    face_image_names.append(k)
    face_vectors.append(datas[k])
face_vectors = np.array(face_vectors).astype("float32")

# faissを用いた
nlist = 100
m = 8
k = 9  # 類似顔8こほしいので
d = 128  # 顔特徴ベクトルの次元数
quantizer = faiss.IndexFlatL2(d)  # this remains the same
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
index.train(face_vectors)
index.add(face_vectors)
print("indexing is end")


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
        try:
            frame = frame_manager[0]
        except OSError:
            print("OSerror occured")
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
        if face_rect_manager[:] == []:
            for rect in rects:
                dst = frame[rect.top():rect.bottom(), rect.left():rect.right()]
                dsts.append(dst)
                face_rect_manager.append(rect.top())
                face_rect_manager.append(rect.bottom())
                face_rect_manager.append(rect.left())
                face_rect_manager.append(rect.right())
        else:
            for x, rect in enumerate(rects):
                dst = frame[rect.top():rect.bottom(), rect.left():rect.right()]
                dsts.append(dst)
                face_rect_manager[4 * x] = rect.top()
                face_rect_manager[4 * x + 1] = rect.bottom()
                face_rect_manager[4 * x + 2] = rect.left()
                face_rect_manager[4 * x + 3] = rect.right()

        # 距離測定（とりあえず一人だけ）
        # 顔情報のベクトル化　類似配列の生成
        try:
            target_image_encoded = face_recognition.face_encodings(dsts[0])[0]
        except IndexError:
            continue

        target_vector = np.array(list(target_image_encoded)).astype("float32")
        target_vector.resize((1, 128))

        similar_paths = []
        D, I = index.search(target_vector, k)
        for i in range(1, len(I[0])):
            similar_paths.append(face_image_names[I[0][i]])
        print("I", I)
        print(similar_paths)

        print("finish about one face")
        if similar_paths_manager[:] == []:
            similar_paths_manager.append(similar_paths)
        else:
            similar_paths_manager[0] = similar_paths


# カメラの撮影と結果表示
if __name__ == '__main__':
    with Manager() as manager:
        # マネージャーの作成
        similar_paths_manager = manager.list()
        frame_manager = manager.list()
        face_rect_manager = manager.list()

        # プロセスの生成
        recommend_process = Process(target=recommend_faces, args=[similar_paths_manager, frame_manager],
                                    name="recommend")

        # プロセスの開始
        recommend_process.start()

        cap = cv2.VideoCapture(0)  # 引数はカメラのデバイス番号
        while True:
            ret, frame = cap.read()

            # 配列への変換/共有メモリへの代入
            if frame_manager[:] == []:
                frame_manager.append(list(frame))
            else:
                frame_manager[0] = list(frame)

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

            # 結果表示部分
            # 顔認識部分の読み込み
            rect_top = face_rect_manager[0]
            rect_bottom = face_rect_manager[1]
            rect_left = face_rect_manager[2]
            rect_right = face_rect_manager[3]
            print(rect_top, rect_bottom, rect_left, rect_right)
            height = im0.shape[0]
            width = im0.shape[1]

            frame = cv2.rectangle(frame, (rect_left, rect_top), (rect_right, rect_bottom), (0, 255, 3), 3)
            try:
                part_frame = frame[rect_top + 45:rect_top + 218 + 45, rect_left - 178:rect_left]
                blended_image = cv2.addWeighted(part_frame, 0.5, im0, 0.5, 0)
                frame[rect_top + 45:rect_top + 218 + 45, rect_left - 178:rect_left] = blended_image
            except ValueError:
                pass
            try:
                frame[rect_top + 45:rect_top + 218 + 45, rect_right:rect_right + 178] = im1
            except ValueError:
                pass
            try:
                frame[rect_top - 218:rect_top, rect_left:rect_left + 178 + 30] = im2
            except ValueError:
                pass

            frame = cv2.flip(frame, 1)
            cv2.imshow('tile camera', frame)
            k = cv2.waitKey(1)

            if k == 27:
                print("released!")
                break
        cap.release()
        cv2.destroyAllWindows()
        print("release camera!!!")
