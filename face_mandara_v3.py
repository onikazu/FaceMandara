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
k = 9 # 類似顔8こほしいので
d = 128 # 顔特徴ベクトルの次元数
quantizer = faiss.IndexFlatL2(d)  # this remains the same
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
index.train(face_vectors)
index.add(face_vectors)



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

        target_vector = np.array(list(target_image_encoded)).astype("float32")
        target_vector.resize((1, 128))

        similar_paths = []
        D, I = index.search(target_vector, k)
        for i in range(1, len(I[0])):
            similar_paths.append(face_image_names[I[0][i]])
        print("I", I)
        print(similar_paths)

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


