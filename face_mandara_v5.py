"""
複数人を対象に取ることができる
"""

from multiprocessing import Process, Manager, Value
import pickle
import math
import time
import sys

import cv2
import face_recognition
import dlib
import numpy as np
import faiss

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
        # face_rect_managerには複数人数になっても顔特徴ベクトルがそのまま入る
        if face_rect_manager[:] == []:
            for rect in rects:
                dst = frame[rect.top():rect.bottom(), rect.left():rect.right()]
                dsts.append(dst)
                rect_size = [rect.top(), rect.bottom(), rect.left(), rect.right()]
                face_rect_manager.append(rect_size)
        else:
            face_rect_manager[:] = []
            for x, rect in enumerate(rects):
                dst = frame[rect.top():rect.bottom(), rect.left():rect.right()]
                dsts.append(dst)
                rect_size = [rect.top(), rect.bottom(), rect.left(), rect.right()]
                face_rect_manager.append(rect_size)


        # 距離測定(人数分)
        # 顔情報のベクトル化　類似配列の生成
        similar_paths_manager[:] = []
        for i in range(len(dsts)):
            try:
                target_image_encoded = face_recognition.face_encodings(dsts[i])[0]
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
            # 画像パスの保存
            similar_paths_manager.append(similar_paths)

        # 距離の保存
        if similar_distance_manager[:] == []:
            similar_distance_manager.append(D)
        else:
            similar_distance_manager[0] = D


# カメラの撮影と結果表示
if __name__ == '__main__':
    with Manager() as manager:
        # マネージャーの作成
        similar_paths_manager = manager.list()
        similar_distance_manager = manager.list()
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
            # print("frame_manager", frame_manager[:])
            if frame_manager[:] == []:
                frame_manager.append(list(frame))
            else:
                frame_manager[0] = list(frame)

            # まだ結果が出ていないなら
            if not similar_paths_manager:
                # frame = cv2.flip(frame, 1)
                # cv2.imshow("tile camera", frame)
                # k = cv2.waitKey(1)
                # if k == 27:
                #     print("released!")
                #     break
                continue

            # 類似顔を入れておく配列
            all_images = []
            try:
                for i in range(len(similar_paths_manager)):
                    images = []
                    for j in range(len(similar_paths_manager[i])):
                        images.append(cv2.imread("./big_database/{}".format(similar_paths_manager[i][j])))
                    all_images.append(images)
            except:
                print("something occured")

            # 結果表示部分
            # 顔認識部分の読み込み
            rects = []
            # print("typeaaaaaa", type(face_rect_manager[0]))
            # print("face_rect_manager[i][0]", face_rect_manager[0][0])
            try:
                for i in range(len(face_rect_manager)):
                    rect = []
                    # print("i", i)
                    rect.append(face_rect_manager[i][0])# top
                    rect.append(face_rect_manager[i][1]) # bottom
                    rect.append(face_rect_manager[i][2]) # left
                    rect.append(face_rect_manager[i][3]) # right
                    rects.append(rect)
            except:
                print("something occured")

            if similar_distance_manager:
                distance = similar_distance_manager[0]
                distance_no1 = distance[0][0]
                distance_no2 = distance[0][1]
                distance_no3 = distance[0][2]

            # 以下画像加工部分
            # オーバーレイの作成
            overlay = frame.copy()

            # 距離データ枠の挿入
            if similar_distance_manager:
                cv2.rectangle(overlay, (1000, 350), (1250, 50), (0, 0, 0), -1)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.addWeighted(frame, 0.5, overlay, 0.5, 0, frame)

            # 顔認識のフレーム表示
            for i in range(len(rects)):
                frame = cv2.rectangle(frame, (rects[i][2], rects[i][0]), (rects[i][3], rects[i][1]), (0, 255, 3), 3)

            # 類似顔表示
            print("len rects", len(rects))
            print(rects)
            # print("images", images)
            for i in range(len(rects)):
                # 左側に表示
                try:
                    part_frame = frame[rects[i][0]:rects[i][0] + 218, rects[i][2] - 178:rects[i][2]]
                    blended_image = cv2.addWeighted(part_frame, 0, all_images[i][0], 1, 0)
                    frame[rects[i][0]:rects[i][0] + 218, rects[i][2] - 178:rects[i][2]] = blended_image
                except:
                    print("im im here")
                    print(sys.exc_info())
                    pass
                # 右側に表示
                try:
                    part_frame = frame[rects[i][0]:rects[i][0] + 218, rects[i][3]:rects[i][3] + 178]
                    blended_image = cv2.addWeighted(part_frame, 0.5, all_images[i][1], 0.5, 0)
                    frame[rects[i][0]:rects[i][0] + 218, rects[i][3]:rects[i][3] + 178] = blended_image
                except:
                    print(sys.exc_info())
                    pass
                # 右上に表示
                # try:
                #     part_frame = frame[rects[i][0] - 218:rects[i][0], rects[i][2]:rects[i][2] + 178]
                #     blended_image = cv2.addWeighted(part_frame, 0.5, all_images[i][2], 0.5, 0)
                #     frame[rects[i][0] - 218:rects[i][0], rects[i][2]:rects[i][2] + 178] = blended_image
                # except:
                #     print(sys.exc_info())
                #     pass
                # 左上に表示
                try:
                    part_frame = frame[rects[i][0] - 218:rects[i][0], rects[i][3]-178:rects[i][3]]
                    blended_image = cv2.addWeighted(part_frame, 0.5, all_images[i][3], 0.5, 0)
                    frame[rects[i][0] - 218:rects[i][0], rects[i][3]-178:rects[i][3]] = blended_image
                except:
                    print(sys.exc_info())
                    pass
                # 左下に
                try:
                    part_frame = frame[rects[i][1]:rects[i][1]+218, rects[i][3]-178:rects[i][3]]
                    blended_image = cv2.addWeighted(part_frame, 0.5, all_images[i][4], 0.5, 0)
                    frame[rects[i][1]:rects[i][1]+218, rects[i][3]-178:rects[i][3]] = blended_image
                except:
                    print(sys.exc_info())
                    pass
                # 右下に表示
                try:
                    part_frame = frame[rects[i][1]:rects[i][1]+218, rects[i][2]:rects[i][2] + 178]
                    blended_image = cv2.addWeighted(part_frame, 0.5, all_images[i][5], 0.5, 0)
                    frame[rects[i][1]:rects[i][1]+218, rects[i][2]:rects[i][2] + 178] = blended_image
                except:
                    print(sys.exc_info())
                    pass

                # 左角に表示
                try:
                    part_frame = frame[rects[i][0] - 218:rects[i][0], rects[i][2]-178:rects[i][2]]
                    blended_image = cv2.addWeighted(part_frame, 0.5, all_images[i][6], 0.5, 0)
                    frame[rects[i][0] - 218:rects[i][0], rects[i][2]-178:rects[i][2]] = blended_image
                except:
                    print(sys.exc_info())
                    pass

                # 右角
                try:
                    part_frame = frame[rects[i][0] - 218:rects[i][0], rects[i][3]:rects[i][3]+178]
                    blended_image = cv2.addWeighted(part_frame, 0.5, all_images[i][7], 0.5, 0)
                    frame[rects[i][0] - 218:rects[i][0], rects[i][3]:rects[i][3]+178] = blended_image
                except:
                    print(sys.exc_info())
                    pass



            # 鏡のように表示
            frame = cv2.flip(frame, 1)

            # cv2.putText(frame,'1:{0:.4f}'.format(distance_no1),(30, 100), font, 1,(255,255,255),2,cv2.LINE_AA)
            # cv2.putText(frame,'2:{0:.4f}'.format(distance_no2),(30, 150), font, 1,(255,255,255),2,cv2.LINE_AA)
            # cv2.putText(frame,'3:{0:.4f}'.format(distance_no3),(30, 200), font, 1,(255,255,255),2,cv2.LINE_AA)

            # cv2.putText(frame,'1:{0:.10f}'.format(distance_no1),(30, 100), font, 0.5,(255,255,255),2,cv2.FONT_HERSHEY_TRIPLEX)
            # cv2.putText(frame,'2:{0:.10f}'.format(distance_no2),(30, 125), font, 0.5,(255,255,255),2,cv2.FONT_HERSHEY_TRIPLEX)
            # cv2.putText(frame,'3:{0:.10f}'.format(distance_no3),(30, 150), font, 0.5,(255,255,255),2,cv2.FONT_HERSHEY_TRIPLEX)

            for i in range(len(distance[0])):
                cv2.putText(frame,'{0:.20f}'.format(distance[0][i]),(30, 100+25*i), font, 0.5,(255,255,255),2,cv2.FONT_HERSHEY_TRIPLEX)


            cv2.imshow('tile camera', frame)
            k = cv2.waitKey(1)

            if k == 27:
                print("released!")
                break
        cap.release()
        cv2.destroyAllWindows()
        print("release camera!!!")
