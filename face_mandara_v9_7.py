"""
枠をつけた
"""

# ライブラリインポート
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
from objects import similar_window_v9_7, line_v3


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
        D = []
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
            # 画像パスの保存(何らかの問題あり)
            print("similar_paths", similar_paths)
            similar_paths_manager.append(similar_paths)

        # 距離の保存
        similar_distance_manager[:] = []
        similar_distance_manager.append(D)


if __name__ == '__main__':
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

    # faissを用いたPQの準備
    nlist = 100
    m = 8
    k = 8  # 類似顔7こほしいのでk=8
    d = 128  # 顔特徴ベクトルの次元数
    quantizer = faiss.IndexFlatL2(d)  # this remains the same
    index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
    index.train(face_vectors)
    index.add(face_vectors)
    print("indexing is end")


    # マネージャの作成
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
        start_time = time.time()
        cap = cv2.VideoCapture(0)  # 引数はカメラのデバイス番号

        # 撮影の開始
        while True:
            ret, frame = cap.read()

            # 配列への変換/共有メモリへの代入
            if frame_manager[:] == []:
                frame_manager.append(list(frame))
            else:
                frame_manager[0] = list(frame)

            # まだ結果が出ていないなら
            if not similar_paths_manager:
                continue

            # 類似顔を入れておく配列
            # similar_paths_managerをPILオブジェクトにした（v8_2との相違点）
            # imagesは一つのターゲットについての検索結果が入る変数。
            # all_imagesは動画内のターゲットすべてについての検索結果
            all_images = []
            try:
                for i in range(len(similar_paths_manager)):
                    images = []
                    for j in range(len(similar_paths_manager[i])):
                        images.append(Image.open("./big_database/{}".format(similar_paths_manager[i][j])))
                    all_images.append(images)
            except:
                print("something occured")

            # 顔認識部分の読み込み
            # face_rect_managerを名前付けしてやっただけ
            rects = []
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
                distance_worst = distance[0][-1]

            # インスタンス作成
            # similar_windows = 二重リスト
            if len(face_rect_manager) != len(all_images):
                continue

            similar_windows = []
            lines = []
            print("len all_images", len(all_images))
            for i in range(len(face_rect_manager)):
                similar_windows_one_rect = []
                lines_for_one_rect = []
                print("i", i)
                for j in range(len(all_images[i])):
                    print("j", j)
                    # 真下
                    if  j == 0:
                        amount_movement_x = 0
                        amount_movement_y = 270
                    # 真左
                    elif j == 1:
                        amount_movement_x = -320
                        amount_movement_y = 0
                    # 真右
                    elif j == 2:
                        amount_movement_x = 320
                        amount_movement_y = 0
                    elif j== 3:
                        amount_movement_x = 200
                        amount_movement_y = 185
                    elif j == 4:
                        amount_movement_x = -200
                        amount_movement_y = 185
                    elif j == 5:
                        amount_movement_x = 160
                        amount_movement_y = 40
                    else:
                        amount_movement_x = -160
                        amount_movement_y = 40

                    sw = similar_window_v9_7.SimilarWindow(distance=distance_no1, \
                    place=[0, 0], image=all_images[i][j], \
                    similar_num=len(all_images[i])-j, \
                    movement_amount=(amount_movement_x, amount_movement_y))
                    similar_windows_one_rect.append(sw)
                    li = line_v3.Line(movement_amount_x=amount_movement_x, movement_amount_y=amount_movement_y)
                    lines_for_one_rect.append(li)
                similar_windows.append(similar_windows_one_rect)
                lines.append(lines_for_one_rect)

            face_frames = []
            # face_frames = []
            for i in range(len(rects)):
                ff = face_frame_v2.FaceFrame(rects[i][0], rects[i][1], rects[i][2], rects[i][3])
                face_frames.append(ff)

            # アニメーション、出力部分
            while True:
                print("animation start")
                ret, frame = cap.read()


                # 人数が変更したらまた位置から認識する
                x = len(rects)
                rects = []
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
                if x != len(rects):
                    print("len rects")
                    print("broke")
                    break

                # 鏡のように表示
                frame = cv2.flip(frame, 1)

                frame_manager[:] = []
                frame_manager.append(list(frame))

                # 以下画像加工部分(前半部分はCV２を用いている)
                # オーバーレイの作成
                overlay = frame.copy()
                # 距離データ枠の挿入
                if similar_distance_manager:
                    cv2.rectangle(overlay, (0, 320), (185, 40), (0, 0, 0), -1)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.addWeighted(frame, 0.5, overlay, 0.5, 0, frame)

                # 距離データの挿入
                try:
                    for i in range(len(distance[0])):
                        distance[0][i] = distance[0][i] + random.uniform(0, 0.00000001)
                    cv2.putText(frame, "distances",(30, 75), font, 0.5,(255,255,255),2,cv2.FONT_HERSHEY_TRIPLEX)
                    for i in range(len(distance[0])):
                        print("distance", distance[0][i])
                        d = round(distance[0][i], similar_windows[0][0].time%20)
                        cv2.putText(frame, str(d),(30, 100+25*i), font, 0.5,(255,255,255),2,cv2.FONT_HERSHEY_TRIPLEX)
                except:
                    pass

                # 顔認識フレーム情報の更新
                for i in range(len(face_frames)):
                    try:
                        face_frames[i].top = face_rect_manager[i][0]
                        face_frames[i].bottom = face_rect_manager[i][1]
                        face_frames[i].left = face_rect_manager[i][2]
                        face_frames[i].right = face_rect_manager[i][3]
                    except:
                        continue

                # フレームデータのPIL変換
                frame = frame[:, :, ::-1].copy()
                frame = Image.fromarray(frame)

                # 顔認識のフレーム表示
                for i in range(len(face_frames)):
                    frame = face_frames[i].draw_frame(frame)

                face_frame_centers = []
                for i in range(len(face_frames)):
                    face_frame_centers.append(face_frames[i].cal_center())


                # 類似顔表示
                end_frame_num = 15
                easing_type = "ease_in_out_circular"
                for i in range(len(similar_windows)):
                    rect_top = rects[i][0]
                    rect_bottom = rects[i][1]
                    rect_left = rects[i][2]
                    rect_right = rects[i][3]
                    rect = (rect_left, rect_top, rect_right, rect_bottom)
                    face_frame_center_x = face_frame_centers[i][1]
                    face_frame_center_y = face_frame_centers[i][0]
                    for j in reversed(range(len(similar_windows[i]))):
                        t = similar_windows[i][j].time
                        if t > end_frame_num:
                            t = end_frame_num

                        x = easing.easing(t, face_frame_center_x, similar_windows[i][j].movement_amount_x, end_frame_num, easing_type)
                        y = easing.easing(t, face_frame_center_y, similar_windows[i][j].movement_amount_y, end_frame_num, easing_type)

                        # 情報更新・直線描画
                        lines[i][j].setter(face_frame_center_x, face_frame_center_y, x, y)
                        lines[i][j].draw_line(frame, rect)

                        x = int(x)
                        y = int(y)
                        print("xy",x, y)
                        similar_windows[i][j].put_on_frame(frame=frame, place=[y, x])
                        print("put", i)

                # アニメーション終了から15秒後に次の検索に入る
                if similar_windows[0][0].time >= end_frame_num+15:
                    break

                # cv2への変換
                frame=np.asarray(frame)
                frame = frame[:, :, ::-1]

                cv2.imshow('FaceMandara', frame)
                k = cv2.waitKey(1)
            if k == 27:
                print("released!")
                break
        cap.release()
        cv2.destroyAllWindows()
        print("release camera!!!")
