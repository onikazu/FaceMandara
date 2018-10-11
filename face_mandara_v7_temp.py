"""
オブジェクト指向を取り入れる
"""


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
k = 9  # 類似顔8こほしいのでk=9
d = 128  # 顔特徴ベクトルの次元数
quantizer = faiss.IndexFlatL2(d)  # this remains the same
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
index.train(face_vectors)
index.add(face_vectors)

print("indexing is end")


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
            # 画像パスの保存(何らかの問題あり)
            similar_paths_manager.append(similar_paths)

        # 距離の保存
        if similar_distance_manager[:] == []:
            similar_distance_manager.append(D)
        else:
            similar_distance_manager[0] = D

# 時刻tでの円を描く図の座標を返す
def circle_mover(t, frame, rect):
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]
    rect_top = rect[0]
    rect_bottom = rect[1]
    rect_left = rect[2]
    rect_right = rect[3]
    rad = (t / 360) * 2 * math.pi
    x = math.cos(rad) * 200 + frame_width/2 - 100
    y = math.sin(rad) * 200 + frame_height/2 - 100
    return [y, x]


class similar_window:
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
            frame = self.exe_put(self.place_y, self.place_y+window_height, self.place_x, self.place_x+window_width, image)
            return frame
        except:
            print("something is happened in put_on_frame")
            print("image", type(image))
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_traceback,
                              limit=2, file=sys.stdout)
            # 未処理のフレームを返す
            return frame

    def exe_put(self, top, bottom, left, right, image):
        part_frame = frame[top:bottom, left:right]
        blended_image = cv2.addWeighted(part_frame, 0, image, 1, 0)
        frame[top:bottom, left:right] = blended_image
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
        cv2.putText(image_num, str(distance),(0, image.shape[0]-padding), font, 0.5,(255,255,255),2,cv2.FONT_HERSHEY_TRIPLEX)
        return image_num


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

        start_time = time.time()

        cap = cv2.VideoCapture(0)  # 引数はカメラのデバイス番号
        while True:
            ret, frame = cap.read()

            # 配列への変換/共有メモリへの代入
            # print("frame_manager", frame_manager[:])
            if frame_manager[:] == []:
                frame_manager.append(list(frame))
            else:
                frame_manager[0] = list(frame)

            # 起動後すぐの写真は識別精度が下がるため
            check_time = time.time()
            if check_time - start_time > 10:
                print("ready for mandara")
                pass
            else:
                continue



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


            # インスタンス作成
            similar_windows = []
            for i in range(len(face_rect_manager)):
                for j in range(len(all_images[i])):
                    sw = similar_window(distance=distance_no1, place=[0, 0], image=all_images[i][j])
                    similar_windows.append(sw)
                    similar_windows[i].time += j * 20

            print("len similar_windows", len(similar_windows))
            print("len all images", len(all_images[0]))

            while True:
                #print("distance_no1", distance_no1)
                ret, frame = cap.read()
                # 鏡のように表示
                frame = cv2.flip(frame, 1)

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


                # 配列への変換/共有メモリへの代入
                # print("frame_manager", frame_manager[:])

                if frame_manager[:] == []:
                    frame_manager.append(list(frame))
                else:
                    frame_manager[0] = list(frame)

                # 以下画像加工部分
                # オーバーレイの作成
                overlay = frame.copy()

                # 距離データ枠の挿入
                if similar_distance_manager:
                    cv2.rectangle(overlay, (0, 320), (185, 40), (0, 0, 0), -1)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.addWeighted(frame, 0.5, overlay, 0.5, 0, frame)

                # 顔認識のフレーム表示
                for i in range(len(rects)):
                    frame = cv2.rectangle(frame, (rects[i][2], rects[i][0]), (rects[i][3], rects[i][1]), (0, 255, 3), 3)

                # 類似顔表示
                speed = 5
                print("len similar_windows", len(similar_windows))
                for i in range(len(similar_windows)):
                    y = i * 50
                    x = 0
                    similar_windows[i].put_on_frame(frame=frame, place=[y, x])
                    print("put", i)

                for i in range(len(distance[0])):
                    distance[0][i] = distance[0][i] + random.uniform(0, 0.00000001)
                    print(distance[0][i])

                cv2.putText(frame, "distances",(30, 75), font, 0.5,(255,255,255),2,cv2.FONT_HERSHEY_TRIPLEX)
                for i in range(len(distance[0])):
                    print("distance", distance[0][i])
                    d = round(distance[0][i], similar_windows[0].time%20)
                    cv2.putText(frame, str(d),(30, 100+25*i), font, 0.5,(255,255,255),2,cv2.FONT_HERSHEY_TRIPLEX)


                cv2.imshow('tile camera', frame)
                k = cv2.waitKey(1)

                if similar_windows[0].time >= 600:
                    break

            if k == 27:
                print("released!")
                break
        cap.release()
        cv2.destroyAllWindows()
        print("release camera!!!")
