# face_reco_with_db.py
# 实时人脸识别 + MySQL 识别记录写入 / Real-time face recognition with DB logging
# 基于 face_reco_from_camera_ot.py，增加识别结果写入 MySQL 功能

import dlib
import numpy as np
import cv2
import os
import pandas as pd
import time
import logging
from PIL import Image, ImageDraw, ImageFont
from db_manager import init_db, add_recognition_log

# ============================================================
# Dlib 模型加载
# ============================================================
detector     = dlib.get_frontal_face_detector()
predictor    = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")


class Face_Recognizer_DB:
    def __init__(self, camera_id=0, save_snapshot=False, snapshot_dir="data/snapshots"):
        """
        camera_id:       摄像头编号，写入日志时使用
        save_snapshot:   是否保存识别成功时的抓拍图片
        snapshot_dir:    抓拍图片保存目录
        """
        self.camera_id     = camera_id
        self.save_snapshot = save_snapshot
        self.snapshot_dir  = snapshot_dir

        self.font         = cv2.FONT_ITALIC
        self.font_chinese = ImageFont.truetype("simsun.ttc", 30)

        # FPS
        self.frame_time       = 0
        self.frame_start_time = 0
        self.fps              = 0
        self.fps_show         = 0
        self.start_time       = time.time()
        self.frame_cnt        = 0

        # 人脸数据库
        self.face_features_known_list = []
        self.face_name_known_list     = []

        # 帧间状态（OT 追踪）
        self.last_frame_face_centroid_list  = []
        self.current_frame_face_centroid_list = []
        self.last_frame_face_name_list      = []
        self.current_frame_face_name_list   = []
        self.last_frame_face_cnt            = 0
        self.current_frame_face_cnt         = 0

        self.current_frame_face_X_e_distance_list = []
        self.current_frame_face_position_list      = []
        self.current_frame_face_feature_list       = []
        self.last_current_frame_centroid_e_distance = 0

        self.reclassify_interval_cnt = 0
        self.reclassify_interval     = 10

        # 最新一帧每张人脸对应的欧氏距离（用于写入 DB）
        self._current_e_distances = {}

        if save_snapshot and not os.path.exists(snapshot_dir):
            os.makedirs(snapshot_dir)

    # ----------------------------------------------------------
    def get_face_database(self):
        if os.path.exists("data/features_all.csv"):
            csv_rd = pd.read_csv("data/features_all.csv", header=None)
            for i in range(csv_rd.shape[0]):
                arr = []
                self.face_name_known_list.append(csv_rd.iloc[i][0])
                for j in range(1, 129):
                    arr.append(csv_rd.iloc[i][j] if csv_rd.iloc[i][j] != '' else '0')
                self.face_features_known_list.append(arr)
            logging.info("DB 中人脸数 / Faces in database: %d", len(self.face_features_known_list))
            return True
        else:
            logging.warning("找不到 features_all.csv，请先运行 features_extraction_to_csv.py")
            return False

    @staticmethod
    def euclidean_distance(f1, f2):
        f1, f2 = np.array(f1), np.array(f2)
        return float(np.sqrt(np.sum(np.square(f1 - f2))))

    def update_fps(self):
        now = time.time()
        if str(self.start_time).split(".")[0] != str(now).split(".")[0]:
            self.fps_show = self.fps
        self.start_time   = now
        self.frame_time   = now - self.frame_start_time
        self.fps          = 1.0 / self.frame_time
        self.frame_start_time = now

    # ----------------------------------------------------------
    # OT 质心追踪
    def centroid_tracker(self):
        for i in range(len(self.current_frame_face_centroid_list)):
            dists = [
                self.euclidean_distance(
                    self.current_frame_face_centroid_list[i],
                    self.last_frame_face_centroid_list[j]
                )
                for j in range(len(self.last_frame_face_centroid_list))
            ]
            self.current_frame_face_name_list[i] = \
                self.last_frame_face_name_list[dists.index(min(dists))]

    # ----------------------------------------------------------
    # 识别单张人脸，返回 (name, e_distance)
    def _recognize_face(self, feature):
        e_list = []
        for known_feat in self.face_features_known_list:
            if str(known_feat[0]) != '0.0':
                e_list.append(self.euclidean_distance(feature, known_feat))
            else:
                e_list.append(999999999)
        min_e  = min(e_list)
        idx    = e_list.index(min_e)
        name   = self.face_name_known_list[idx] if min_e < 0.4 else "unknown"
        return name, min_e

    # ----------------------------------------------------------
    # 写识别日志（含抓拍）
    def _log_recognition(self, name, e_distance, img_rd=None):
        if name == "unknown":
            return  # unknown 不写日志，可按需开启
        snapshot_path = ""
        if self.save_snapshot and img_rd is not None:
            ts = time.strftime("%Y%m%d_%H%M%S")
            snapshot_path = os.path.join(
                self.snapshot_dir, f"{name}_{ts}.jpg"
            )
            cv2.imwrite(snapshot_path, img_rd)

        written = add_recognition_log(
            person_name   = name,
            e_distance    = round(e_distance, 4),
            camera_id     = self.camera_id,
            snapshot_path = snapshot_path,
        )
        if written:
            logging.info("[DB] 写入记录: %s  e=%.4f", name, e_distance)

    # ----------------------------------------------------------
    def draw_note(self, img_rd):
        cv2.putText(img_rd, "Face Recognizer + DB", (20, 40),
                    self.font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img_rd, f"Frame: {self.frame_cnt}",         (20, 100), self.font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img_rd, f"FPS:   {self.fps_show:.2f}",      (20, 130), self.font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img_rd, f"Faces: {self.current_frame_face_cnt}", (20, 160), self.font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "Q: Quit", (20, 450), self.font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

        for i in range(len(self.current_frame_face_name_list)):
            cx, cy = int(self.current_frame_face_centroid_list[i][0]), \
                     int(self.current_frame_face_centroid_list[i][1])
            cv2.putText(img_rd, f"Face_{i+1}", (cx, cy), self.font, 0.8, (255, 190, 0), 1, cv2.LINE_AA)

    def draw_names(self, img_rd):
        img_pil = Image.fromarray(cv2.cvtColor(img_rd, cv2.COLOR_BGR2RGB))
        draw    = ImageDraw.Draw(img_pil)
        for i in range(self.current_frame_face_cnt):
            draw.text(
                xy   = self.current_frame_face_position_list[i],
                text = self.current_frame_face_name_list[i],
                font = self.font_chinese,
                fill = (255, 255, 0)
            )
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # ----------------------------------------------------------
    def process(self, stream):
        if not self.get_face_database():
            return

        while stream.isOpened():
            self.frame_cnt += 1
            flag, img_rd = stream.read()
            if not flag:
                break
            kk    = cv2.waitKey(1)
            faces = detector(img_rd, 0)

            self.last_frame_face_cnt    = self.current_frame_face_cnt
            self.current_frame_face_cnt = len(faces)
            self.last_frame_face_name_list     = self.current_frame_face_name_list[:]
            self.last_frame_face_centroid_list  = self.current_frame_face_centroid_list
            self.current_frame_face_centroid_list = []

            # --- 场景 1：人脸数不变 ---
            if (self.current_frame_face_cnt == self.last_frame_face_cnt and
                    self.reclassify_interval_cnt != self.reclassify_interval):

                self.current_frame_face_position_list = []
                if "unknown" in self.current_frame_face_name_list:
                    self.reclassify_interval_cnt += 1

                if self.current_frame_face_cnt != 0:
                    for k, d in enumerate(faces):
                        self.current_frame_face_position_list.append(
                            (d.left(), int(d.bottom() + (d.bottom() - d.top()) / 4))
                        )
                        self.current_frame_face_centroid_list.append(
                            [(d.left() + d.right()) / 2, (d.top() + d.bottom()) / 2]
                        )
                        img_rd = cv2.rectangle(img_rd,
                                               (d.left(), d.top()), (d.right(), d.bottom()),
                                               (255, 255, 255), 2)

                if self.current_frame_face_cnt != 1:
                    self.centroid_tracker()

                img_rd = self.draw_names(img_rd)
                self.draw_note(img_rd)

            # --- 场景 2：人脸数变化 or 需要重识别 ---
            else:
                self.current_frame_face_position_list    = []
                self.current_frame_face_X_e_distance_list = []
                self.current_frame_face_feature_list      = []
                self.reclassify_interval_cnt = 0

                if self.current_frame_face_cnt == 0:
                    self.current_frame_face_name_list = []
                else:
                    self.current_frame_face_name_list = []
                    # 提取特征
                    for i in range(len(faces)):
                        shape = predictor(img_rd, faces[i])
                        self.current_frame_face_feature_list.append(
                            face_reco_model.compute_face_descriptor(img_rd, shape)
                        )
                        self.current_frame_face_name_list.append("unknown")

                    for k in range(len(faces)):
                        self.current_frame_face_centroid_list.append(
                            [(faces[k].left() + faces[k].right()) / 2,
                             (faces[k].top()  + faces[k].bottom()) / 2]
                        )
                        self.current_frame_face_position_list.append(
                            (faces[k].left(),
                             int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4))
                        )

                        name, e_dist = self._recognize_face(
                            self.current_frame_face_feature_list[k]
                        )
                        self.current_frame_face_name_list[k] = name

                        # 矩形框
                        cv2.rectangle(img_rd,
                                      (faces[k].left(), faces[k].top()),
                                      (faces[k].right(), faces[k].bottom()),
                                      (255, 255, 255), 2)

                        # 写数据库
                        self._log_recognition(name, e_dist, img_rd)

                    self.draw_note(img_rd)

            if kk == ord('q'):
                break

            self.update_fps()
            cv2.namedWindow("camera", 1)
            cv2.imshow("camera", img_rd)

    def run(self):
        cap = cv2.VideoCapture(self.camera_id)
        self.process(cap)
        cap.release()
        cv2.destroyAllWindows()


def main():
    logging.basicConfig(level=logging.INFO)
    # 确保数据库和表已创建
    init_db()
    reco = Face_Recognizer_DB(
        camera_id     = 0,
        save_snapshot = False,   # 改为 True 可保存抓拍截图
        snapshot_dir  = "data/snapshots",
    )
    reco.run()


if __name__ == '__main__':
    main()
