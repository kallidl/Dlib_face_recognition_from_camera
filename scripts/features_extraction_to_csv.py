# features_extraction_to_csv.py
# 从人脸图像文件中提取人脸特征存入 "features_all.csv"
# 修复：使用 PIL 读取图片以兼容 Windows 中文路径

import os
import sys
import dlib
import csv
import numpy as np
import logging
import cv2
from PIL import Image

# 强制 UTF-8 输出（解决 Windows 子进程编码问题）
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 要读取人脸图像文件的路径
path_images_from_camera = "data/data_faces_from_camera/"

# Dlib 正向人脸检测器
detector = dlib.get_frontal_face_detector()

# Dlib 人脸 landmark 特征点检测器
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')

# Dlib Resnet 人脸识别模型，提取 128D 的特征矢量
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")


def return_128d_features(path_img):
    """
    返回单张图像的 128D 特征。
    使用 PIL 读取图片，兼容 Windows 中文路径（cv2.imread 在 Windows 下不支持中文路径）。
    """
    try:
        # 用 PIL 读取（兼容中文路径），转为 BGR numpy array 供 dlib 使用
        pil_img = Image.open(path_img).convert('RGB')
        img_np  = np.array(pil_img)
        img_rd  = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    except Exception as e:
        logging.warning("无法读取图片 %s: %s", path_img, e)
        return 0

    faces = detector(img_rd, 1)
    logging.info("%-40s %-20s", "检测到人脸的图像:", path_img)

    if len(faces) != 0:
        shape = predictor(img_rd, faces[0])
        face_descriptor = face_reco_model.compute_face_descriptor(img_rd, shape)
    else:
        face_descriptor = 0
        logging.warning("no face detected in: %s", path_img)
    return face_descriptor


def return_features_mean_personX(path_face_personX):
    """返回 personX 的 128D 特征均值。"""
    features_list_personX = []

    if not os.path.isdir(path_face_personX):
        logging.warning("文件夹不存在: %s", path_face_personX)
        return np.zeros(128, dtype=object, order='C')

    photos_list = [f for f in os.listdir(path_face_personX)
                   if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    if not photos_list:
        logging.warning("文件夹内图像文件为空 / Warning: No images in %s/", path_face_personX)
        return np.zeros(128, dtype=object, order='C')

    for photo in photos_list:
        full_path = os.path.join(path_face_personX, photo)
        logging.info("正在读的人脸图像: %s", full_path)
        features_128d = return_128d_features(full_path)
        if features_128d == 0:
            continue
        features_list_personX.append(features_128d)

    if features_list_personX:
        features_mean_personX = np.array(features_list_personX, dtype=object).mean(axis=0)
    else:
        logging.warning("文件夹 %s 中没有检测到有效人脸图片", path_face_personX)
        features_mean_personX = np.zeros(128, dtype=object, order='C')

    return features_mean_personX


def main():
    logging.basicConfig(level=logging.INFO)

    if not os.path.isdir(path_images_from_camera):
        logging.error("人脸图像目录不存在: %s", path_images_from_camera)
        sys.exit(1)

    person_list = os.listdir(path_images_from_camera)
    person_list.sort()

    if not person_list:
        logging.warning("人脸图像目录为空，没有可提取的人脸数据")
        # 创建空的 csv 文件，避免后续读取报错
        open("data/features_all.csv", "w").close()
        return

    valid_count = 0
    with open("data/features_all.csv", "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        for person_folder in person_list:
            folder_full_path = os.path.join(path_images_from_camera, person_folder)

            # 跳过非文件夹
            if not os.path.isdir(folder_full_path):
                continue

            # 解析真实姓名：文件夹格式为 person_<姓名>
            # split('_', 1) 只分割第一个下划线，避免姓名中含下划线出错
            if person_folder.startswith("person_"):
                person_name = person_folder[len("person_"):]
            else:
                person_name = person_folder

            if not person_name:
                logging.warning("跳过空名称文件夹: %s", person_folder)
                continue

            logging.info("处理人员: %s  文件夹: %s", person_name, folder_full_path)

            # 检查是否有照片，没有则跳过（不写入 csv，避免全零特征干扰识别）
            photos = [f for f in os.listdir(folder_full_path)
                      if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            if not photos:
                logging.warning("跳过空文件夹（无照片）: %s", folder_full_path)
                continue

            features_mean_personX = return_features_mean_personX(folder_full_path)

            # 如果特征全为 0（没有检测到人脸），也跳过，不写入 csv
            if np.all(np.array(features_mean_personX, dtype=float) == 0):
                logging.warning("跳过（未能提取到有效特征）: %s", person_name)
                continue

            row = np.insert(features_mean_personX, 0, person_name, axis=0)
            writer.writerow(row)
            valid_count += 1
            logging.info("已写入: %s\n", person_name)

    logging.info("完成！共提取 %d 人的特征，保存至 data/features_all.csv", valid_count)


if __name__ == '__main__':
    main()
