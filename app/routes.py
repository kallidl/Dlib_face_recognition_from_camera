# app.py — Flask 后端 + 项目启动入口
import os, time, threading, logging, subprocess, sys
from datetime import datetime
from flask import (Flask, Response, render_template, request,
                   jsonify, send_from_directory)

import cv2
import dlib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from db_manager import (
    init_db, add_person, update_person, delete_person,
    get_all_persons, search_persons, get_person_by_name,
    increment_photo_count, add_recognition_log,
    query_logs, get_log_stats, export_logs_to_csv,
)

# ─────────────────────────────────────────
app = Flask(__name__, template_folder="../templates", static_folder="../static")
logging.basicConfig(level=logging.INFO)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FACES_DIR     = os.path.join(BASE_DIR, "data", "data_faces_from_camera")
SNAPSHOT_DIR  = os.path.join(BASE_DIR, "data", "snapshots")
DATA_DLIB_DIR = os.path.join(BASE_DIR, "data", "data_dlib")

os.makedirs(FACES_DIR,    exist_ok=True)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

EXTRACT_SCRIPT = os.path.join(BASE_DIR, "scripts", "features_extraction_to_csv.py")


# ─────────────────────────────────────────
# ✅ 修复1：直接在Python进程内提取特征，不依赖子进程
# 原来用subprocess运行脚本，子进程会因路径/环境问题静默失败
# ─────────────────────────────────────────
def extract_features_inprocess():
    """
    直接在当前Python进程内提取所有人脸特征并写入 features_all.csv。
    返回 (success: bool, msg: str, count: int)
    """
    try:
        load_dlib()
        csv_path = os.path.join(BASE_DIR, "data", "features_all.csv")

        all_names = []
        all_features = []

        if not os.path.isdir(FACES_DIR):
            return True, "人脸目录为空", 0

        for folder_name in sorted(os.listdir(FACES_DIR)):
            folder_path = os.path.join(FACES_DIR, folder_name)
            if not os.path.isdir(folder_path):
                continue
            # 支持 person_姓名 和 直接姓名 两种目录格式
            if folder_name.startswith("person_"):
                name = folder_name[len("person_"):]
            else:
                name = folder_name
            if not name:
                continue

            photo_files = [f for f in os.listdir(folder_path)
                           if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            if not photo_files:
                continue

            person_features = []
            for fname in photo_files:
                img_path = os.path.join(folder_path, fname)
                try:
                    img = cv2.imdecode(
                        np.fromfile(img_path, dtype=np.uint8),
                        cv2.IMREAD_COLOR
                    )
                    if img is None:
                        continue
                    faces = _detector(img, 1)
                    if len(faces) != 1:
                        continue
                    shape = _predictor(img, faces[0])
                    feat = _reco_model.compute_face_descriptor(img, shape)
                    person_features.append(list(feat))
                except Exception as e:
                    logging.warning("提取 %s/%s 失败: %s", name, fname, e)

            if not person_features:
                continue

            # 取平均特征（多张照片取均值，提高鲁棒性）
            avg_feat = np.mean(person_features, axis=0).tolist()
            all_names.append(name)
            all_features.append(avg_feat)
            logging.info("已提取特征: %s (%d张)", name, len(person_features))

        # 写CSV：第0列姓名，第1~128列特征值
        import csv
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for name, feat in zip(all_names, all_features):
                writer.writerow([name] + feat)

        logging.info("特征提取完成，共 %d 人，已写入 %s", len(all_names), csv_path)
        return True, "ok", len(all_names)

    except Exception as e:
        logging.error("特征提取失败: %s", e)
        return False, str(e), 0


def run_extract_script(timeout=120):
    """保留原来的子进程方式（备用，不再主动调用）"""
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONUTF8'] = '1'
    try:
        result = subprocess.run(
            [sys.executable, "-X", "utf8", EXTRACT_SCRIPT],
            capture_output=True, text=True, timeout=timeout,
            encoding='utf-8', errors='replace', env=env,
            cwd=BASE_DIR
        )
        if result.returncode != 0:
            err_msg = result.stderr[:800] if result.stderr else result.stdout[:800]
            return False, err_msg
        return True, "ok"
    except subprocess.TimeoutExpired:
        return False, "特征提取超时"
    except Exception as e:
        return False, str(e)


def _async_extract_and_reload(log_prefix=""):
    """
    ✅ 修复：异步提取特征并重载特征库
    使用进程内提取（extract_features_inprocess），不依赖子进程，更可靠
    """
    def _run():
        ok, msg, count = extract_features_inprocess()
        if ok:
            CAM.load_features()
            logging.info("%s 特征提取完成，共 %d 人", log_prefix, len(CAM.known_names))
        else:
            logging.warning("%s 特征提取失败: %s", log_prefix, msg)
    threading.Thread(target=_run, daemon=True).start()


# ─────────────────────────────────────────
# Windows 中文路径安全写图工具函数
# ─────────────────────────────────────────
def safe_imwrite(path, img_bgr):
    try:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        pil_img.save(path)
        return True
    except Exception as e:
        logging.error("safe_imwrite failed: %s", e)
        return False


# ─────────────────────────────────────────
# Dlib 延迟加载（线程安全）
# ─────────────────────────────────────────
_detector = _predictor = _reco_model = None
_dlib_lock = threading.Lock()

def load_dlib():
    global _detector, _predictor, _reco_model
    if _detector is None:
        with _dlib_lock:
            if _detector is None:
                _detector   = dlib.get_frontal_face_detector()
                _predictor  = dlib.shape_predictor(
                    os.path.join(DATA_DLIB_DIR, "shape_predictor_68_face_landmarks.dat"))
                _reco_model = dlib.face_recognition_model_v1(
                    os.path.join(DATA_DLIB_DIR, "dlib_face_recognition_resnet_model_v1.dat"))


# ─────────────────────────────────────────
# 从文件系统同步人员到数据库
# ─────────────────────────────────────────
def sync_persons_from_filesystem():
    if not os.path.isdir(FACES_DIR):
        return 0
    added = 0
    for folder_name in os.listdir(FACES_DIR):
        folder_path = os.path.join(FACES_DIR, folder_name)
        if not os.path.isdir(folder_path):
            continue
        if folder_name.startswith("person_"):
            name = folder_name[len("person_"):]
        else:
            name = folder_name
        if not name:
            continue
        photos = [f for f in os.listdir(folder_path)
                  if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        existing = get_person_by_name(name)
        if existing is None:
            new_id = add_person(name, folder_path=folder_path)
            if new_id:
                update_person(name, photo_count=len(photos))
                logging.info("同步人员到数据库: %s（照片数: %d）", name, len(photos))
                added += 1
        else:
            update_person(name, photo_count=len(photos))
    return added


# ─────────────────────────────────────────
# 摄像头枚举
# ─────────────────────────────────────────
def list_cameras(max_test=5):
    available = []
    for i in range(max_test):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available


def _set_cap_props(cap):
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


# ─────────────────────────────────────────
# 识别摄像头状态
# ─────────────────────────────────────────
class CameraState:
    def __init__(self):
        self.cap             = None
        self.running         = False
        self.lock            = threading.Lock()
        self.current_frame   = None
        self.face_count      = 0
        self.detected_names  = []
        self.known_features  = []
        self.known_names     = []
        self.camera_id       = 0
        # OT 质心追踪状态
        self.last_frame_face_centroid_list    = []
        self.current_frame_face_centroid_list = []
        self.last_frame_face_name_list        = []
        self.current_frame_face_name_list     = []
        self.last_frame_face_cnt              = 0
        self.current_frame_face_cnt           = 0
        self.reclassify_interval_cnt          = 0
        self.reclassify_interval              = 10  # 每10帧强制重识别一次

    def load_features(self):
        self.known_features, self.known_names = [], []
        # 同时重置OT追踪状态，防止旧名字残留
        self.last_frame_face_centroid_list    = []
        self.current_frame_face_centroid_list = []
        self.last_frame_face_name_list        = []
        self.current_frame_face_name_list     = []
        self.last_frame_face_cnt              = 0
        self.current_frame_face_cnt           = 0
        self.reclassify_interval_cnt          = 0
        self.detected_names                   = []

        csv_path = os.path.join(BASE_DIR, "data", "features_all.csv")
        if not os.path.exists(csv_path):
            return "empty"
        if os.path.getsize(csv_path) == 0:
            return "empty"
        try:
            df = pd.read_csv(csv_path, header=None, dtype=str)
            for i in range(df.shape[0]):
                self.known_names.append(str(df.iloc[i][0]))
                arr = [df.iloc[i][j] if df.iloc[i][j] != '' else '0'
                       for j in range(1, 129)]
                self.known_features.append(arr)
            logging.info("特征库加载完成，共 %d 人", len(self.known_names))
            return True
        except Exception as e:
            logging.error("加载特征库失败: %s", e)
            return False

    @staticmethod
    def euclid(f1, f2):
        f1, f2 = np.array(f1, dtype=float), np.array(f2, dtype=float)
        return float(np.sqrt(np.sum(np.square(f1 - f2))))

    def recognize(self, feat):
        if not self.known_features:
            return "unknown", 9999.0
        dists = [self.euclid(feat, kf) for kf in self.known_features]
        min_d = min(dists)
        idx   = dists.index(min_d)
        name  = str(self.known_names[idx]) if min_d < 0.4 else "unknown"
        return name, min_d

    def centroid_tracker(self):
        """OT质心追踪：用上一帧质心匹配当前帧人脸，复用上帧识别结果"""
        for i in range(len(self.current_frame_face_centroid_list)):
            dists = [
                self.euclid(self.current_frame_face_centroid_list[i], last_c)
                for last_c in self.last_frame_face_centroid_list
            ]
            best = dists.index(min(dists))
            self.current_frame_face_name_list[i] = self.last_frame_face_name_list[best]

CAM = CameraState()


# ─────────────────────────────────────────
# 录入摄像头状态
# ─────────────────────────────────────────
class RegisterState:
    def __init__(self):
        self.cap           = None
        self.running       = False
        self.lock          = threading.Lock()
        self.current_frame = None
        self.face_count    = 0
        self.current_name  = ""
        self.save_count    = 0
        self.camera_id     = 0
        self.out_of_range  = False
        self.raw_frame     = None

REG = RegisterState()

def _register_loop():
    detector_reg = dlib.get_frontal_face_detector()
    cam = REG.cap
    while REG.running and cam.isOpened():
        ok, frame = cam.read()
        if not ok:
            break
        with REG.lock:
            REG.raw_frame = frame.copy()

        faces = detector_reg(frame, 0)
        draw = frame.copy()
        out_of_range = False
        for d in faces:
            h = d.bottom() - d.top()
            w = d.right() - d.left()
            hh, ww = h // 4, w // 4
            if (d.right() + ww > frame.shape[1] or d.bottom() + hh > frame.shape[0]
                    or d.left() - ww < 0 or d.top() - hh < 0):
                color = (80, 80, 255)
                out_of_range = True
            else:
                color = (0, 255, 128)
            x1 = max(0, d.left() - ww)
            y1 = max(0, d.top() - hh)
            x2 = min(frame.shape[1], d.right() + ww)
            y2 = min(frame.shape[0], d.bottom() + hh)
            cv2.rectangle(draw, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(draw, (d.left(), d.top()), (d.right(), d.bottom()), color, 1)

        REG.out_of_range = out_of_range
        cv2.putText(draw, f"Faces: {len(faces)}", (14, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 128), 1)
        if out_of_range:
            cv2.putText(draw, "Too close - adjust position", (14, 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 255), 1)
        if REG.current_name:
            cv2.putText(draw, f"Saving: {REG.save_count} photos",
                        (14, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1)

        _, buf = cv2.imencode('.jpg', draw, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with REG.lock:
            REG.current_frame = buf.tobytes()
            REG.face_count = len(faces)

    cam.release()
    REG.running = False

def _gen_register_frames():
    while True:
        with REG.lock:
            frame = REG.current_frame
        if frame:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.033)
        else:
            time.sleep(0.01)


# ─────────────────────────────────────────
# 识别摄像头主循环（OT质心追踪方案，与app.py保持一致）
#
# 核心逻辑：
#   场景1：人脸数不变 且 未到重识别间隔 → 只做dlib检测(快)+质心追踪，复用上帧名字，不调特征提取
#   场景2：人脸数变化 或 到达重识别间隔 → 完整识别（检测+特征提取+比对），耗时但频率极低
#
# 效果：识别到人之后绝大多数帧走场景1，dlib特征提取(100~500ms)极少触发，画面流畅不卡顿。
# ─────────────────────────────────────────
def _camera_loop():
    load_dlib()
    CAM.load_features()
    cam_id = CAM.cap

    try:
        font_path = os.path.join(BASE_DIR, "static", "fonts", "simsun.ttc")
        font_ch = ImageFont.truetype(font_path, 28)
    except Exception:
        font_ch = None

    COOLDOWN = {}  # {name: last_log_time}

    while CAM.running and cam_id.isOpened():
        ok, frame = cam_id.read()
        if not ok:
            break

        faces = _detector(frame, 0)

        # 先保存上帧数，再更新当前帧数
        CAM.last_frame_face_cnt    = CAM.current_frame_face_cnt
        CAM.current_frame_face_cnt = len(faces)

        # 保存上帧列表，清空当前帧质心列表
        CAM.last_frame_face_name_list        = CAM.current_frame_face_name_list[:]
        CAM.last_frame_face_centroid_list     = CAM.current_frame_face_centroid_list[:]
        CAM.current_frame_face_centroid_list  = []

        # ── 场景1：人脸数不变 且 不需要重识别 ──
        # 只做质心追踪，复用上帧名字，不调 compute_face_descriptor，耗时 < 5ms
        if (CAM.current_frame_face_cnt == CAM.last_frame_face_cnt and
                CAM.reclassify_interval_cnt != CAM.reclassify_interval):

            if "unknown" in CAM.current_frame_face_name_list:
                CAM.reclassify_interval_cnt += 1

            if CAM.current_frame_face_cnt != 0:
                for d in faces:
                    CAM.current_frame_face_centroid_list.append(
                        [(d.left() + d.right()) / 2, (d.top() + d.bottom()) / 2]
                    )
                # 多人脸时用质心追踪重新对应名字（单人脸直接继承，不需要）
                if CAM.current_frame_face_cnt != 1:
                    CAM.centroid_tracker()

        # ── 场景2：人脸数变化 或 到达重识别间隔 ──
        # 完整识别，耗时较高，但只在人脸数变化或每10帧触发一次
        else:
            CAM.reclassify_interval_cnt = 0

            if CAM.current_frame_face_cnt == 0:
                CAM.current_frame_face_name_list = []
                CAM.detected_names = []
            else:
                CAM.current_frame_face_name_list = []
                names_with_dist = []

                face_features = []
                for face in faces:
                    CAM.current_frame_face_centroid_list.append(
                        [(face.left() + face.right()) / 2, (face.top() + face.bottom()) / 2]
                    )
                    shape = _predictor(frame, face)
                    feat  = _reco_model.compute_face_descriptor(frame, shape)
                    face_features.append(feat)
                    CAM.current_frame_face_name_list.append("unknown")

                for k, feat in enumerate(face_features):
                    name, dist = CAM.recognize(feat)
                    CAM.current_frame_face_name_list[k] = name
                    names_with_dist.append({"name": name, "e_dist": round(dist, 4)})

                    if name != "unknown":
                        now = time.time()
                        if now - COOLDOWN.get(name, 0) > 60:
                            ts   = time.strftime("%Y%m%d_%H%M%S")
                            snap = os.path.join(SNAPSHOT_DIR, f"{name}_{ts}.jpg")
                            frame_copy = frame.copy()
                            threading.Thread(
                                target=lambda p=snap, f=frame_copy: safe_imwrite(p, f),
                                daemon=True).start()
                            add_recognition_log(name, round(dist, 4), CAM.camera_id, snap)
                            COOLDOWN[name] = now

                CAM.detected_names = names_with_dist

        # 场景1追踪后同步 detected_names（保留已有的 e_dist）
        if (CAM.current_frame_face_cnt != 0 and
                len(CAM.current_frame_face_name_list) == CAM.current_frame_face_cnt):
            CAM.detected_names = [
                {"name": CAM.current_frame_face_name_list[i],
                 "e_dist": CAM.detected_names[i].get("e_dist", 0)
                           if i < len(CAM.detected_names) else 0}
                for i in range(CAM.current_frame_face_cnt)
            ]

        # ── 绘制 ──
        draw_frame = frame.copy()
        for i, d in enumerate(faces):
            name  = (CAM.current_frame_face_name_list[i]
                     if i < len(CAM.current_frame_face_name_list) else "unknown")
            color = (0, 255, 128) if name != "unknown" else (80, 80, 255)
            cv2.rectangle(draw_frame, (d.left(), d.top()), (d.right(), d.bottom()), color, 2)
            sz = 12
            for ddx, ddy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
                cx = d.right() if ddx == 1 else d.left()
                cy = d.bottom() if ddy == 1 else d.top()
                cv2.line(draw_frame, (cx, cy), (cx + ddx * sz, cy), color, 2)
                cv2.line(draw_frame, (cx, cy), (cx, cy + ddy * sz), color, 2)
            if font_ch:
                pil_img = Image.fromarray(cv2.cvtColor(draw_frame, cv2.COLOR_BGR2RGB))
                d_draw  = ImageDraw.Draw(pil_img)
                d_draw.text((d.left(), d.bottom() + 6), name, font=font_ch,
                            fill=(0, 255, 128) if name != "unknown" else (80, 80, 255))
                draw_frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            else:
                cv2.putText(draw_frame, name, (d.left(), d.bottom() + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)

        cv2.putText(draw_frame, f"Faces: {CAM.current_frame_face_cnt}", (14, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 128), 1)

        _, buf = cv2.imencode('.jpg', draw_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with CAM.lock:
            CAM.current_frame = buf.tobytes()
            CAM.face_count    = CAM.current_frame_face_cnt

    cam_id.release()
    CAM.running = False


def _gen_frames():
    while True:
        with CAM.lock:
            frame = CAM.current_frame
        if frame:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.033)
        else:
            time.sleep(0.01)


# ─────────────────────────────────────────
# 路由
# ─────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/cameras')
def cameras_list():
    cams = list_cameras()
    return jsonify(cameras=cams)

@app.route('/api/camera/start', methods=['POST'])
def camera_start():
    if CAM.running:
        return jsonify(ok=True, msg="已在运行")
    data = request.json or {}
    cam_id = int(data.get('camera_id', 0))
    CAM.camera_id = cam_id
    if CAM.cap is not None:
        try:
            CAM.cap.release()
        except Exception:
            pass
        CAM.cap = None
    CAM.current_frame = None
    CAM.cap = cv2.VideoCapture(cam_id)
    if not CAM.cap.isOpened():
        return jsonify(ok=False, msg="无法打开摄像头"), 500
    _set_cap_props(CAM.cap)
    CAM.running = True
    t = threading.Thread(target=_camera_loop, daemon=True)
    t.start()
    return jsonify(ok=True)

@app.route('/api/camera/stop', methods=['POST'])
def camera_stop():
    CAM.running = False
    return jsonify(ok=True)

@app.route('/api/camera/status')
def camera_status():
    return jsonify(running=CAM.running, face_count=CAM.face_count,
                   detected=CAM.detected_names)

@app.route('/video_feed')
def video_feed():
    return Response(_gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/register/camera/start', methods=['POST'])
def register_camera_start():
    if REG.running:
        return jsonify(ok=True, msg="已在运行")
    data = request.json or {}
    cam_id = int(data.get('camera_id', 0))
    REG.camera_id = cam_id
    REG.cap = cv2.VideoCapture(cam_id)
    if not REG.cap.isOpened():
        return jsonify(ok=False, msg="无法打开摄像头"), 500
    _set_cap_props(REG.cap)
    REG.running = True
    t = threading.Thread(target=_register_loop, daemon=True)
    t.start()
    return jsonify(ok=True)

@app.route('/api/register/camera/stop', methods=['POST'])
def register_camera_stop():
    REG.running = False
    return jsonify(ok=True)

@app.route('/api/register/camera/status')
def register_camera_status():
    return jsonify(running=REG.running, face_count=REG.face_count,
                   save_count=REG.save_count, out_of_range=REG.out_of_range,
                   current_name=REG.current_name)

@app.route('/register_feed')
def register_feed():
    return Response(_gen_register_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/register/set_name', methods=['POST'])
def register_set_name():
    data = request.json or {}
    name = (data.get('name') or '').strip()
    if not name:
        REG.current_name = ''
        REG.save_count = 0
        return jsonify(ok=True, save_count=0)
    folder = os.path.join(FACES_DIR, f"person_{name}")
    os.makedirs(folder, exist_ok=True)
    REG.current_name = name
    REG.save_count = len([f for f in os.listdir(folder)
                           if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    existing = get_person_by_name(name)
    if existing is None:
        add_person(name,
                   employee_id=data.get('employee_id', ''),
                   department=data.get('department', ''),
                   folder_path=folder)
    return jsonify(ok=True, save_count=REG.save_count)

@app.route('/api/register/capture', methods=['POST'])
def register_capture():
    if not REG.current_name:
        return jsonify(ok=False, msg="请先设置录入姓名"), 400
    if REG.face_count == 0:
        return jsonify(ok=False, msg="当前帧未检测到人脸"), 400
    if REG.face_count > 1:
        return jsonify(ok=False, msg="检测到多张人脸，请确保画面中只有一张"), 400
    if REG.out_of_range:
        return jsonify(ok=False, msg="人脸超出范围，请调整位置"), 400

    with REG.lock:
        raw = REG.raw_frame
    if raw is None:
        return jsonify(ok=False, msg="无法获取当前帧"), 400

    folder = os.path.join(FACES_DIR, f"person_{REG.current_name}")
    os.makedirs(folder, exist_ok=True)
    REG.save_count += 1
    filename = f"img_face_{REG.save_count}.jpg"
    safe_imwrite(os.path.join(folder, filename), raw)
    increment_photo_count(REG.current_name, 1)
    return jsonify(ok=True, save_count=REG.save_count, filename=filename)

@app.route('/api/register/extract_features', methods=['POST'])
def register_extract_features():
    """
    人脸录入页手动触发的特征提取（同步等待完成）
    ✅ 使用进程内提取，不依赖子进程，更可靠
    """
    ok, msg, count = extract_features_inprocess()
    if not ok:
        return jsonify(ok=False, msg=msg), 500
    CAM.load_features()
    return jsonify(ok=True, msg="特征提取完成，已重载特征库",
                   count=len(CAM.known_names))

# ── 人员管理 ──
@app.route('/api/persons', methods=['GET'])
def persons_list():
    kw = request.args.get('q', '').strip()
    rows = search_persons(kw) if kw else get_all_persons()
    for r in rows:
        r['created_at'] = str(r.get('created_at', ''))[:19]
        r['updated_at'] = str(r.get('updated_at', ''))[:19]
    return jsonify(rows)

@app.route('/api/persons', methods=['POST'])
def persons_add():
    data = request.json or {}
    name = (data.get('name') or '').strip()
    if not name:
        return jsonify(ok=False, msg="姓名不能为空"), 400
    folder = os.path.join(FACES_DIR, f"person_{name}")
    new_id = add_person(name,
                        employee_id=data.get('employee_id', ''),
                        department=data.get('department', ''),
                        folder_path=folder)
    if new_id is None:
        return jsonify(ok=False, msg=f"姓名 '{name}' 已存在"), 409
    os.makedirs(folder, exist_ok=True)
    return jsonify(ok=True, id=new_id)

@app.route('/api/persons/<name>', methods=['PUT'])
def persons_update(name):
    data = request.json or {}
    update_person(name,
                  employee_id=data.get('employee_id'),
                  department=data.get('department'))
    return jsonify(ok=True)

@app.route('/api/persons/<name>', methods=['DELETE'])
def persons_delete(name):
    import shutil
    folder = os.path.join(FACES_DIR, f"person_{name}")
    if os.path.isdir(folder):
        shutil.rmtree(folder)
    delete_person(name)
    _async_extract_and_reload(log_prefix=f"[删除:{name}]")
    return jsonify(ok=True)

@app.route('/api/persons/<name>/photos', methods=['GET'])
def persons_photos(name):
    folder = os.path.join(FACES_DIR, f"person_{name}")
    if not os.path.isdir(folder):
        return jsonify([])
    files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    return jsonify(files)

@app.route('/api/persons/<name>/upload_photo', methods=['POST'])
def upload_photo(name):
    if 'photos' not in request.files:
        return jsonify(ok=False, msg="未收到照片文件"), 400

    existing = get_person_by_name(name)
    if existing is None:
        return jsonify(ok=False, msg=f"人员 '{name}' 不存在，请先创建"), 404

    folder = os.path.join(FACES_DIR, f"person_{name}")
    os.makedirs(folder, exist_ok=True)

    existing_photos = [f for f in os.listdir(folder)
                       if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    start_idx = len(existing_photos) + 1

    saved = 0
    files = request.files.getlist('photos')
    for i, file in enumerate(files):
        if file and file.filename:
            ext = os.path.splitext(file.filename)[1].lower() or '.jpg'
            filename = f"img_face_{start_idx + i}{ext}"
            save_path = os.path.join(folder, filename)
            try:
                pil_img = Image.open(file.stream).convert('RGB')
                pil_img.save(save_path)
                saved += 1
            except Exception as e:
                logging.error("上传照片失败 %s: %s", filename, e)

    total = len([f for f in os.listdir(folder)
                 if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

    if saved > 0:
        update_person(name, photo_count=total)
        # ✅ 异步进程内提取特征，上传照片后自动更新特征库
        _async_extract_and_reload(log_prefix=f"[上传照片:{name}]")

    return jsonify(ok=True, saved=saved, total=total)

@app.route('/api/persons/photos/<name>/<filename>')
def serve_photo(name, filename):
    folder = os.path.join(FACES_DIR, f"person_{name}")
    return send_from_directory(folder, filename)

@app.route('/api/features/reload', methods=['POST'])
def reload_features():
    result = CAM.load_features()
    ok = (result is not False)
    return jsonify(ok=ok, count=len(CAM.known_names))

@app.route('/api/features/count', methods=['GET'])
def features_count():
    return jsonify(count=len(CAM.known_names))

@app.route('/api/persons/sync', methods=['POST'])
def persons_sync():
    added = sync_persons_from_filesystem()
    return jsonify(ok=True, added=added)

# ── 识别记录 ──
@app.route('/api/logs', methods=['GET'])
def logs_list():
    name    = request.args.get('name') or None
    start_s = request.args.get('start') or None
    end_s   = request.args.get('end')   or None
    limit   = int(request.args.get('limit', 200))

    def parse_dt(s):
        if not s: return None
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try: return datetime.strptime(s, fmt)
            except: pass
        return None

    rows = query_logs(person_name=name, start_dt=parse_dt(start_s),
                      end_dt=parse_dt(end_s), limit=limit)
    for r in rows:
        r['recognized_at'] = str(r.get('recognized_at', ''))[:19]
        r['e_distance']    = round(r['e_distance'], 4) if r.get('e_distance') else None
        if r.get('snapshot_path'):
            r['snapshot_path'] = os.path.basename(r['snapshot_path'])
    return jsonify(rows)

@app.route('/api/logs/stats', methods=['GET'])
def logs_stats():
    days = int(request.args.get('days', 7))
    rows = get_log_stats(days)
    for r in rows:
        r['last_seen'] = str(r.get('last_seen', ''))[:19]
    return jsonify(rows)

@app.route('/api/logs/export', methods=['GET'])
def logs_export():
    import tempfile
    tmp = os.path.join(tempfile.gettempdir(), f"logs_{int(time.time())}.csv")
    name    = request.args.get('name') or None
    start_s = request.args.get('start') or None
    end_s   = request.args.get('end')   or None

    def parse_dt(s):
        if not s: return None
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try: return datetime.strptime(s, fmt)
            except: pass
        return None

    export_logs_to_csv(tmp, person_name=name,
                       start_dt=parse_dt(start_s), end_dt=parse_dt(end_s))
    with open(tmp, 'rb') as f:
        data = f.read()
    os.unlink(tmp)
    return Response(data, mimetype='text/csv',
                    headers={"Content-Disposition": "attachment; filename=recognition_logs.csv"})

@app.route('/api/logs/clear', methods=['POST'])
def logs_clear():
    from db_manager import get_connection
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM recognition_logs")
            cnt = cur.fetchone()[0]
            cur.execute("DELETE FROM recognition_logs")
        conn.commit()
        return jsonify(ok=True, deleted=cnt)
    except Exception as e:
        return jsonify(ok=False, msg=str(e)), 500
    finally:
        conn.close()

@app.route('/api/snapshots/<filename>')
def serve_snapshot(filename):
    return send_from_directory(SNAPSHOT_DIR, filename)

# ─────────────────────────────────────────
# 启动
# ─────────────────────────────────────────
if __name__ == '__main__':
    init_db()
    n = sync_persons_from_filesystem()
    if n > 0:
        logging.info("启动时同步了 %d 位人员到数据库", n)
    print("\n" + "═"*52)
    print("  人脸识别系统已启动")
    print("  浏览器访问 → http://127.0.0.1:5000")
    print("═"*52 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
