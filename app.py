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
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

FACES_DIR     = "data/data_faces_from_camera/"
SNAPSHOT_DIR  = "data/snapshots/"
DATA_DLIB_DIR = "data/data_dlib/"
os.makedirs(FACES_DIR,    exist_ok=True)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# ─────────────────────────────────────────
# Windows 中文路径安全写图工具函数
# cv2.imwrite 在 Windows 下不支持中文路径，用 PIL 代替
# ─────────────────────────────────────────
def safe_imwrite(path, img_bgr):
    """兼容 Windows 中文路径的图片保存（BGR numpy array）。"""
    try:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        pil_img.save(path)
        return True
    except Exception as e:
        logging.error("safe_imwrite failed: %s", e)
        return False

# ─────────────────────────────────────────
# Dlib 延迟加载
# ─────────────────────────────────────────
_detector = _predictor = _reco_model = None

def load_dlib():
    global _detector, _predictor, _reco_model
    if _detector is None:
        _detector   = dlib.get_frontal_face_detector()
        _predictor  = dlib.shape_predictor(DATA_DLIB_DIR + "shape_predictor_68_face_landmarks.dat")
        _reco_model = dlib.face_recognition_model_v1(DATA_DLIB_DIR + "dlib_face_recognition_resnet_model_v1.dat")

# ─────────────────────────────────────────
# 从文件系统同步人员到数据库
# ─────────────────────────────────────────
def sync_persons_from_filesystem():
    """扫描 data/data_faces_from_camera/ 将未入库人员自动添加到 persons 表"""
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
        self.last_cnt        = 0
        self.reclassify_cnt  = 0
        self.reclassify_interval = 10
        self.known_features  = []
        self.known_names     = []
        self.camera_id       = 0
        # OT 追踪状态
        self.last_frame_face_centroid_list  = []
        self.current_frame_face_centroid_list = []
        self.last_frame_face_name_list      = []
        self.current_frame_face_name_list   = []
        self.last_frame_face_cnt            = 0
        self.reclassify_interval_cnt        = 0

    def load_features(self):
        """加载特征库，同时重置OT追踪状态和识别缓存，确保删除人员后立即生效"""
        self.known_features, self.known_names = [], []
        # 重置OT状态，避免旧名字残留
        self.last_frame_face_centroid_list  = []
        self.current_frame_face_centroid_list = []
        self.last_frame_face_name_list      = []
        self.current_frame_face_name_list   = []
        self.last_frame_face_cnt            = 0
        self.reclassify_interval_cnt        = 0
        self.detected_names                 = []

        csv_path = "data/features_all.csv"
        if not os.path.exists(csv_path):
            return False
        # 空文件检查
        if os.path.getsize(csv_path) == 0:
            return True
        try:
            df = pd.read_csv(csv_path, header=None)
            for i in range(df.shape[0]):
                self.known_names.append(df.iloc[i][0])
                arr = [df.iloc[i][j] if df.iloc[i][j] != '' else '0' for j in range(1, 129)]
                self.known_features.append(arr)
        except Exception as e:
            logging.error("加载特征库失败: %s", e)
            return False
        return True

    @staticmethod
    def euclid(f1, f2):
        f1, f2 = np.array(f1, dtype=float), np.array(f2, dtype=float)
        return float(np.sqrt(np.sum(np.square(f1 - f2))))

    def recognize(self, feat):
        if not self.known_features:
            return "unknown", 9999.0
        dists = []
        for kf in self.known_features:
            if str(kf[0]) != '0.0':
                dists.append(self.euclid(feat, kf))
            else:
                dists.append(9999.0)
        min_d = min(dists)
        idx   = dists.index(min_d)
        name  = self.known_names[idx] if min_d < 0.4 else "unknown"
        return name, min_d

    def centroid_tracker(self):
        """OT质心追踪：用上一帧质心匹配当前帧人脸，避免每帧重新识别"""
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
            hh, ww = h // 2, w // 2
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

# ─────────────────────────────────────────
# 识别视频流线程
# ─────────────────────────────────────────
def _camera_loop():
    """
    使用 OT（质心追踪）算法的识别循环：
    - 人脸数变化 或 reclassify_interval 到达时，才做完整识别（计算特征描述子）
    - 其余帧只做检测 + 质心追踪，大幅提升 FPS
    - load_features() 时已重置 OT 状态，保证删除人员后立即生效
    """
    load_dlib()
    CAM.load_features()
    cam_id = CAM.cap

    try:
        font_ch = ImageFont.truetype("simsun.ttc", 28)
    except Exception:
        font_ch = None

    COOLDOWN = {}  # {name: last_log_time}

    while CAM.running and cam_id.isOpened():
        ok, frame = cam_id.read()
        if not ok:
            break

        faces = _detector(frame, 0)
        cur_cnt = len(faces)

        # 更新帧间状态
        CAM.last_frame_face_cnt = cur_cnt  # 用完立刻更新（下面判断前先保存上帧值）
        last_cnt_snapshot = CAM.last_cnt
        CAM.last_cnt = cur_cnt

        # ── 场景一：人脸数不变 且 不需要重识别 ──
        if cur_cnt == last_cnt_snapshot and CAM.reclassify_interval_cnt != CAM.reclassify_interval:
            if "unknown" in [x.get("name") for x in CAM.detected_names]:
                CAM.reclassify_interval_cnt += 1

            CAM.current_frame_face_centroid_list = []
            if cur_cnt > 0:
                for d in faces:
                    CAM.current_frame_face_centroid_list.append(
                        [(d.left() + d.right()) / 2, (d.top() + d.bottom()) / 2]
                    )
                # 多人脸时用质心追踪复用上帧识别结果
                if cur_cnt != 1 and len(CAM.last_frame_face_centroid_list) == cur_cnt:
                    CAM.centroid_tracker()

        # ── 场景二：人脸数变化 或 需要重识别 ──
        else:
            CAM.reclassify_interval_cnt = 0
            CAM.current_frame_face_centroid_list = []

            if cur_cnt == 0:
                CAM.current_frame_face_name_list = []
                CAM.detected_names = []
            else:
                CAM.current_frame_face_name_list = []
                names_with_dist = []

                for i, face in enumerate(faces):
                    CAM.current_frame_face_centroid_list.append(
                        [(face.left() + face.right()) / 2, (face.top() + face.bottom()) / 2]
                    )
                    shape = _predictor(frame, face)
                    feat  = _reco_model.compute_face_descriptor(frame, shape)
                    name, dist = CAM.recognize(feat)
                    CAM.current_frame_face_name_list.append(name)
                    names_with_dist.append({"name": name, "e_dist": round(dist, 4)})

                    # 写识别日志（带冷却）
                    if name != "unknown":
                        now = time.time()
                        if now - COOLDOWN.get(name, 0) > 60:
                            ts   = time.strftime("%Y%m%d_%H%M%S")
                            snap = os.path.join(SNAPSHOT_DIR, f"{name}_{ts}.jpg")
                            safe_imwrite(snap, frame)
                            add_recognition_log(name, round(dist, 4), CAM.camera_id, snap)
                            COOLDOWN[name] = now

                CAM.detected_names = names_with_dist

        # 更新质心历史
        CAM.last_frame_face_centroid_list = list(CAM.current_frame_face_centroid_list)
        CAM.last_frame_face_name_list     = list(CAM.current_frame_face_name_list)

        # 同步 detected_names（场景一追踪后更新）
        if cur_cnt > 0 and len(CAM.current_frame_face_name_list) == cur_cnt:
            CAM.detected_names = [
                {"name": CAM.current_frame_face_name_list[i], "e_dist": 0}
                if i >= len(CAM.detected_names)
                else {"name": CAM.current_frame_face_name_list[i],
                      "e_dist": CAM.detected_names[i].get("e_dist", 0)
                              if i < len(CAM.detected_names) else 0}
                for i in range(cur_cnt)
            ]

        # ── 绘制 ──
        draw_frame = frame.copy()
        for i, d in enumerate(faces):
            name  = CAM.current_frame_face_name_list[i] if i < len(CAM.current_frame_face_name_list) else "unknown"
            color = (0, 255, 128) if name != "unknown" else (80, 80, 255)
            cv2.rectangle(draw_frame, (d.left(), d.top()), (d.right(), d.bottom()), color, 2)
            # 四角标记
            sz = 12
            for ddx, ddy in [(-1,-1),(1,-1),(1,1),(-1,1)]:
                cx = d.right() if ddx == 1 else d.left()
                cy = d.bottom() if ddy == 1 else d.top()
                cv2.line(draw_frame, (cx, cy), (cx + ddx*sz, cy), color, 2)
                cv2.line(draw_frame, (cx, cy), (cx, cy + ddy*sz), color, 2)
            # 名字
            if font_ch:
                pil_img = Image.fromarray(cv2.cvtColor(draw_frame, cv2.COLOR_BGR2RGB))
                d_draw  = ImageDraw.Draw(pil_img)
                d_draw.text((d.left(), d.bottom() + 6), name, font=font_ch,
                            fill=(0, 255, 128) if name != "unknown" else (80, 80, 255))
                draw_frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            else:
                cv2.putText(draw_frame, name, (d.left(), d.bottom() + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)

        cv2.putText(draw_frame, f"Faces: {cur_cnt}", (14, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 128), 1)

        _, buf = cv2.imencode('.jpg', draw_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with CAM.lock:
            CAM.current_frame = buf.tobytes()
            CAM.face_count    = cur_cnt

    cam_id.release()
    CAM.running = False

def _gen_frames():
    while True:
        with CAM.lock:
            frame = CAM.current_frame
        if frame:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.033)

# ─────────────────────────────────────────
# 路由
# ─────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')

# ── 摄像头列表 ──
@app.route('/api/cameras')
def cameras_list():
    cams = list_cameras()
    return jsonify(cameras=cams)

# ── 识别摄像头 ──
@app.route('/api/camera/start', methods=['POST'])
def camera_start():
    if CAM.running:
        return jsonify(ok=True, msg="已在运行")
    data = request.json or {}
    cam_id = int(data.get('camera_id', 0))
    CAM.camera_id = cam_id
    CAM.cap = cv2.VideoCapture(cam_id)
    if not CAM.cap.isOpened():
        return jsonify(ok=False, msg="无法打开摄像头"), 500
    CAM.running = True
    CAM.load_features()
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

# ── 录入摄像头 ──
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
    # 空名称时仅重置当前录入对象（录入下一个人用）
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
        new_id = add_person(name,
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
    # 使用 safe_imwrite 避免 Windows 中文路径乱码问题
    safe_imwrite(os.path.join(folder, filename), raw)
    increment_photo_count(REG.current_name, 1)
    return jsonify(ok=True, save_count=REG.save_count, filename=filename)

@app.route('/api/register/extract_features', methods=['POST'])
def register_extract_features():
    try:
        # 显式指定 UTF-8 编码环境，避免 Windows 下子进程编码问题
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONUTF8'] = '1'
        result = subprocess.run(
            [sys.executable, "-X", "utf8", "features_extraction_to_csv.py"],
            capture_output=True, text=True, timeout=120,
            encoding='utf-8', errors='replace', env=env
        )
        if result.returncode != 0:
            err_msg = result.stderr[:800] if result.stderr else result.stdout[:800]
            return jsonify(ok=False, msg=err_msg), 500
        ok = CAM.load_features()
        return jsonify(ok=True, msg="特征提取完成，已重载特征库",
                       count=len(CAM.known_names))
    except subprocess.TimeoutExpired:
        return jsonify(ok=False, msg="特征提取超时"), 500
    except Exception as e:
        return jsonify(ok=False, msg=str(e)), 500

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
    # 删除人员后立即重载特征库，避免识别缓存残留
    CAM.load_features()
    return jsonify(ok=True)

@app.route('/api/persons/<name>/photos', methods=['GET'])
def persons_photos(name):
    folder = os.path.join(FACES_DIR, f"person_{name}")
    if not os.path.isdir(folder):
        return jsonify([])
    files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    return jsonify(files)

# ── 新增：上传照片 API（支持新增/编辑人员时上传照片）──
@app.route('/api/persons/<name>/upload_photo', methods=['POST'])
def upload_photo(name):
    """
    为指定姓名的人员上传一张或多张照片。
    前端通过 multipart/form-data 发送，字段名为 photos（可多文件）。
    """
    if 'photos' not in request.files:
        return jsonify(ok=False, msg="未收到照片文件"), 400

    existing = get_person_by_name(name)
    if existing is None:
        return jsonify(ok=False, msg=f"人员 '{name}' 不存在，请先创建"), 404

    folder = os.path.join(FACES_DIR, f"person_{name}")
    os.makedirs(folder, exist_ok=True)

    # 计算当前已有照片数量，续号保存
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
            # 用 PIL 读取并保存，避免 Windows 中文路径问题
            try:
                pil_img = Image.open(file.stream).convert('RGB')
                pil_img.save(save_path)
                saved += 1
            except Exception as e:
                logging.error("上传照片失败 %s: %s", filename, e)

    if saved > 0:
        # 更新数据库中的照片数
        total = len([f for f in os.listdir(folder)
                     if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        update_person(name, photo_count=total)

    return jsonify(ok=True, saved=saved,
                   total=len([f for f in os.listdir(folder)
                               if f.lower().endswith(('.jpg', '.png', '.jpeg'))]))

@app.route('/api/persons/photos/<name>/<filename>')
def serve_photo(name, filename):
    folder = os.path.join(FACES_DIR, f"person_{name}")
    return send_from_directory(folder, filename)

@app.route('/api/features/reload', methods=['POST'])
def reload_features():
    ok = CAM.load_features()
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
