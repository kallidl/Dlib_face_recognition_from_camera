# db_manager.py
# MySQL 数据库管理模块 / MySQL database manager


import pymysql
import logging
from datetime import datetime

# ============================================================
# 数据库连接配置 / Database connection config — 按需修改
# ============================================================
DB_CONFIG = {
    "host":     "localhost",
    "port":     3306,
    "user":     "root",
    "password": "Aa@@123456",   # ← 修改为你的密码
    "database": "face_reco_db",    # ← 数据库名，会自动创建
    "charset":  "utf8mb4",
}


def get_connection():
    """获取数据库连接 / Get a new DB connection."""
    return pymysql.connect(**DB_CONFIG)


# ============================================================
# 初始化 — 建库建表 / Create database and tables if not exist
# ============================================================
def init_db():
    """首次运行时调用，自动建库建表。"""
    # 先不指定 database，用于创建库
    cfg = {k: v for k, v in DB_CONFIG.items() if k != "database"}
    conn = pymysql.connect(**cfg)
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"CREATE DATABASE IF NOT EXISTS `{DB_CONFIG['database']}` "
                f"DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
            )
        conn.commit()
    finally:
        conn.close()

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            # ---------- persons 表 ----------
            cur.execute("""
                CREATE TABLE IF NOT EXISTS persons (
                    id            INT AUTO_INCREMENT PRIMARY KEY,
                    name          VARCHAR(100) NOT NULL UNIQUE COMMENT '姓名',
                    employee_id   VARCHAR(50)  DEFAULT '' COMMENT '工号',
                    department    VARCHAR(100) DEFAULT '' COMMENT '部门',
                    photo_count   INT          DEFAULT 0  COMMENT '录入照片数量',
                    folder_path   VARCHAR(255) DEFAULT '' COMMENT '照片文件夹路径',
                    created_at    DATETIME     DEFAULT CURRENT_TIMESTAMP,
                    updated_at    DATETIME     DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                ) ENGINE=InnoDB COMMENT='人员信息表';
            """)

            # ---------- recognition_logs 表 ----------
            cur.execute("""
                CREATE TABLE IF NOT EXISTS recognition_logs (
                    id            INT AUTO_INCREMENT PRIMARY KEY,
                    person_id     INT          DEFAULT NULL COMMENT '关联 persons.id，NULL 表示未知人脸',
                    person_name   VARCHAR(100) DEFAULT 'unknown' COMMENT '识别到的姓名',
                    e_distance    FLOAT        DEFAULT NULL COMMENT '欧氏距离（越小越相似）',
                    camera_id     INT          DEFAULT 0  COMMENT '摄像头编号',
                    snapshot_path VARCHAR(255) DEFAULT '' COMMENT '抓拍图片路径（可选）',
                    recognized_at DATETIME     DEFAULT CURRENT_TIMESTAMP COMMENT '识别时间',
                    INDEX idx_person_id (person_id),
                    INDEX idx_recognized_at (recognized_at),
                    INDEX idx_person_name (person_name)
                ) ENGINE=InnoDB COMMENT='人脸识别记录表';
            """)
        conn.commit()
        logging.info("数据库初始化完成 / DB init done.")
    finally:
        conn.close()


# ============================================================
# persons 表操作 / CRUD for persons
# ============================================================
def add_person(name, employee_id="", department="", folder_path=""):
    """新增人员，返回新记录的 id。"""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO persons (name, employee_id, department, folder_path) "
                "VALUES (%s, %s, %s, %s)",
                (name, employee_id, department, folder_path)
            )
            new_id = cur.lastrowid
        conn.commit()
        return new_id
    except pymysql.err.IntegrityError:
        logging.warning("人员 '%s' 已存在 / Person '%s' already exists.", name, name)
        return None
    finally:
        conn.close()


def update_person(name, employee_id=None, department=None, photo_count=None, folder_path=None):
    """按姓名更新人员信息，只更新传入的非 None 字段。"""
    fields, values = [], []
    if employee_id  is not None: fields.append("employee_id=%s");  values.append(employee_id)
    if department   is not None: fields.append("department=%s");    values.append(department)
    if photo_count  is not None: fields.append("photo_count=%s");   values.append(photo_count)
    if folder_path  is not None: fields.append("folder_path=%s");   values.append(folder_path)
    if not fields:
        return
    values.append(name)
    sql = f"UPDATE persons SET {', '.join(fields)} WHERE name=%s"
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, values)
        conn.commit()
    finally:
        conn.close()


def delete_person(name):
    """删除人员记录（不删除磁盘文件，文件由调用方处理）。"""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM persons WHERE name=%s", (name,))
        conn.commit()
    finally:
        conn.close()


def get_all_persons():
    """返回所有人员列表，每条为 dict。"""
    conn = get_connection()
    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute("SELECT * FROM persons ORDER BY created_at DESC")
            return cur.fetchall()
    finally:
        conn.close()


def get_person_by_name(name):
    """按姓名查询单条人员记录，返回 dict 或 None。"""
    conn = get_connection()
    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute("SELECT * FROM persons WHERE name=%s", (name,))
            return cur.fetchone()
    finally:
        conn.close()


def search_persons(keyword):
    """模糊搜索姓名或工号，返回列表。"""
    conn = get_connection()
    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cur:
            like = f"%{keyword}%"
            cur.execute(
                "SELECT * FROM persons WHERE name LIKE %s OR employee_id LIKE %s "
                "ORDER BY name",
                (like, like)
            )
            return cur.fetchall()
    finally:
        conn.close()


def increment_photo_count(name, delta=1):
    """人脸照片数量 +delta（新增照片时调用）。"""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE persons SET photo_count = photo_count + %s WHERE name=%s",
                (delta, name)
            )
        conn.commit()
    finally:
        conn.close()


# ============================================================
# recognition_logs 表操作 / CRUD for recognition logs
# ============================================================

# 防重复打卡缓存：{person_name: last_log_datetime}
_cooldown_cache: dict = {}
COOLDOWN_SECONDS = 60  # 同一人 60 秒内不重复写入，可按需修改


def add_recognition_log(person_name, e_distance=None, camera_id=0, snapshot_path=""):
    """
    写入一条识别记录。
    - 若同一人在 COOLDOWN_SECONDS 内已写入，则跳过（防重复）。
    - 自动关联 persons.id（若存在）。
    返回：True=写入成功，False=被 cooldown 跳过。
    """
    from datetime import timedelta
    now = datetime.now()

    # cooldown 检查
    last = _cooldown_cache.get(person_name)
    if last and (now - last).total_seconds() < COOLDOWN_SECONDS:
        return False

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            # 查 person_id
            cur.execute("SELECT id FROM persons WHERE name=%s", (person_name,))
            row = cur.fetchone()
            person_id = row[0] if row else None

            cur.execute(
                "INSERT INTO recognition_logs "
                "(person_id, person_name, e_distance, camera_id, snapshot_path) "
                "VALUES (%s, %s, %s, %s, %s)",
                (person_id, person_name, e_distance, camera_id, snapshot_path)
            )
        conn.commit()
        _cooldown_cache[person_name] = now
        return True
    finally:
        conn.close()


def query_logs(person_name=None, start_dt=None, end_dt=None, limit=200):
    """
    查询识别记录。
    - person_name: 精确匹配（None=全部）
    - start_dt / end_dt: datetime 对象，时间范围过滤
    - limit: 最多返回条数
    返回 list[dict]。
    """
    conditions, params = [], []
    if person_name:
        conditions.append("person_name=%s"); params.append(person_name)
    if start_dt:
        conditions.append("recognized_at >= %s"); params.append(start_dt)
    if end_dt:
        conditions.append("recognized_at <= %s"); params.append(end_dt)

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    sql = f"SELECT * FROM recognition_logs {where} ORDER BY recognized_at DESC LIMIT %s"
    params.append(limit)

    conn = get_connection()
    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute(sql, params)
            return cur.fetchall()
    finally:
        conn.close()


def get_log_stats(days=7):
    """
    最近 N 天每人识别次数统计，返回 list[dict]。
    字段：person_name, count, last_seen
    """
    conn = get_connection()
    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute("""
                SELECT person_name,
                       COUNT(*)          AS count,
                       MAX(recognized_at) AS last_seen
                FROM recognition_logs
                WHERE recognized_at >= DATE_SUB(NOW(), INTERVAL %s DAY)
                GROUP BY person_name
                ORDER BY count DESC
            """, (days,))
            return cur.fetchall()
    finally:
        conn.close()


def export_logs_to_csv(filepath, person_name=None, start_dt=None, end_dt=None):
    """将查询结果导出为 CSV 文件。"""
    import csv
    rows = query_logs(person_name=person_name, start_dt=start_dt, end_dt=end_dt, limit=100000)
    if not rows:
        logging.warning("没有符合条件的记录 / No records to export.")
        return 0
    with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    logging.info("导出 %d 条记录到 %s", len(rows), filepath)
    return len(rows)


# ============================================================
# 快速测试 / Quick test
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    init_db()
    print("数据库初始化成功！")
    print("所有人员：", get_all_persons())
