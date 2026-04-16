# main.py — 项目启动入口
# 原来直接运行 app.py，现在改为运行 main.py
import logging
from app.db_manager import init_db
from app.routes import app, sync_persons_from_filesystem

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    init_db()
    n = sync_persons_from_filesystem()
    if n > 0:
        logging.info("启动时同步了 %d 位人员到数据库", n)
    print("\n" + "═" * 52)
    print("  人脸识别系统已启动")
    print("  浏览器访问 → http://127.0.0.1:5000")
    print("═" * 52 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
