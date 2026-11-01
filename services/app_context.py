
from pathlib import Path
import os
import sqlite3

from flask import jsonify

BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE = BASE_DIR / "database" / "parking.db"

EVENT_LABELS = {
    "ENTRY": "Vào bãi thành công",
    "EXIT": "Ra bãi thành công",
    "ENTRY_DENIED": "Từ chối vào",
    "EXIT_DENIED": "Từ chối ra",
}

# Token đăng ký khuôn mặt tạm thời, hết hạn sau 5 phút.
PENDING_FACES = {}


def get_db():
    """Mở SQLite hiện có với thời gian chờ khóa thống nhất."""
    if not DATABASE.exists():
        raise FileNotFoundError(
            f"Không tìm thấy cơ sở dữ liệu: {DATABASE}"
        )

    conn = sqlite3.connect(
        str(DATABASE),
        timeout=10,
        check_same_thread=True
    )

    conn.row_factory = sqlite3.Row

    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 10000")

    return conn


def json_error(message, status=400):
    return jsonify({"success": False, "message": message}), status


def normalize_phone(value):
    import re

    phone = re.sub(r"[\s().-]", "", str(value or ""))
    if phone.startswith("+84"):
        phone = "0" + phone[3:]
    if not re.fullmatch(r"0\d{9}", phone):
        return None
    return phone


def valid_full_name(value):
    import re

    name = str(value or "").strip()
    if not name or len(name) > 100 or re.search(r"\d", name):
        return None
    if not re.search(r"[^\W\d_]", name, re.UNICODE):
        return None
    return name



FIXED_PLATES = (
    "76A99999",
    "59A66666",
    "65A99999",
    "64A88888",
)


def init_vehicles():
    """
    Tạo bảng quản lý biển số nếu chưa tồn tại.
    Giữ nguyên bốn biển số kể cả khi xóa hồ sơ tài xế.
    Không xóa hoặc ghi đè dữ liệu cũ.
    """
    conn = get_db()

    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS vehicles (
                plate_number TEXT PRIMARY KEY,
                driver_id TEXT UNIQUE,
                created_at TEXT NOT NULL
                    DEFAULT (datetime('now', 'localtime')),

                FOREIGN KEY (driver_id)
                    REFERENCES drivers(driver_id)
                    ON DELETE SET NULL
            )
        """)

        for plate in FIXED_PLATES:
            conn.execute("""
                INSERT OR IGNORE INTO vehicles (plate_number)
                VALUES (?)
            """, (plate,))

        conn.commit()

    except sqlite3.Error:
        conn.rollback()
        raise

    finally:
        conn.close()
