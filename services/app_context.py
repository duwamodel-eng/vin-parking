
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
    """Mở SQLite hiện có, không tự tạo lại database."""
    if not DATABASE.exists():
        raise FileNotFoundError(f"Không tìm thấy cơ sở dữ liệu: {DATABASE}")

    conn = sqlite3.connect(str(DATABASE), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
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
