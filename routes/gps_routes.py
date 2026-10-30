import sqlite3
from flask import Blueprint, current_app, jsonify, request, render_template
from services.app_context import get_db, json_error
from services.auth_utils import login_required

gps_bp = Blueprint("gps_api", __name__)


@gps_bp.route("/gps")
@login_required
def gps_page():
    """Giao diện xem bản đồ GPS"""
    return render_template("gps.html")


@gps_bp.route("/api/gps/update", methods=["POST"])
def update_gps():
    """API nhận dữ liệu từ ESP32-S3 gửi lên"""
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return json_error("Dữ liệu JSON không hợp lệ.")

    try:
        raw_lat = data.get("latitude") if "latitude" in data else data.get("lat")
        raw_lng = data.get("longitude") if "longitude" in data else data.get("lng")

        if raw_lat is None or raw_lng is None:
            return json_error("Thiếu tọa độ GPS.")

        lat = float(raw_lat)
        lng = float(raw_lng)
        speed = float(data.get("speed", 0.0))
        sats = int(data.get("satellites", data.get("sats", 0)))
    except (TypeError, ValueError):
        return json_error("Dữ liệu số không hợp lệ.")

    conn = get_db()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS gps_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                speed REAL NOT NULL,
                satellites INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
            )
        """)
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("""
            INSERT INTO gps_logs (latitude, longitude, speed, satellites)
            VALUES (?, ?, ?, ?)
        """, (lat, lng, speed, sats))
        conn.commit()

        return jsonify({"success": True, "message": "Đã lưu tọa độ GPS."})
    except sqlite3.Error:
        conn.rollback()
        current_app.logger.exception("Lỗi lưu GPS vào database")
        return json_error("Lỗi cơ sở dữ liệu.", 500)
    finally:
        conn.close()


@gps_bp.route("/api/gps/latest")
@login_required
def get_latest_gps():
    """API cho file HTML lấy tọa độ mới nhất cập nhật bản đồ"""
    conn = get_db()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS gps_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                speed REAL NOT NULL,
                satellites INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
            )
        """)
        row = conn.execute("""
            SELECT latitude, longitude, speed, satellites, created_at
            FROM gps_logs
            ORDER BY id DESC
            LIMIT 1
        """).fetchone()

        if not row:
            return jsonify({"success": True, "has_data": False})

        return jsonify({
            "success": True,
            "has_data": True,
            "latitude": row["latitude"],
            "longitude": row["longitude"],
            "speed": row["speed"],
            "satellites": row["satellites"],
            "updated_at": row["created_at"]
        })
    except sqlite3.Error:
        current_app.logger.exception("Lỗi đọc dữ liệu GPS")
        return json_error("Không thể đọc GPS.", 500)
    finally:
        conn.close()
