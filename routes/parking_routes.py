
import sqlite3

from flask import Blueprint, current_app, jsonify, request

from services.app_context import EVENT_LABELS, get_db, json_error
from services.auth_utils import login_required


parking_bp = Blueprint("parking_api", __name__)


@parking_bp.route("/api/status")
@login_required
def status():
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT slot_id, status, updated_at "
            "FROM parking_slots ORDER BY slot_id"
        ).fetchall()
    finally:
        conn.close()

    slots = [
        {
            "slot_id": row["slot_id"],
            "status": row["status"],
            "updated_at": row["updated_at"],
        }
        for row in rows
    ]
    occupied = sum(1 for slot in slots if slot["status"] == 0)

    return jsonify({
        "total": len(slots),
        "occupied": occupied,
        "free": len(slots) - occupied,
        "slots": slots,
    })


@parking_bp.route("/api/history")
@login_required
def history():
    try:
        page = max(1, int(request.args.get("page", 1)))
        limit = min(100, max(1, int(request.args.get("limit", 20))))
    except (TypeError, ValueError):
        return json_error("page/limit không hợp lệ.")

    event_type = request.args.get("event_type", "").strip().upper()
    if event_type and event_type not in EVENT_LABELS:
        return json_error("Loại sự kiện không hợp lệ.")

    where = "WHERE h.event_type = ?" if event_type else ""
    params = [event_type] if event_type else []
    conn = get_db()

    try:
        total = conn.execute(
            f"SELECT COUNT(*) FROM parking_history AS h {where}",
            params,
        ).fetchone()[0]

        rows = conn.execute(f"""
            SELECT h.id, h.event_type, h.slot_id, h.uid, h.created_at,
                   d.driver_id, d.full_name
            FROM parking_history AS h
            LEFT JOIN rfid_cards AS r ON UPPER(r.uid) = UPPER(h.uid)
            LEFT JOIN drivers AS d ON d.driver_id = r.driver_id
            {where}
            ORDER BY h.id DESC
            LIMIT ? OFFSET ?
        """, params + [limit, (page - 1) * limit]).fetchall()
    finally:
        conn.close()

    events = [
        {
            "id": row["id"],
            "event_type": row["event_type"],
            "label": EVENT_LABELS.get(
                row["event_type"], row["event_type"]
            ),
            "slot_id": row["slot_id"],
            "uid": row["uid"],
            "created_at": row["created_at"],
            "driver_id": row["driver_id"],
            "full_name": row["full_name"],
        }
        for row in rows
    ]

    return jsonify({
        "events": events,
        "total": total,
        "page": page,
        "limit": limit,
    })


@parking_bp.route("/api/parking/update", methods=["POST"])
@login_required
def update_parking():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return json_error("JSON không hợp lệ.")

    required = ("P1", "P2", "P3", "P4")
    for slot in required:
        if (
            slot not in data
            or type(data[slot]) is not int
            or data[slot] not in (0, 1)
        ):
            return json_error(f"{slot} phải là số nguyên 0 hoặc 1.")

    conn = get_db()
    try:
        conn.execute("BEGIN IMMEDIATE")

        for slot in required:
            cur = conn.execute(
                """
                UPDATE parking_slots
                SET status = ?,
                    updated_at = datetime('now', 'localtime')
                WHERE slot_id = ?
                """,
                (data[slot], slot),
            )

            if cur.rowcount == 0:
                conn.rollback()
                return json_error(
                    f"Không tìm thấy vị trí {slot} trong cơ sở dữ liệu.",
                    404,
                )

        conn.commit()
        return jsonify({
            "success": True,
            "message": "Cập nhật thành công",
            **{slot: data[slot] for slot in required},
        })

    except sqlite3.Error:
        conn.rollback()
        current_app.logger.exception("Lỗi cập nhật trạng thái bãi xe")
        return json_error("Không thể cập nhật trạng thái bãi xe.", 500)
    finally:
        conn.close()
