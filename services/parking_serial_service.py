import re
import sqlite3
import threading
import time
from datetime import datetime

import serial


# ============================================================
# VIN PARKING - CENTRAL SERIAL SERVICE
#
# ESP32-S3    /dev/ttyUSB0:
#   EXIT,UID
#   EXIT_DENIED,UID
#   PARKING,0,1,0,1
#
# ESP32-WROOM /dev/ttyUSB1:
#   CARD,UID
#   Pi -> ESP32: RESULT,REGISTERED
#                RESULT,ENTRY_ALLOWED
#                RESULT,ENTRY_DENIED
#                RESULT,ERROR
#
# Registration mode:
#   UID -> registration queue; NEVER opens barrier.
#
# Normal mode:
#   Pi checks authorization, records history, then
#   tells ESP32 whether it may open the barrier.
# ============================================================

S3_PORT = "/dev/ttyUSB0"
WROOM_PORT = "/dev/ttyUSB1"
BAUD_RATE = 115200

# Short timeout lets the WROOM worker respond promptly.
SERIAL_TIMEOUT = 0.2

DB_PATH = "/home/admin/smart_parking/database/parking.db"

db_lock = threading.Lock()

# Preserve the current test cards from the old ESP32 firmware.
# A registered card with an ACTIVE driver is also authorized.
LEGACY_VALID_UIDS = {
    "3E:51:C9:06",
    "BB:99:C9:06",
    "FF:A7:08:E6",
}


def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def is_valid_uid(uid):
    if not isinstance(uid, str):
        return False

    return re.fullmatch(
        r"[0-9A-F]{2}(?::[0-9A-F]{2})+",
        uid.strip().upper()
    ) is not None


def connect_database():
    conn = sqlite3.connect(DB_PATH, timeout=5)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


def check_database(conn):
    conn.execute(
        "SELECT slot_id, status, updated_at FROM parking_slots LIMIT 1"
    )
    conn.execute(
        """
        SELECT id, event_type, slot_id, uid, created_at
        FROM parking_history
        LIMIT 1
        """
    )
    print("[OK] Database tables checked")


def save_history(conn, event_type, uid):
    uid = uid.strip().upper()

    if not is_valid_uid(uid):
        print(f"[ERROR] Invalid UID: {uid}")
        return False

    created_at = get_current_time()

    with db_lock:
        conn.execute(
            """
            INSERT INTO parking_history
                (event_type, slot_id, uid, created_at)
            VALUES (?, NULL, ?, ?)
            """,
            (event_type, uid, created_at)
        )
        conn.commit()

    print(f"[DB] {event_type:<13} UID={uid} TIME={created_at}")
    return True


def update_parking_slots(conn, p1, p2, p3, p4):
    states = {
        "P1": p1,
        "P2": p2,
        "P3": p3,
        "P4": p4,
    }

    if any(value not in (0, 1) for value in states.values()):
        raise ValueError("Parking state must be 0 or 1")

    now = get_current_time()

    try:
        with db_lock:
            for slot_id, status in states.items():
                cursor = conn.execute(
                    """
                    UPDATE parking_slots
                    SET status = ?, updated_at = ?
                    WHERE slot_id = ?
                    """,
                    (status, now, slot_id)
                )

                if cursor.rowcount != 1:
                    raise RuntimeError(
                        f"Parking slot {slot_id} not found"
                    )

            conn.commit()

    except Exception:
        conn.rollback()
        raise

    occupied = [p1, p2, p3, p4].count(0)
    free_slots = [p1, p2, p3, p4].count(1)

    print(
        f"[PARKING] P1={p1} P2={p2} P3={p3} P4={p4} "
        f"| occupied={occupied} free={free_slots}"
    )


# ------------------------------------------------------------
# ESP32-S3: exit events and parking sensors
# ------------------------------------------------------------

def process_s3_line(conn, line):
    line = line.strip()
    if not line:
        return

    command = line.upper()

    if command.startswith("PARKING,"):
        parts = line.split(",")

        if len(parts) != 5:
            print(f"[S3] Invalid PARKING format: {line}")
            return

        try:
            states = [int(value.strip()) for value in parts[1:]]
        except ValueError:
            print(f"[S3] Invalid PARKING numbers: {line}")
            return

        if any(value not in (0, 1) for value in states):
            print(f"[S3] Invalid parking state: {line}")
            return

        update_parking_slots(conn, *states)
        return

    for prefix, event_type in (
        ("EXIT,", "EXIT"),
        ("EXIT_DENIED,", "EXIT_DENIED"),
    ):
        if command.startswith(prefix):
            parts = line.split(",", 1)

            if len(parts) != 2:
                print(f"[S3] Invalid event: {line}")
                return

            uid = parts[1].strip().upper()
            save_history(conn, event_type, uid)
            return


# ------------------------------------------------------------
# Registration state and queue
# ------------------------------------------------------------

def registration_is_active(conn):
    """
    Flask starts a 30-second registration window by updating
    rfid_scan_state.active_until.
    """
    try:
        row = conn.execute(
            """
            SELECT active_until
            FROM rfid_scan_state
            WHERE id = 1
            """
        ).fetchone()
    except sqlite3.Error as exc:
        print(f"[REGISTRATION] Cannot read scan state: {exc}")
        return False

    return bool(row and float(row["active_until"]) > time.time())


def enqueue_registration_uid(conn, uid):
    """
    Store the scanned UID and close the current scan window.
    Flask's /api/rfid/scan/result retrieves this queue item.
    """
    uid = uid.strip().upper()

    if not is_valid_uid(uid):
        return False

    try:
        with db_lock:
            conn.execute("BEGIN IMMEDIATE")

            row = conn.execute(
                """
                SELECT active_until
                FROM rfid_scan_state
                WHERE id = 1
                """
            ).fetchone()

            if not row or float(row["active_until"]) <= time.time():
                conn.rollback()
                return False

            conn.execute(
                """
                INSERT INTO rfid_scan_queue (uid, source, claimed)
                VALUES (?, 'REGISTRATION', 0)
                """,
                (uid,)
            )

            conn.execute(
                """
                UPDATE rfid_scan_state
                SET active_until = 0
                WHERE id = 1
                """
            )

            conn.commit()

        print(f"[REGISTRATION] UID queued: {uid}")
        return True

    except Exception:
        conn.rollback()
        raise


# ------------------------------------------------------------
# Authorization for normal entry
# ------------------------------------------------------------

def get_entry_driver(conn, uid):
    """Trả về tài xế hợp lệ có RFID và dữ liệu khuôn mặt."""
    uid = uid.strip().upper()

    try:
        return conn.execute(
            """
            SELECT
                d.driver_id,
                d.full_name
            FROM rfid_cards AS r
            JOIN drivers AS d
                ON d.driver_id = r.driver_id
            JOIN face_encodings AS f
                ON f.driver_id = d.driver_id
            WHERE UPPER(r.uid) = ?
              AND d.status = 'ACTIVE'
            LIMIT 1
            """,
            (uid,)
        ).fetchone()

    except sqlite3.Error as exc:
        print(f"[AUTH] Database error: {exc}")
        return None



# ------------------------------------------------------------
# ESP32-WROOM: two-way RFID decision protocol
# ------------------------------------------------------------

def send_wroom_result(ser, result):
    message = f"RESULT,{result}\n"
    ser.write(message.encode("utf-8"))
    ser.flush()
    print(f"[WROOM -> ESP32] {message.strip()}")

FACE_AUTH_TIMEOUT = 32.0
FACE_AUTH_POLL_INTERVAL = 0.2


def create_face_auth_request(conn, uid, driver_id):
    """Tạo yêu cầu chờ xác thực khuôn mặt từ giao diện web."""
    now = time.time()

    cursor = conn.execute(
        """
        INSERT INTO face_auth_requests
            (uid, driver_id, state, created_at, message)
        VALUES (?, ?, 'PENDING', ?, 'Đang chờ xác thực khuôn mặt')
        """,
        (uid, driver_id, now)
    )
    conn.commit()

    request_id = cursor.lastrowid
    print(
        f"[FACE AUTH] Request #{request_id}: "
        f"UID={uid}, driver={driver_id}"
    )
    return request_id


def wait_face_auth_result(conn, request_id):
    """Chờ web xác thực; quá hạn hoặc lỗi đều không cấp quyền."""
    deadline = time.monotonic() + FACE_AUTH_TIMEOUT

    while time.monotonic() < deadline:
        row = conn.execute(
            """
            SELECT state, message
            FROM face_auth_requests
            WHERE id = ?
            """,
            (request_id,)
        ).fetchone()

        if row is None:
            return "DENY"

        state = row["state"] if isinstance(row, sqlite3.Row) else row[0]

        if state in ("ALLOW", "DENY", "EXPIRED"):
            return state

        time.sleep(FACE_AUTH_POLL_INTERVAL)

    # Chỉ chuyển sang EXPIRED nếu yêu cầu vẫn còn PENDING.
    conn.execute(
        """
        UPDATE face_auth_requests
        SET state = 'EXPIRED',
            processed_at = ?,
            message = 'Hết thời gian xác thực'
        WHERE id = ? AND state = 'PENDING'
        """,
        (time.time(), request_id)
    )
    conn.commit()

    row = conn.execute(
        "SELECT state FROM face_auth_requests WHERE id = ?",
        (request_id,)
    ).fetchone()

    state = row["state"] if isinstance(row, sqlite3.Row) else row[0]
    return state

def process_wroom_card(conn, ser, uid):
    uid = uid.strip().upper()

    if not is_valid_uid(uid):
        print(f"[WROOM] Invalid UID: {uid}")
        send_wroom_result(ser, "ENTRY_DENIED")
        return

    # Ưu tiên chế độ đăng ký RFID.
    if registration_is_active(conn):
        try:
            if enqueue_registration_uid(conn, uid):
                send_wroom_result(ser, "REGISTERED")
            else:
                # Cửa sổ đăng ký vừa hết hạn: từ chối an toàn.
                save_history(conn, "ENTRY_DENIED", uid)
                send_wroom_result(ser, "ENTRY_DENIED")
        except Exception as exc:
            conn.rollback()
            print(f"[REGISTRATION][ERROR] {exc}")
            send_wroom_result(ser, "ENTRY_DENIED")
        return

    # Bước 1: RFID phải thuộc tài xế ACTIVE có dữ liệu khuôn mặt.
    driver = get_entry_driver(conn, uid)

    if driver is None:
        print(f"[AUTH] RFID không hợp lệ, tài xế không ACTIVE "
              f"hoặc chưa có dữ liệu khuôn mặt: {uid}")
        save_history(conn, "ENTRY_DENIED", uid)
        send_wroom_result(ser, "ENTRY_DENIED")
        return

    driver_id = (
        driver["driver_id"]
        if isinstance(driver, sqlite3.Row)
        else driver[0]
    )

    # Bước 2: tạo yêu cầu để giao diện camera trên laptop xử lý.
    try:
        request_id = create_face_auth_request(
            conn, uid, driver_id
        )
    except Exception as exc:
        conn.rollback()
        print(f"[FACE AUTH][ERROR] Không tạo được yêu cầu: {exc}")
        save_history(conn, "ENTRY_DENIED", uid)
        send_wroom_result(ser, "ENTRY_DENIED")
        return

    # Bước 3: chỉ chờ quyết định từ API xác thực khuôn mặt.
    try:
        result = wait_face_auth_result(conn, request_id)
    except Exception as exc:
        conn.rollback()
        print(f"[FACE AUTH][ERROR] {exc}")
        result = "DENY"

    # Bước 4: fail-closed — chỉ ALLOW mới được vào.
    if result == "ALLOW":
        save_history(conn, "ENTRY", uid)
        send_wroom_result(ser, "ENTRY_ALLOWED")
        print(f"[ACCESS GRANTED] UID={uid}")
    else:
        save_history(conn, "ENTRY_DENIED", uid)
        send_wroom_result(ser, "ENTRY_DENIED")
        print(f"[ACCESS DENIED] UID={uid}, result={result}")


def process_wroom_line(conn, ser, line):
    line = line.strip()

    if not line:
        return

    print(f"[WROOM] {line}")

    command = line.upper()

    # New firmware sends CARD,UID and waits for RESULT,...
    if command.startswith("CARD,"):
        parts = line.split(",", 1)

        if len(parts) != 2:
            send_wroom_result(ser, "ERROR")
            return

        process_wroom_card(conn, ser, parts[1])
        return

    # Compatibility logging for old firmware messages.
    # These old messages arrive AFTER the old ESP32 opens the barrier.
    # They cannot provide safe registration-mode behavior.
    for prefix, event_type in (
        ("ENTRY,", "ENTRY"),
        ("ENTRY_DENIED,", "ENTRY_DENIED"),
    ):
        if command.startswith(prefix):
            parts = line.split(",", 1)

            if len(parts) == 2:
                save_history(conn, event_type, parts[1].strip().upper())

            return


# ------------------------------------------------------------
# Serial workers
# ------------------------------------------------------------

def s3_worker():
    ser = None
    conn = None

    try:
        print(f"[S3] Opening {S3_PORT}...")

        ser = serial.Serial(
            port=S3_PORT,
            baudrate=BAUD_RATE,
            timeout=SERIAL_TIMEOUT
        )

        conn = connect_database()
        print(f"[S3] Connected: {S3_PORT}")

        while True:
            raw = ser.readline()

            if not raw:
                continue

            line = raw.decode("utf-8", errors="ignore").strip()
            if not line:
                continue

            print(f"[S3] {line}")

            try:
                process_s3_line(conn, line)
            except Exception as exc:
                conn.rollback()
                print(f"[S3][ERROR] {exc}")

    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"[S3][FATAL] {exc}")
    finally:
        if ser is not None:
            ser.close()
        if conn is not None:
            conn.close()
        print("[S3] STOPPED")


def wroom_worker():
    ser = None
    conn = None

    try:
        print(f"[WROOM] Opening {WROOM_PORT}...")

        ser = serial.Serial(
            port=WROOM_PORT,
            baudrate=BAUD_RATE,
            timeout=SERIAL_TIMEOUT,
            write_timeout=2
        )

        conn = connect_database()
        print(f"[WROOM] Connected: {WROOM_PORT}")

        while True:
            raw = ser.readline()

            if not raw:
                continue

            line = raw.decode("utf-8", errors="ignore").strip()
            if not line:
                continue

            try:
                process_wroom_line(conn, ser, line)
            except Exception as exc:
                conn.rollback()
                print(f"[WROOM][ERROR] {exc}")

    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"[WROOM][FATAL] {exc}")
    finally:
        if ser is not None:
            ser.close()
        if conn is not None:
            conn.close()
        print("[WROOM] STOPPED")


def main():
    print("=" * 60)
    print("VIN PARKING - CENTRAL SERIAL SERVICE")
    print(f"S3    : {S3_PORT}")
    print(f"WROOM : {WROOM_PORT}")
    print(f"BAUD  : {BAUD_RATE}")
    print(f"DB    : {DB_PATH}")
    print("=" * 60)

    conn = None

    try:
        conn = connect_database()
        check_database(conn)
        print("[OK] Database ready")
    except Exception as exc:
        print(f"[FATAL] Database not ready: {exc}")
        return
    finally:
        if conn is not None:
            conn.close()

    threads = [
        threading.Thread(target=s3_worker, name="S3Worker", daemon=True),
        threading.Thread(target=wroom_worker, name="WROOMWorker", daemon=True),
    ]

    for thread in threads:
        thread.start()

    print("[OK] Serial workers started. Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping central serial service...")


if __name__ == "__main__":
    main()