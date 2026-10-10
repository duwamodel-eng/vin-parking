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

def is_authorized_for_entry(conn, uid):
    """
    Preserve the three existing test UIDs.
    Also allow a UID linked to an ACTIVE driver in the database.
    """
    uid = uid.strip().upper()

    if uid in LEGACY_VALID_UIDS:
        return True

    try:
        row = conn.execute(
            """
            SELECT 1
            FROM rfid_cards AS card
            JOIN drivers AS driver
              ON driver.driver_id = card.driver_id
            WHERE UPPER(card.uid) = ?
              AND driver.status = 'ACTIVE'
            LIMIT 1
            """,
            (uid,)
        ).fetchone()

        return row is not None

    except sqlite3.Error as exc:
        # Fail closed if authorization data cannot be checked.
        print(f"[AUTH] Database authorization error: {exc}")
        return False


# ------------------------------------------------------------
# ESP32-WROOM: two-way RFID decision protocol
# ------------------------------------------------------------

def send_wroom_result(ser, result):
    message = f"RESULT,{result}\n"
    ser.write(message.encode("utf-8"))
    ser.flush()
    print(f"[WROOM -> ESP32] {message.strip()}")


def process_wroom_card(conn, ser, uid):
    uid = uid.strip().upper()

    if not is_valid_uid(uid):
        print(f"[WROOM] Invalid UID: {uid}")
        send_wroom_result(ser, "ERROR")
        return

    # PRIORITY 1: registration.
    # Do not record an ENTRY event and do not authorize opening.
    if registration_is_active(conn):
        try:
            if enqueue_registration_uid(conn, uid):
                send_wroom_result(ser, "REGISTERED")
            else:
                # The registration window may have expired between checks.
                if is_authorized_for_entry(conn, uid):
                    save_history(conn, "ENTRY", uid)
                    send_wroom_result(ser, "ENTRY_ALLOWED")
                else:
                    save_history(conn, "ENTRY_DENIED", uid)
                    send_wroom_result(ser, "ENTRY_DENIED")
        except Exception as exc:
            print(f"[REGISTRATION][ERROR] {exc}")
            send_wroom_result(ser, "ERROR")

        return

    # PRIORITY 2: normal entry.
    if is_authorized_for_entry(conn, uid):
        save_history(conn, "ENTRY", uid)
        send_wroom_result(ser, "ENTRY_ALLOWED")
    else:
        save_history(conn, "ENTRY_DENIED", uid)
        send_wroom_result(ser, "ENTRY_DENIED")


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