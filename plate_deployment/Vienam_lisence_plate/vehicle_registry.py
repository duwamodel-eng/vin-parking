"""
vehicle_registry.py
Danh sach xe dang ky va doi chieu bien so OCR cho Vin Parking.

Quy tac A/4:
- Chi xem xet ky tu tai vi tri thu 3 (index 2) cua bien so dang ky
  dang 8 ky tu sau khi chuan hoa, vi du: 65A99999.
- Chi sua 4 -> A tai vi tri nay khi ket qua sau hieu chinh khop DUY NHAT
  voi mot bien so trong VEHICLES.
- Ket qua hieu chinh chi la ung vien, khong duoc xem la xac minh danh tinh.
"""


VEHICLES = {
    "76A99999": {
        "car_id": "CAR01",
        "driver": "Tri",
    },
    "59A66666": {
        "car_id": "CAR02",
        "driver": "Khanh",
    },
    "65A99999": {
        "car_id": "CAR03",
        "driver": "Truong",
    },
    "64A88888": {
        "car_id": "CAR04",
        "driver": "Huy",
    },
}

# Vi tri chu cai trong dang bien so 2 so + 1 chu + 5 so.
LETTER_POSITION = 2
EXPECTED_LENGTH = 8


def normalize_plate(text):
    """Chuan hoa OCR: viet hoa va loai bo dau gach, cham, khoang trang."""
    if text is None:
        return ""
    return "".join(char for char in str(text).upper() if char.isalnum())


def find_vehicle(ocr_text):
    """
    Tra ve thong tin xe neu bien so khop.

    match_type:
      - EXACT: OCR khop nguyen ven voi bien dang ky.
      - CORRECTED_A4: chi co the khop sau khi doi ky tu 4 tai vi tri
        chu cai thu 3 thanh A. Can xac minh them.
    """
    plate = normalize_plate(ocr_text)

    if not plate:
        return None

    # 1. Uu tien khop chinh xac.
    if plate in VEHICLES:
        return {
            **VEHICLES[plate],
            "matched_plate": plate,
            "ocr_plate": plate,
            "match_type": "EXACT",
        }

    # 2. Chi thu sua dung vi tri chu cai thu 3.
    if len(plate) != EXPECTED_LENGTH:
        return None

    if plate[LETTER_POSITION] != "4":
        return None

    candidate = plate[:LETTER_POSITION] + "A" + plate[LETTER_POSITION + 1:]

    if candidate in VEHICLES:
        return {
            **VEHICLES[candidate],
            "matched_plate": candidate,
            "ocr_plate": plate,
            "match_type": "CORRECTED_A4",
        }

    return None


if __name__ == "__main__":
    test_plates = [
        "76A-999.99",
        "59A-666.66",
        "65A-999.99",
        "64A-888.88",
        "654-999.99",   # Ky vong sua thanh 65A99999
        "644-888.88",   # Ky vong sua thanh 64A88888
        "764-999.99",   # Ky vong sua thanh 76A99999
        "654A-999.99",  # Sai do dai, khong tu y sua
        "654-888.88",   # Khong co bien so tuong ung
    ]

    for raw_text in test_plates:
        vehicle = find_vehicle(raw_text)
        print(f"\nOCR text: {raw_text}")

        if vehicle is None:
            print("Trang thai: KHONG KHOP BIEN SO DA DANG KY")
            continue

        print("Ma xe:", vehicle["car_id"])
        print("Tai xe:", vehicle["driver"])
        print("OCR goc:", vehicle["ocr_plate"])
        print("Bien so doi chieu:", vehicle["matched_plate"])
        print("Trang thai:", vehicle["match_type"])