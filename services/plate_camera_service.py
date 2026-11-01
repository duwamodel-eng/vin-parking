import os
os.environ["OMP_THREAD_LIMIT"] = "1"

import cv2
import threading
import time
import logging
import re
from pathlib import Path
from collections import deque

import numpy as np
import onnxruntime as ort
from fast_plate_ocr import LicensePlateRecognizer


logger = logging.getLogger(__name__)

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "plate_deployment"
    / "Vienam_lisence_plate"
    / "runs"
    / "detect"
    / "train"
    / "weights"
    / "best.onnx"
)

# Vietnamese License Plate OCR - CCT ONNX
OCR_MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "plate_deployment"
    / "vn_ocr_model"
    / "cct_s_v1_vn.onnx"
)

OCR_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent
    / "plate_deployment"
    / "vn_ocr_model"
    / "cct_s_v1_vn_plate_config.yaml"
)

OCR_MIN_PROBABILITY = 0.70

_plate_ocr_model = None
_plate_ocr_model_lock = threading.Lock()
_plate_ocr_inference_lock = threading.Lock()


def get_plate_ocr_model():
    global _plate_ocr_model

    if _plate_ocr_model is None:
        with _plate_ocr_model_lock:
            if _plate_ocr_model is None:

                if not OCR_MODEL_PATH.is_file():
                    raise FileNotFoundError(
                        f"Không tìm thấy OCR model: {OCR_MODEL_PATH}"
                    )

                if not OCR_CONFIG_PATH.is_file():
                    raise FileNotFoundError(
                        f"Không tìm thấy OCR config: {OCR_CONFIG_PATH}"
                    )

                logger.info("Loading Vietnamese CCT OCR model...")

                _plate_ocr_model = LicensePlateRecognizer(
                    onnx_model_path=str(OCR_MODEL_PATH),
                    plate_config_path=str(OCR_CONFIG_PATH),
                    device="cpu",
                )

                logger.info("Vietnamese CCT OCR model loaded.")

    return _plate_ocr_model

CAMERA_CONFIG = {
    "ENTRY": {
        "device": "/dev/video0",
        "width": 1280,
        "height": 720,
        "fps": 15,
    },
    "EXIT": {
        "device": "/dev/video2",
        "width": 1280,
        "height": 720,
        "fps": 15,
    },
}

INPUT_SIZE = 640
CONF_THRESHOLD = 0.25
NMS_THRESHOLD = 0.45
INFERENCE_INTERVAL = 0.25
OCR_INTERVAL = 0.8
OCR_HISTORY_SIZE = 7
OCR_REQUIRED_VOTES = 3
OCR_MIN_CONFIDENCE = 35.0
OCR_MIN_LENGTH = 7
OCR_MAX_LENGTH = 10

class PlateDetector:
    """Nạp YOLO ONNX một lần và dùng chung cho hai camera."""

    def __init__(self, model_path):
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Không tìm thấy model ONNX: {model_path}"
            )

        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        self.inference_lock = threading.Lock()

        input_shape = self.session.get_inputs()[0].shape

        logger.info(
            "YOLO ONNX loaded: %s | input=%s",
            model_path,
            input_shape,
        )

    @staticmethod
    def letterbox(image, size=640):
        height, width = image.shape[:2]

        scale = min(size / width, size / height)

        new_width = int(round(width * scale))
        new_height = int(round(height * scale))

        resized = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_LINEAR,
        )

        pad_x = (size - new_width) // 2
        pad_y = (size - new_height) // 2

        canvas = np.full(
            (size, size, 3),
            114,
            dtype=np.uint8,
        )

        canvas[
            pad_y:pad_y + new_height,
            pad_x:pad_x + new_width,
        ] = resized

        return canvas, scale, pad_x, pad_y

    def detect(self, frame):
        image, scale, pad_x, pad_y = self.letterbox(
            frame,
            INPUT_SIZE,
        )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        tensor = image.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)
        tensor = np.ascontiguousarray(tensor)

        # Dùng khóa chung để tránh hai camera chạy inference
        # đồng thời làm quá tải CPU Raspberry Pi.
        with self.inference_lock:
            output = self.session.run(
                [self.output_name],
                {self.input_name: tensor},
            )[0]

        predictions = np.asarray(output)

        if predictions.ndim != 3 or predictions.shape[0] != 1:
            raise ValueError(
                f"Output YOLO không đúng định dạng: {predictions.shape}"
            )

        predictions = predictions[0]

        # Model hiện tại có output [1, 5, 8400].
        # Nếu output đã là [1, 8400, 5], không transpose.
        if predictions.shape[0] == 5:
            predictions = predictions.T
        elif predictions.shape[1] != 5:
            raise ValueError(
                f"Không nhận diện được cấu trúc output: {output.shape}"
            )

        predictions = predictions.astype(np.float32)

        if not np.isfinite(predictions).all():
            raise ValueError("Output YOLO chứa NaN hoặc Infinity.")

        boxes = []
        scores = []
        original_boxes = []

        frame_height, frame_width = frame.shape[:2]

        for prediction in predictions:
            cx, cy, width, height, confidence = prediction[:5]

            if confidence < CONF_THRESHOLD:
                continue

            # YOLOv8 một lớp: confidence chính là điểm lớp.
            x1 = (cx - width / 2 - pad_x) / scale
            y1 = (cy - height / 2 - pad_y) / scale
            x2 = (cx + width / 2 - pad_x) / scale
            y2 = (cy + height / 2 - pad_y) / scale

            x1 = int(np.clip(x1, 0, frame_width - 1))
            y1 = int(np.clip(y1, 0, frame_height - 1))
            x2 = int(np.clip(x2, 0, frame_width - 1))
            y2 = int(np.clip(y2, 0, frame_height - 1))

            box_width = x2 - x1
            box_height = y2 - y1

            if box_width < 2 or box_height < 2:
                continue

            boxes.append([
                x1,
                y1,
                box_width,
                box_height,
            ])
            scores.append(float(confidence))
            original_boxes.append((x1, y1, x2, y2))

        if not boxes:
            return []

        indices = cv2.dnn.NMSBoxes(
            boxes,
            scores,
            CONF_THRESHOLD,
            NMS_THRESHOLD,
        )

        if len(indices) == 0:
            return []

        indices = np.asarray(indices).reshape(-1)

        detections = []

        for index in indices:
            x1, y1, x2, y2 = original_boxes[int(index)]

            detections.append({
                "box": (x1, y1, x2, y2),
                "confidence": scores[int(index)],
            })

        return detections


_detector = None
_detector_lock = threading.Lock()

# Chỉ cho phép một camera thực hiện một đợt xác thực tại một thời điểm.
_plate_capture_lock = threading.Lock()


def get_detector():
    global _detector

    if _detector is None:
        with _detector_lock:
            if _detector is None:
                _detector = PlateDetector(MODEL_PATH)

    return _detector

def normalize_plate_text(text):
    """
    Chuẩn hóa kết quả OCR:
    - Chuyển chữ hoa.
    - Bỏ dấu cách, dấu chấm, dấu gạch ngang.
    - Chỉ giữ chữ và số.
    """
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def read_plate_ocr(plate_image):
    """
    Nhận diện biển số Việt Nam bằng CCT ONNX.
    Trả về text đã chuẩn hóa và confidence theo phần trăm.
    """

    if plate_image is None or plate_image.size == 0:
        return None

    height, width = plate_image.shape[:2]

    if width < 60 or height < 20:
        return None

    try:
        recognizer = get_plate_ocr_model()

        # fast-plate-ocr yêu cầu RGB theo cấu hình model đã thử nghiệm.
        image_rgb = cv2.cvtColor(
            plate_image,
            cv2.COLOR_BGR2RGB,
        )

        # Không cho hai camera gọi cùng model đồng thời.
        with _plate_ocr_inference_lock:
            predictions = recognizer.run(
                image_rgb,
                return_confidence=True,
            )

        if not predictions:
            return None

        prediction = predictions[0]

        raw_text = prediction.plate or ""
        plate_text = normalize_plate_text(raw_text)

        if not (
            OCR_MIN_LENGTH
            <= len(plate_text)
            <= OCR_MAX_LENGTH
        ):
            logger.warning(
                "[CCT OCR] Độ dài biển số không hợp lệ: %s",
                plate_text,
            )
            return None

        digit_count = sum(
            char.isdigit() for char in plate_text
        )

        letter_count = sum(
            char.isalpha() for char in plate_text
        )

        if digit_count < 2 or letter_count < 1:
            logger.warning(
                "[CCT OCR] Cấu trúc ký tự không hợp lệ: %s",
                plate_text,
            )
            return None

        char_probs = prediction.char_probs

        if char_probs is None:
            logger.warning(
                "[CCT OCR] Model không trả confidence: %s",
                plate_text,
            )
            return None

        probabilities = np.asarray(
            char_probs,
            dtype=np.float32,
        ).reshape(-1)

        if (
            probabilities.size == 0
            or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0)
            or np.any(probabilities > 1)
        ):
            logger.warning(
                "[CCT OCR] Confidence không hợp lệ: %s",
                plate_text,
            )
            return None

        average_probability = float(
            np.mean(probabilities)
        )

        if average_probability < OCR_MIN_PROBABILITY:
            logger.info(
                "[CCT OCR] Confidence thấp: plate=%s probability=%.4f",
                plate_text,
                average_probability,
            )
            return None

        confidence = average_probability * 100.0

        logger.info(
            "[CCT OCR] plate=%s confidence=%.2f%%",
            plate_text,
            confidence,
        )

        return {
            "text": plate_text,
            "confidence": confidence,
        }

    except Exception:
        logger.exception("Lỗi CCT OCR")
        return None


class CameraWorker:
    def __init__(self, gate, config):
        self.gate = gate
        self.config = config

        self.lock = threading.Lock()
        self.frame_jpeg = None

        # Lưu tối đa 5 frame gốc gần nhất để xác thực OCR.
        # Không mở camera lần thứ hai khi chụp biển số.
        self.raw_frames = deque(maxlen=5)

        self.running = False        
        self.thread = None
        self.error = None

        self.detections = []
        self.last_inference = 0.0
        self.detector_error_logged = False

        self.last_ocr_time = 0.0
        self.last_detection_time = 0.0

        self.ocr_history = deque(maxlen=OCR_HISTORY_SIZE)

        self.stable_plate = None
        self.stable_plate_confidence = 0.0

        self.current_ocr_candidate = None
        self.current_ocr_confidence = 0.0        

    def start(self):
        # Không mở camera nếu worker hiện tại vẫn đang chạy
        # hoặc thread cũ chưa kết thúc hoàn toàn.
        if self.running:
            return

        if self.thread and self.thread.is_alive():
            return

        with self.lock:
            self.frame_jpeg = None
            self.raw_frames.clear()

        self.error = None
        self.detections = []
        self.last_inference = 0.0
        self.last_ocr_time = 0.0
        self.last_detection_time = 0.0

        self.running = True

        self.thread = threading.Thread(
            target=self._capture_loop,
            name=f"camera-{self.gate.lower()}",
            daemon=True,
        )

        self.thread.start()

    def _run_ocr(self, frame):
        """
        Chọn vùng biển số có confidence YOLO cao nhất
        rồi đọc bằng Tesseract.
        """

        if not self.detections:
            return

        detection = max(
            self.detections,
            key=lambda item: item["confidence"],
        )

        x1, y1, x2, y2 = detection["box"]

        frame_height, frame_width = frame.shape[:2]

        # Mở rộng vùng crop một chút để tránh cắt mất ký tự.
        margin_x = int((x2 - x1) * 0.04)
        margin_y = int((y2 - y1) * 0.08)

        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(frame_width, x2 + margin_x)
        y2 = min(frame_height, y2 + margin_y)

        plate_crop = frame[y1:y2, x1:x2]

        try:
            result = read_plate_ocr(plate_crop)

        except Exception:
            logger.exception(
                "OCR camera %s gặp lỗi",
                self.gate,
            )
            return

        if result is None:
            self.current_ocr_candidate = None
            self.current_ocr_confidence = 0.0
            return

        candidate = result["text"]
        confidence = result["confidence"]

        self.current_ocr_candidate = candidate
        self.current_ocr_confidence = confidence

        self.ocr_history.append(candidate)

        # Tìm kết quả xuất hiện nhiều nhất trong lịch sử.
        counts = {}

        for plate in self.ocr_history:
            counts[plate] = counts.get(plate, 0) + 1

        best_plate = max(
            counts,
            key=counts.get,
        )

        best_votes = counts[best_plate]

        if best_votes >= OCR_REQUIRED_VOTES:
            self.stable_plate = best_plate
            self.stable_plate_confidence = confidence            

    def _reset_old_ocr(self, now):
        """
        Xóa kết quả cũ khi camera không còn thấy biển số.
        """

        if (
            self.last_detection_time > 0
            and now - self.last_detection_time > 4
        ):
            self.ocr_history.clear()
            self.stable_plate = None
            self.stable_plate_confidence = 0.0
            self.current_ocr_candidate = None
            self.current_ocr_confidence = 0.0

    def _draw_detections(self, frame):
        for detection in self.detections:
            x1, y1, x2, y2 = detection["box"]
            confidence = detection["confidence"]

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            label = f"LICENSE PLATE {confidence:.2f}"

            cv2.putText(
                frame,
                label,
                (x1, max(25, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        cv2.putText(
            frame,
            f"{self.gate} | YOLO ONNX",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if self.stable_plate:
            ocr_label = (
                f"PLATE: {self.stable_plate} "
                f"({self.stable_plate_confidence:.0f}%)"
            )
            color = (0, 255, 0)

        elif self.current_ocr_candidate:
            ocr_label = (
                f"OCR: {self.current_ocr_candidate} "
                f"({self.current_ocr_confidence:.0f}%)"
            )
            color = (0, 200, 255)

        else:
            ocr_label = "OCR: Waiting for clear plate"

            color = (0, 200, 255)

        cv2.putText(
            frame,
            ocr_label,
            (15, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2,
            cv2.LINE_AA,
        )

    def _capture_loop(self):
        cap = None

        try:
            cap = cv2.VideoCapture(
                self.config["device"],
                cv2.CAP_V4L2,
            )

            if not cap.isOpened():
                self.error = (
                    f"Không mở được camera {self.gate}: "
                    f"{self.config['device']}"
                )
                logger.error(self.error)
                return

            cap.set(
                cv2.CAP_PROP_FOURCC,
                cv2.VideoWriter_fourcc(*"MJPG"),
            )
            cap.set(
                cv2.CAP_PROP_FRAME_WIDTH,
                self.config["width"],
            )
            cap.set(
                cv2.CAP_PROP_FRAME_HEIGHT,
                self.config["height"],
            )
            cap.set(
                cv2.CAP_PROP_FPS,
                self.config["fps"],
            )
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            self.error = None

            logger.info(
                "Camera %s started: %s",
                self.gate,
                self.config["device"],
            )

            while self.running:
                ok, frame = cap.read()

                if not ok or frame is None:
                    self.error = f"Mất tín hiệu camera {self.gate}"
                    time.sleep(0.1)
                    continue

                now = time.monotonic()

                # Lưu frame gốc, không có chữ/bbox overlay.
                with self.lock:
                    self.raw_frames.append((now, frame.copy()))

                if now - self.last_inference >= INFERENCE_INTERVAL:
                    try:
                        detector = get_detector()

                        self.detections = detector.detect(frame)
                        self.last_inference = now
                        self.detector_error_logged = False

                        if self.detections:
                            self.last_detection_time = now

                    except Exception as exc:
                        self.error = f"YOLO {self.gate}: {exc}"

                        if not self.detector_error_logged:
                            logger.exception(
                                "Lỗi YOLO camera %s",
                                self.gate,
                            )
                            self.detector_error_logged = True

                # OCR chạy định kỳ, không chạy ở mọi frame.
                if (
                    self.detections
                    and now - self.last_ocr_time >= OCR_INTERVAL
                ):
                    self.last_ocr_time = now
                    self._run_ocr(frame)

                self._reset_old_ocr(now)

                self._draw_detections(frame)

                success, encoded = cv2.imencode(
                    ".jpg",
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 80],
                )

                if not success:
                    continue

                with self.lock:
                    self.frame_jpeg = encoded.tobytes()

                if self.error and not self.detector_error_logged:
                    self.error = None

        except Exception as exc:
            self.error = str(exc)
            logger.exception("Camera %s gặp lỗi", self.gate)

        finally:
            if cap is not None:
                cap.release()

            self.running = False
            logger.info("Camera %s stopped", self.gate)

    def get_frame(self):
        with self.lock:
            return self.frame_jpeg

    def get_raw_frames_since(self, timestamp):
        """
        Lấy các frame gốc được camera ghi nhận sau timestamp.
        Chỉ trả về frame mới, không lấy lại frame cũ.
        """
        with self.lock:
            return [
                frame.copy()
                for captured_at, frame in self.raw_frames
                if captured_at > timestamp
            ]

    def stop(self):
        self.running = False

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)

            if self.thread.is_alive():
                logger.error(
                    "Camera %s chưa dừng hoàn toàn sau 5 giây.",
                    self.gate,
                )

class PlateCameraManager:
    def __init__(self):
        self.lock = threading.Lock()

        self.cameras = {
            gate: CameraWorker(gate, config)
            for gate, config in CAMERA_CONFIG.items()
        }

    def start_camera(self, gate):
        with self.lock:
            camera = self.cameras[gate]

            if not camera.running:
                camera.start()

        return camera

    def stop_camera(self, gate):
        gate = str(gate).strip().upper()

        camera = self.cameras.get(gate)

        if camera is not None:
            camera.stop()

    def stop_all(self):
        for camera in self.cameras.values():
            camera.stop()


camera_manager = PlateCameraManager()

def capture_plate_burst(gate, frame_count=5):
    """
    Lấy frame từ CameraWorker đang phát preview.
    Không mở VideoCapture thứ hai.
    Chạy YOLO + OCR trên các frame mới.
    """

    gate = str(gate).strip().upper()

    if gate not in CAMERA_CONFIG:
        return {
            "success": False,
            "message": "Cổng camera không hợp lệ.",
            "candidate": None,
            "confidence": 0.0,
            "valid_frames": 0,
            "votes": 0,
        }

    frame_count = max(3, min(int(frame_count), 5))

    camera = camera_manager.start_camera(gate)

    # Chỉ lấy frame được ghi nhận sau thời điểm bấm chụp.
    capture_started = time.monotonic()
    deadline = time.monotonic() + 8.0

    frames = []

    while time.monotonic() < deadline:
        if camera.error:
            return {
                "success": False,
                "message": camera.error,
                "candidate": None,
                "confidence": 0.0,
                "valid_frames": 0,
                "votes": 0,
            }

        frames = camera.get_raw_frames_since(capture_started)

        if len(frames) >= frame_count:
            frames = frames[-frame_count:]
            break

        if not camera.running:
            return {
                "success": False,
                "message": f"Camera {gate} đã dừng.",
                "candidate": None,
                "confidence": 0.0,
                "valid_frames": 0,
                "votes": 0,
            }

        time.sleep(0.05)

    if len(frames) < frame_count:
        return {
            "success": False,
            "message": (
                "Không thu đủ frame mới từ camera. "
                "Kiểm tra kết nối và thử chụp lại."
            ),
            "candidate": None,
            "confidence": 0.0,
            "valid_frames": len(frames),
            "votes": 0,
        }

    candidates = []
    confidence_values = []

    try:
        detector = get_detector()
        for frame_index, frame in enumerate(frames, start=1):
            detections = detector.detect(frame)

            logger.info(
                "[PLATE DEBUG] gate=%s frame=%d size=%s detections=%d",
                gate,
                frame_index,
                frame.shape,
                len(detections),
            )

            if not detections:
                logger.warning(
                    "[PLATE DEBUG] Frame %d không phát hiện được biển số.",
                    frame_index,
                )
                continue        

            # Ưu tiên biển số có confidence và diện tích lớn.
            detection = max(
                detections,
                key=lambda item: (
                    item["confidence"]
                    * (
                        (item["box"][2] - item["box"][0])
                        * (item["box"][3] - item["box"][1])
                    )
                ),
            )

            x1, y1, x2, y2 = detection["box"]

            height, width = frame.shape[:2]

            margin_x = int((x2 - x1) * 0.04)
            margin_y = int((y2 - y1) * 0.08)

            x1 = max(0, x1 - margin_x)
            y1 = max(0, y1 - margin_y)
            x2 = min(width, x2 + margin_x)
            y2 = min(height, y2 + margin_y)

            plate_crop = frame[y1:y2, x1:x2]
            logger.info(
                "[PLATE DEBUG] frame=%d YOLO_conf=%.3f bbox=(%d,%d,%d,%d) crop=%dx%d",
                frame_index,
                detection["confidence"],
                x1, y1, x2, y2,
                plate_crop.shape[1] if plate_crop.size else 0,
                plate_crop.shape[0] if plate_crop.size else 0,
            )

            if plate_crop.size > 0:
                cv2.imwrite(
                    f"/tmp/vin_parking_{gate}_crop_{frame_index}.jpg",
                    plate_crop,
                )            

            if (
                plate_crop.size == 0
                or plate_crop.shape[1] < 60
                or plate_crop.shape[0] < 20
            ):
                continue

            try:
                result = read_plate_ocr(plate_crop)
            except Exception:
                logger.exception("OCR lỗi tại camera %s", gate)
                result = None

            if result:
                plate_text = normalize_plate_text(result["text"])

                if plate_text:
                    candidates.append(plate_text)
                    confidence_values.append(
                        float(result["confidence"])
                    )

        if not candidates:
            return {
                "success": False,
                "message": (
                    "Không đọc được biển số rõ ràng. "
                    "Hãy kiểm tra góc camera và vị trí xe."
                ),
                "candidate": None,
                "confidence": 0.0,
                "valid_frames": 0,
                "votes": 0,
            }

        counts = {}

        for plate in candidates:
            counts[plate] = counts.get(plate, 0) + 1

        best_plate = max(counts, key=counts.get)
        best_votes = counts[best_plate]

        best_confidences = [
            confidence_values[index]
            for index, plate in enumerate(candidates)
            if plate == best_plate
        ]

        average_confidence = float(np.mean(best_confidences))

        accepted = best_votes >= 3

        return {
            "success": accepted,
            "message": (
                "Đã nhận diện ổn định biển số."
                if accepted
                else (
                    "Kết quả OCR chưa đủ đồng thuận "
                    "giữa các khung hình. Vui lòng chụp lại."
                )
            ),
            "candidate": best_plate,
            "confidence": average_confidence,
            "valid_frames": len(candidates),
            "votes": best_votes,
        }

    except Exception as exc:
        logger.exception("Lỗi xử lý camera %s", gate)

        return {
            "success": False,
            "message": f"Lỗi xử lý camera: {str(exc)}",
            "candidate": None,
            "confidence": 0.0,
            "valid_frames": len(candidates),
            "votes": 0,
        }