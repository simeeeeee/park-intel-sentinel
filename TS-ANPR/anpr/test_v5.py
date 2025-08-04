import sys
import platform
import os
import cv2
import numpy as np
import ctypes
import json
import logging
import time
from tsanpr.tsanpr import TSANPR

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

EXAMPLES_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

def get_engine_file_name():
    arch = platform.machine()
    if sys.platform.startswith("win"):
        if arch in ("AMD64", "x86_64"):
            return os.path.join(EXAMPLES_BASE_DIR, "bin/windows-x86_64/tsanpr.dll")
        elif arch in ("x86", "i386"):
            return os.path.join(EXAMPLES_BASE_DIR, "bin/windows-x86/tsanpr.dll")
    elif sys.platform.startswith("linux"):
        if arch in ("x86_64", "amd64"):
            return os.path.join(EXAMPLES_BASE_DIR, "bin/linux-x86_64/libtsanpr.so")
        elif arch == "aarch64":
            return os.path.join(EXAMPLES_BASE_DIR, "bin/linux-aarch64/libtsanpr.so")
    return ""

def get_pixel_format(img):
    channels = 1 if len(img.shape) == 2 else img.shape[2]
    if channels == 1:
        return "GRAY"
    elif channels == 2:
        return "BGR565"
    elif channels == 3:
        return "BGR"
    elif channels == 4:
        return "BGRA"
    return None

def recognize_from_frame(tsanpr, frame, label="CAM"):
    height, width = frame.shape[:2]
    stride = frame.strides[0]
    pixel_format = get_pixel_format(frame)
    if pixel_format is None:
        logging.error(f"{label} 알 수 없는 픽셀 포맷")
        return []

    img_ptr = frame.ctypes.data_as(ctypes.c_void_p)
    try:
        result_json = tsanpr.anpr_read_pixels(
            img_ptr, width, height, stride, pixel_format, "json", "m"
        )
    except Exception as e:
        logging.error(f"{label} ANPR 처리 오류: {e}")
        return []

    results = []
    if result_json:
        try:
            plates = json.loads(result_json)
            for plate in plates:
                area = plate.get("area", {})
                w, h = area.get("w", 0), area.get("h", 0)
                size = w * h
                results.append({
                    "text": plate.get("text", ""),
                    "ev": "EV" if plate.get("ev", False) else "일반",
                    "x": area.get("x", 0),
                    "y": area.get("y", 0),
                    "w": w,
                    "h": h,
                    "size": size,
                    "camera": label
                })
        except Exception as e:
            logging.error(f"{label} JSON 파싱 오류: {e}")

    return results

def draw_top2_plates(frame, plates, label):
    top2 = sorted(plates, key=lambda p: p["size"], reverse=True)[:2]
    for i, plate in enumerate(top2, 1):
        x, y, w, h = plate["x"], plate["y"], plate["w"], plate["h"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{plate['text']} ({plate['ev']})", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        logging.info(f"[{label}][{i}번] 번호판: {plate['text']} | EV 여부: {plate['ev']} | 크기: {plate['size']}")
    return frame

def realtime_recognition(tsanpr):
    cap_left = cv2.VideoCapture(2)
    cap_right = cv2.VideoCapture(0)

    if not cap_left.isOpened() or not cap_right.isOpened():
        logging.error("카메라 열기 실패")
        return

    logging.info("ESC 키를 누르면 종료됩니다.")

    try:
        while True:
            ret_left, frame_left = cap_left.read()
            ret_right, frame_right = cap_right.read()

            if not ret_left or not ret_right:
                logging.warning("프레임 캡처 실패")
                continue

            start_time = time.time()

            left_results = recognize_from_frame(tsanpr, frame_left, "LEFT")
            right_results = recognize_from_frame(tsanpr, frame_right, "RIGHT")

            frame_left = draw_top2_plates(frame_left, left_results, "LEFT")
            frame_right = draw_top2_plates(frame_right, right_results, "RIGHT")

            end_time = time.time()

            total_time = end_time - start_time
            logging.info(f"추론 소요 시간: {total_time:.4f} 초")

            cv2.imshow("LEFT Camera", frame_left)
            cv2.imshow("RIGHT Camera", frame_right)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC 키
                break

    except KeyboardInterrupt:
        logging.info("종료 요청 감지됨")

    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()

def main():
    engine_path = get_engine_file_name()
    if not engine_path or not os.path.exists(engine_path):
        logging.error("엔진 파일을 찾을 수 없습니다.")
        return

    try:
        tsanpr = TSANPR(engine_path)
    except Exception as e:
        logging.error(f"TSANPR 초기화 실패: {e}")
        return

    if tsanpr.anpr_initialize("json;country=KR;multi=true;func=m"):
        logging.error("anpr_initialize() 실패")
        return

    realtime_recognition(tsanpr)

if __name__ == "__main__":
    main()
