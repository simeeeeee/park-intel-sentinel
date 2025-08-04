import sys
import platform
import os
import cv2
import numpy as np
import ctypes
import json
import logging
import time
import serial
from PIL import ImageFont, ImageDraw, Image
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
    return {1: "GRAY", 2: "BGR565", 3: "BGR", 4: "BGRA"}.get(channels)

def recognize_from_frame(tsanpr, frame, label="CAM"):
    height, width = frame.shape[:2]
    stride = frame.strides[0]
    pixel_format = get_pixel_format(frame)
    if not pixel_format:
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
                results.append({
                    "text": plate.get("text", ""),
                    "ev": "EV" if plate.get("ev", False) else "일반",
                    "x": area.get("x", 0),
                    "y": area.get("y", 0),
                    "w": w,
                    "h": h,
                    "size": w * h,
                    "camera": label
                })
        except Exception as e:
            logging.error(f"{label} JSON 파싱 오류: {e}")
    return results

def draw_top2_plates_on_image(frame, plates, label, save_path):
    top2 = sorted(plates, key=lambda p: p["size"], reverse=True)[:2]
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
    font = ImageFont.truetype(font_path, 24)

    for i, plate in enumerate(top2, 1):
        x, y, w, h = plate["x"], plate["y"], plate["w"], plate["h"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        draw.text((x, y - 30), f"{plate['text']} ({plate['ev']})", font=font, fill=(255, 0, 0))
        logging.info(f"[{label}][{i}번] 번호판: {plate['text']} | EV 여부: {plate['ev']} | 크기: {plate['size']}")

    result = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, result)

def wait_for_rfid(port="/dev/ttyUSB0", baudrate=9600):
    try:
        ser = serial.Serial(port, baudrate, timeout=0.1)
        if not ser.is_open:
            ser.open()
        while True:
            data = ser.readline().decode().strip()
            if data:
                logging.info(f"RFID 태그 감지됨: {data}")
                return True
    except Exception as e:
        logging.error(f"RFID 포트 오류: {e}")
    return False

def capture_and_infer_loop(tsanpr):
    cap_left = cv2.VideoCapture(2)
    cap_right = cv2.VideoCapture(0)

    if not cap_left.isOpened() or not cap_right.isOpened():
        logging.error("카메라 연결 실패")
        return

    logging.info("웹캠이 항상 켜진 상태로 RFID 대기 중...")

    while True:
        if wait_for_rfid():
            logging.info("RFID 감지 → 이미지 캡처 및 추론 시작")

            ret_left, frame_left = cap_left.read()
            ret_right, frame_right = cap_right.read()

            if not ret_left or not ret_right:
                logging.error("캡처 실패")
                continue

            start_time = time.time()
            left_results = recognize_from_frame(tsanpr, frame_left, "LEFT")
            right_results = recognize_from_frame(tsanpr, frame_right, "RIGHT")
            end_time = time.time()

            draw_top2_plates_on_image(frame_left, left_results, "LEFT", "left_result.jpg")
            draw_top2_plates_on_image(frame_right, right_results, "RIGHT", "right_result.jpg")

            logging.info(f"총 추론 소요 시간: {end_time - start_time:.4f} 초")

def main():
    engine_path = get_engine_file_name()
    if not engine_path or not os.path.exists(engine_path):
        logging.error("TSANPR 엔진 파일을 찾을 수 없습니다.")
        return

    try:
        tsanpr = TSANPR(engine_path)
    except Exception as e:
        logging.error(f"TSANPR 초기화 실패: {e}")
        return

    if tsanpr.anpr_initialize("json;country=KR;multi=true;func=m"):
        logging.error("TSANPR 초기화 실패")
        return

    capture_and_infer_loop(tsanpr)

if __name__ == "__main__":
    main()
