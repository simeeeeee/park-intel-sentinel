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
from PIL import ImageFont, ImageDraw, Image
import serial
import select
import termios
import tty
import requests

# 로그 설정
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
                    "size": size,
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
        text = f"{plate['text']} ({plate['ev']})"
        draw.text((10, 30 * i), text, font=font, fill=(255, 0, 0))
        logging.info(f"[{label}][{i}번] 번호판: {plate['text']} | EV 여부: {plate['ev']} | 크기: {plate['size']}")

    result = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, result)

def send_results_to_server(left_results, right_results, sensor_name="RFID_SENSOR_1", server_url="https://222.234.38.97:8443/api/robot/status"):
    def extract_top2(results):
        return sorted(results, key=lambda p: p.get("size", 0), reverse=True)[:2]

    left_top2 = extract_top2(left_results)
    right_top2 = extract_top2(right_results)

    result_data = {
        "rfid": sensor_name,
        "vehicles": {
            "1": {
                "text": left_top2[0]["text"] if len(left_top2) > 0 else "",
                "ev": left_top2[0]["ev"] if len(left_top2) > 0 else ""
            },
            "2": {
                "text": left_top2[1]["text"] if len(left_top2) > 1 else "",
                "ev": left_top2[1]["ev"] if len(left_top2) > 1 else ""
            },
            "3": {
                "text": right_top2[0]["text"] if len(right_top2) > 0 else "",
                "ev": right_top2[0]["ev"] if len(right_top2) > 0 else ""
            },
            "4": {
                "text": right_top2[1]["text"] if len(right_top2) > 1 else "",
                "ev": right_top2[1]["ev"] if len(right_top2) > 1 else ""
            }
        }
    }

    try:
        response = requests.post(server_url, json=result_data, timeout=5, verify=False)
        if response.status_code == 200:
            logging.info("서버에 결과 전송 성공")
        else:
            logging.warning(f"서버 응답 오류: 상태 코드 {response.status_code}")
    except Exception as e:
        logging.error(f"서버 전송 중 예외 발생: {e}")

def wait_for_key_press():
    logging.info("RFID 대신 y 키 입력을 대기합니다...")
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            if select.select([sys.stdin], [], [], 0.1)[0]:
                ch = sys.stdin.read(1)
                if ch.lower() == 'y':
                    logging.info("y 키 입력 감지됨 → RFID 인식으로 처리")
                    return "1234"
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

def wait_for_rfid(port="/dev/ttyAMA10", baudrate=9600):
    try:
        ser = serial.Serial(port, baudrate, timeout=0.1)
        logging.info(f"RFID 포트 {port} 열림")
        data = ser.readline().decode().strip()
        if data:
            logging.info(f"RFID 태그 감지됨: {data}")
            return data
    except Exception as e:
        logging.warning(f"RFID 포트 열기 실패 또는 통신 오류: {e}")
    return wait_for_key_press()

def list_available_cameras():
    logging.info("사용 가능한 카메라 장치 확인 중...")
    for i in range(0, 10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            logging.info(f"카메라 인덱스 {i} 사용 가능 (/dev/video{i})")
            cap.release()

def capture_and_infer(tsanpr, frame_left, frame_right, sensor_name):
    logging.info("이미지 캡처 완료. 추론 시작.")
    start_time = time.time()

    left_results = recognize_from_frame(tsanpr, frame_left, "LEFT")
    right_results = recognize_from_frame(tsanpr, frame_right, "RIGHT")

    draw_top2_plates_on_image(frame_left, left_results, "LEFT", "left_result.jpg")
    draw_top2_plates_on_image(frame_right, right_results, "RIGHT", "right_result.jpg")

    send_results_to_server(left_results, right_results, sensor_name=sensor_name)

    end_time = time.time()
    logging.info(f"총 추론 소요 시간: {end_time - start_time:.4f} 초")

def capture_and_infer_loop(tsanpr):
    list_available_cameras()

    cap_left = cv2.VideoCapture("/dev/video0")  # 첫 번째 웹캠
    cap_right = cv2.VideoCapture("/dev/video2")  # 두 번째 웹캠 (기존 video4 → video2로 수정)

    if not cap_left.isOpened() or not cap_right.isOpened():
        logging.error("카메라 열기 실패")
        return

    logging.info("웹캠 스트리밍 시작됨")

    while True:
        ret_left, frame_left = cap_left.read()
        ret_right, frame_right = cap_right.read()

        if not ret_left or not ret_right:
            logging.warning("프레임 캡처 실패. 다시 시도 중...")
            continue

        sensor_name = wait_for_rfid()
        if sensor_name:
            capture_and_infer(tsanpr, frame_left.copy(), frame_right.copy(), sensor_name)

def main():
    engine_path = get_engine_file_name()
    if not engine_path or not os.path.exists(engine_path):
        logging.error("엔진 파일 경로 오류")
        return

    try:
        tsanpr = TSANPR(engine_path)
    except Exception as e:
        logging.error(f"TSANPR 초기화 실패: {e}")
        return

    if tsanpr.anpr_initialize("json;country=KR;multi=true;func=m"):
        logging.error("anpr_initialize() 실패")
        return

    capture_and_infer_loop(tsanpr)

if __name__ == "__main__":
    main()
