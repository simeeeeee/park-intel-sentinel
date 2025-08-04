import sys
import platform
import os
import cv2
import numpy as np
import ctypes
import json
import logging
import time
import datetime
from tsanpr.tsanpr import TSANPR
from PIL import ImageFont, ImageDraw, Image
import serial
import select
import termios
import tty
import requests

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


def draw_top2_plates_on_image(frame, plates, label, save_path):
    top2 = sorted(plates, key=lambda p: p["size"], reverse=True)[:2]

    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
    font = ImageFont.truetype(font_path, 24)

    for i, plate in enumerate(top2, 1):
        x, y, w, h = plate["x"], plate["y"], plate["w"], plate["h"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = f"{plate['text']} ({plate['ev']})"
        draw.text((x, y - 30), text, font=font, fill=(255, 0, 0))
        logging.info(f"[{label}][{i}번] 번호판: {plate['text']} | EV 여부: {plate['ev']} | 크기: {plate['size']}")

    result = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, result)


def save_results_to_json(left_results, right_results, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)

    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    def simplify_results(results):
        # 크기(size) 내림차순 정렬 후 상위 2개만 선택
        top2 = sorted(results, key=lambda p: p.get("size", 0), reverse=True)[:2]
        simplified = []
        for plate in top2:
            simplified.append({
                "text": plate.get("text", ""),
                "ev": plate.get("ev", ""),
                "size": plate.get("size", 0),
                "confidence": plate.get("confidence", 0.0)  # confidence가 없으면 0.0으로 기본값
            })
        return simplified

    result_data = {
        "timestamp": now,
        "left_camera": simplify_results(left_results),
        "right_camera": simplify_results(right_results)
    }

    file_path = os.path.join(output_dir, f"anpr_result_{now}.json")

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=4)

    logging.info(f"JSON 결과 저장 완료: {file_path}")

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
                    return True
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def wait_for_rfid(port="/dev/ttyUSB0", baudrate=9600):
    try:
        ser = serial.Serial(port, baudrate, timeout=0.1)
        logging.info(f"RFID 포트 {port} 열림")
        data = ser.readline().decode().strip()
        if data:
            logging.info(f"RFID 태그 감지됨: {data}")
            return True
    except Exception as e:
        logging.warning(f"RFID 포트 열기 실패 또는 통신 오류: {e}")
    return wait_for_key_press()


def send_results_to_server(left_results, right_results, server_url):
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    def simplify_results(results):
        top2 = sorted(results, key=lambda p: p.get("size", 0), reverse=True)[:2]
        simplified = []
        for plate in top2:
            simplified.append({
                "text": plate.get("text", ""),
                "ev": plate.get("ev", ""),
                "size": plate.get("size", 0),
                "confidence": plate.get("confidence", 0.0)
            })
        return simplified

    data = {
        "timestamp": now,
        "left_camera": simplify_results(left_results),
        "right_camera": simplify_results(right_results)
    }

    try:
        response = requests.post(server_url, json=data, timeout=10)
        if response.status_code == 200:
            logging.info(f"서버에 결과 전송 성공: {response.text}")
        else:
            logging.warning(f"서버 전송 실패: 상태 코드 {response.status_code} / 응답: {response.text}")
    except Exception as e:
        logging.error(f"서버 전송 중 오류 발생: {e}")

# 기존 capture_and_infer 함수 내 호출부분 수정
def capture_and_infer(tsanpr, frame_left, frame_right, server_url):
    logging.info("이미지 캡처 완료. 추론 시작.")
    start_time = time.time()

    left_results = recognize_from_frame(tsanpr, frame_left, "LEFT")
    right_results = recognize_from_frame(tsanpr, frame_right, "RIGHT")

    draw_top2_plates_on_image(frame_left, left_results, "LEFT", "left_result.jpg")
    draw_top2_plates_on_image(frame_right, right_results, "RIGHT", "right_result.jpg")
    send_results_to_server(left_results, right_results, server_url)

    end_time = time.time()
    logging.info(f"총 추론 소요 시간: {end_time - start_time:.4f} 초")

# capture_and_infer_loop 함수도 수정 필요 (server_url 인자 추가)
def capture_and_infer_loop(tsanpr, server_url):
    cap_left = cv2.VideoCapture(2)
    cap_right = cv2.VideoCapture(0)

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

        if wait_for_rfid():
            capture_and_infer(tsanpr, frame_left.copy(), frame_right.copy(), server_url)


def main():
    server_url = "https://222.234.38.97:8443/api/endpoint"  # 여기에 중앙 서버 주소 넣기

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

    capture_and_infer_loop(tsanpr, server_url)


if __name__ == "__main__":
    main()
