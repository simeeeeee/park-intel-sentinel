# -*- coding: utf-8 -*-
import sys
import platform
import os
os.environ["QT_QPA_PLATFORM"] = "xcb"
import cv2
import numpy as np
import ctypes
import json
import logging
import time
import threading
from tsanpr.tsanpr import TSANPR
from PIL import ImageFont, ImageDraw, Image
import requests
from MFRC522 import MFRC522
from concurrent.futures import ThreadPoolExecutor

# 설정
SAVE_IMAGES = False  # 필요 시 True로 변경

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s', datefmt='%H:%M:%S')
EXAMPLES_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

def get_engine_file_name():
    arch = platform.machine()
    if sys.platform.startswith("linux"):
        if arch == "aarch64":
            return os.path.join(EXAMPLES_BASE_DIR, "bin/linux-aarch64/libtsanpr.so")
        elif arch in ("x86_64", "amd64"):
            return os.path.join(EXAMPLES_BASE_DIR, "bin/linux-x86_64/libtsanpr.so")
    return ""

def get_pixel_format(img):
    channels = 1 if len(img.shape) == 2 else img.shape[2]
    return {1: "GRAY", 2: "BGR565", 3: "BGR", 4: "BGRA"}.get(channels, None)

def recognize_from_frame(tsanpr, frame, label):
    height, width = frame.shape[:2]
    stride = frame.strides[0]
    pixel_format = get_pixel_format(frame)
    if not pixel_format:
        logging.error(f"{label} 알 수 없는 픽셀 포맷")
        return []
    try:
        result_json = tsanpr.anpr_read_pixels(
            frame.ctypes.data_as(ctypes.c_void_p),
            width, height, stride, pixel_format, "json", "m"
        )
        plates = json.loads(result_json) if result_json else []
        results = []
        for plate in plates:
            area = plate.get("area", {})
            size = area.get("w", 0) * area.get("h", 0)
            results.append({
                "text": plate.get("text", ""),
                "ev": "EV" if plate.get("ev", False) else "일반",
                "size": size,
                "camera": label
            })
        return results
    except Exception as e:
        logging.error(f"{label} 처리 오류: {e}")
        return []

def draw_top2_plates_on_image(frame, plates, label, save_path):
    if not SAVE_IMAGES:
        return
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype("/usr/share/fonts/truetype/nanum/NanumGothic.ttf", 24)
    for i, plate in enumerate(sorted(plates, key=lambda p: p["size"], reverse=True)[:2]):
        draw.text((10, 30 * (i+1)), f"{plate['text']} ({plate['ev']})", font=font, fill=(255, 0, 0))
    cv2.imwrite(save_path, cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR))

def send_results_to_server(results, sensor_name):
    def get_top1(results): 
        return sorted(results, key=lambda x: x.get("size", 0), reverse=True)[0] if results else {"text": "", "ev": "일반"}
    result_data = {
        "rfid": sensor_name,
        "vehicles": {
            "ZONE1": get_top1(results.get("1", [])),
            "ZONE2": get_top1(results.get("2", [])),
            "ZONE3": get_top1(results.get("3", [])),
            "ZONE4": get_top1(results.get("4", [])),
        }
    }
    try:
        r = requests.post("https://222.234.38.97:8443/api/robot/status", json=result_data, timeout=5, verify=False)
        logging.info("서버 전송 완료" if r.status_code == 200 else f"서버 응답 오류: {r.status_code}")
    except Exception as e:
        logging.error(f"서버 전송 실패: {e}")

def rfid_thread():
    reader = MFRC522()
    logging.info("RFID 리더 초기화 완료")
    last_read_time = 0
    while True:
        if time.time() - last_read_time < 1:
            time.sleep(0.1)
            continue
        (status, tag_type) = reader.MFRC522_Request(reader.PICC_REQIDL)
        if status == reader.MI_OK:
            (status, uid) = reader.MFRC522_Anticoll(0x93)
            if status == reader.MI_OK:
                # uid 배열에서 첫 바이트 제외하고 4바이트 정방향 16진수 대문자 변환
                uid_str = f"{uid[3]:02X}{uid[2]:02X}{uid[1]:02X}{uid[0]:02X}"
                logging.info(f"Formatted UID: {uid_str}")
                sensor_name_holder["name"] = uid_str
                last_read_time = time.time()
                rfid_event.set()

def process_zone(tsanpr, region, k, results):
    recog = recognize_from_frame(tsanpr, region, f"CAM_{k}")
    results[k] = recog
    draw_top2_plates_on_image(region, recog, f"CAM_{k}", f"result_{k}.jpg")

def capture_and_infer(tsanpr, frame_left, frame_right, sensor_name):
    h, w = frame_left.shape[:2]
    mid = w // 2
    regions = {
        "1": frame_left[:, :mid],     # 왼쪽 카메라 왼쪽  → ZONE1
        "2": frame_left[:, mid:],     # 왼쪽 카메라 오른쪽 → ZONE2
        "3": frame_right[:, :mid],    # 오른쪽 카메라 왼쪽 → ZONE3
        "4": frame_right[:, mid:]     # 오른쪽 카메라 오른쪽 → ZONE4
    }

    results = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {}
        for k in ["1", "2", "3", "4"]:
            futures[k] = executor.submit(process_zone, tsanpr, regions[k], k, results)
        for k in ["1", "2", "3", "4"]:
            futures[k].result()

    for k in ["1", "2", "3", "4"]:
        recog = results.get(k, [])
        if recog:
            top = sorted(recog, key=lambda x: x["size"], reverse=True)[0]
            logging.info(f"[ZONE{k}] 인식된 번호: {top['text']} | EV 여부: {top['ev']}")
        else:
            logging.info(f"[ZONE{k}] 인식된 번호 없음")

    send_results_to_server(results, sensor_name)

def capture_loop(tsanpr):
    cap_l, cap_r = cv2.VideoCapture("/dev/video2"), cv2.VideoCapture("/dev/video0")
    if not cap_l.isOpened() or not cap_r.isOpened():
        logging.error("카메라 열기 실패")
        return

    threading.Thread(target=rfid_thread, daemon=True).start()
    logging.info("웹캠 실행 시작")

    while True:
        ret_l, frame_l = cap_l.read()
        ret_r, frame_r = cap_r.read()
        if not (ret_l and ret_r):
            logging.warning("프레임 캡처 실패")
            continue

        cv2.imshow("Left Camera", frame_l)
        cv2.imshow("Right Camera", frame_r)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if rfid_event.is_set():
            rfid_event.clear()
            capture_and_infer(tsanpr, frame_l, frame_r, sensor_name_holder["name"])

        time.sleep(0.03)

    cap_l.release()
    cap_r.release()
    cv2.destroyAllWindows()

def main():
    path = get_engine_file_name()
    if not path or not os.path.exists(path):
        logging.error("TSANPR 엔진 파일 경로 오류")
        return
    try:
        tsanpr = TSANPR(path)
        if tsanpr.anpr_initialize("json;country=KR;multi=true;func=m"):
            logging.error("anpr_initialize() 실패")
            return
        capture_loop(tsanpr)
    except Exception as e:
        logging.error(f"TSANPR 초기화 실패: {e}")

# 전역 변수
rfid_event = threading.Event()
sensor_name_holder = {"name": None}

if __name__ == "__main__":
    main()
