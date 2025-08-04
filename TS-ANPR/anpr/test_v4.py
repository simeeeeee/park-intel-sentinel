import sys
import platform
import os
import cv2
import numpy as np
import ctypes
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    else:
        return None

def recognize_from_frame(tsanpr, frame, label="IMAGE"):
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
            plates.sort(key=lambda p: p.get("confidence", 0) if p.get("confidence") is not None else 0, reverse=True)

            for plate in plates:
                text = plate.get("text", "")
                is_ev = plate.get("ev", False)
                confidence = plate.get("confidence", None)
                x = plate.get("area", {}).get("x", 0)
                ev_str = "EV" if is_ev else "일반"
                result = {
                    "text": text,
                    "ev": ev_str,
                    "conf": confidence,
                    "x": x,
                    "camera": label
                }
                results.append(result)
        except Exception as e:
            logging.error(f"{label} JSON 파싱 오류: {e}")

    return results

def image_inference_task(tsanpr, image_path, label):
    if not os.path.exists(image_path):
        logging.error(f"{label} 이미지 파일이 존재하지 않습니다: {image_path}")
        return label, [], 0.0

    img = cv2.imread(image_path)
    if img is None:
        logging.error(f"{label} 이미지를 읽을 수 없습니다: {image_path}")
        return label, [], 0.0

    logging.info(f"{label} 이미지 로드 완료: {image_path}, 크기: {img.shape[1]}x{img.shape[0]}")

    start_time = time.time()
    results = recognize_from_frame(tsanpr, img, label)
    end_time = time.time()
    inference_time = end_time - start_time
    logging.info(f"[{label}] 추론 시간: {inference_time:.4f} 초")

    return label, results, inference_time

def image_inference_parallel(tsanpr, left_image_path, right_image_path):
    total_start = time.time()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(image_inference_task, tsanpr, left_image_path, "LEFT"),
            executor.submit(image_inference_task, tsanpr, right_image_path, "RIGHT"),
        ]

        results_dict = {}
        total_infer_time = 0.0
        for future in as_completed(futures):
            label, results, infer_time = future.result()
            results_dict[label] = results
            total_infer_time += infer_time

    # 왼쪽 이미지 상위 2개 번호판 출력
    left_results = results_dict.get("LEFT", [])
    left_top2 = sorted(left_results, key=lambda x: x["conf"] if x["conf"] is not None else 0, reverse=True)[:2]
    for idx, plate in enumerate(left_top2, 1):
        if plate["conf"] is not None:
            logging.info(f"[LEFT][{idx}번] 번호판: {plate['text']} | EV 여부: {plate['ev']} | 정확도: {plate['conf']:.4f}")
        else:
            logging.info(f"[LEFT][{idx}번] 번호판: {plate['text']} | EV 여부: {plate['ev']}")

    # 오른쪽 이미지 상위 2개 번호판 출력
    right_results = results_dict.get("RIGHT", [])
    right_top2 = sorted(right_results, key=lambda x: x["conf"] if x["conf"] is not None else 0, reverse=True)[:2]
    for idx, plate in enumerate(right_top2, 1):
        if plate["conf"] is not None:
            logging.info(f"[RIGHT][{idx}번] 번호판: {plate['text']} | EV 여부: {plate['ev']} | 정확도: {plate['conf']:.4f}")
        else:
            logging.info(f"[RIGHT][{idx}번] 번호판: {plate['text']} | EV 여부: {plate['ev']}")

    total_end = time.time()
    logging.info(f"총 2장 이미지 병렬 추론 총 소요 시간: {total_end - total_start:.4f} 초")

def main():
    engine_file_name = get_engine_file_name()
    if not engine_file_name or not os.path.exists(engine_file_name):
        logging.error("지원하지 않는 운영체제이거나 엔진 파일을 찾을 수 없습니다.")
        return

    try:
        tsanpr = TSANPR(engine_file_name)
    except Exception as e:
        logging.error(f"TSANPR 초기화 실패: {e}")
        return

    error = tsanpr.anpr_initialize("json;country=KR;multi=true;func=m")
    if error:
        logging.error(f"anpr_initialize() 실패: {error}")
        return

    left_image_path = "/home/pi/workspace/final_project/TS-ANPR/img/517830_32591_5954.jpg"
    right_image_path = "/home/pi/workspace/final_project/TS-ANPR/img/img.png"

    image_inference_parallel(tsanpr, left_image_path, right_image_path)

if __name__ == "__main__":
    main()
