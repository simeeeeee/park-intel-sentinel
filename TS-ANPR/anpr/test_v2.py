import sys
import platform
import os
import cv2
import numpy as np
import ctypes
import json
import logging
from tsanpr.tsanpr import TSANPR
from PIL import ImageFont, ImageDraw, Image

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
    else:
        return None

def draw_text(frame, text, position, font_path="/usr/share/fonts/truetype/nanum/NanumGothic.ttf", font_size=24, color=(255, 255, 0)):
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        logging.warning("폰트 로드 실패. 기본 폰트를 사용합니다.")
        font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def webcam_license_plate_recognition(tsanpr, country_code="KR"):
    error = tsanpr.anpr_initialize(f"json;country={country_code};multi=true;func=vmsdr")
    if error:
        logging.error(f"anpr_initialize() 실패: {error}")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("카메라 장치를 열 수 없습니다.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    logging.info("웹캠 번호판 인식 시작 (ESC 누르면 종료)")

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.error("카메라 프레임 읽기 실패")
            break

        height, width = frame.shape[:2]
        stride = frame.strides[0]
        pixel_format = get_pixel_format(frame)
        if pixel_format is None:
            logging.error("알 수 없는 픽셀 포맷")
            break

        img_ptr = frame.ctypes.data_as(ctypes.c_void_p)

        try:
            result_json = tsanpr.anpr_read_pixels(
                img_ptr, width, height, stride, pixel_format, "json", "m"
            )
        except Exception as e:
            logging.error(f"ANPR 처리 오류: {e}")
            continue

        if result_json:
            try:
                plates = json.loads(result_json)
                logging.info(f"번호판 {len(plates)}개 인식됨")

                plates.sort(
                    key=lambda p: p.get("area", {}).get("width", 0) * p.get("area", {}).get("height", 0),
                    reverse=True
                )
                plates = plates[:4]

                for plate in plates:
                    area = plate.get("area", {})
                    x, y = int(area.get("x", 0)), int(area.get("y", 0))
                    w, h = int(area.get("width", 0)), int(area.get("height", 0))

                    text = plate.get("text", "")
                    is_ev = plate.get("ev", False)
                    confidence = plate.get("confidence", None)

                    # EV 여부 텍스트 지정
                    ev_str = "EV" if is_ev else "일반"

                    # 로그 표시
                    if confidence is not None:
                        logging.info(f"번호판: {text} | EV 여부: {ev_str} | 정확도: {confidence:.4f}")
                    else:
                        logging.info(f"번호판: {text} | EV 여부: {ev_str}")

                    # 시각화용 텍스트
                    display_text = f"{text} ({ev_str})"

                    # 사각형 및 텍스트 표시
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    vertices = plate.get("vertexes", [])
                    if isinstance(vertices, list) and len(vertices) == 4:
                        pts = [(int(v["x"]), int(v["y"])) for v in vertices]
                        for pt in pts:
                            cv2.circle(frame, pt, 4, (255, 0, 0), -1)
                        cv2.polylines(frame, [np.array(pts, dtype=np.int32)], isClosed=True, color=(255, 0, 0), thickness=2)

                    frame = draw_text(frame, display_text, (x, max(y - 30, 0)), font_size=24, color=(0, 255, 255))

            except Exception as e:
                logging.error(f"JSON 파싱 오류: {e}")

        cv2.imshow("TS-ANPR Webcam", frame)
        if cv2.waitKey(1) == 27:
            logging.info("ESC 입력 감지. 종료합니다.")
            break

    cap.release()
    cv2.destroyAllWindows()

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

    webcam_license_plate_recognition(tsanpr, country_code="KR")

if __name__ == "__main__":
    main()
