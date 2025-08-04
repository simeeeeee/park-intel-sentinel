import sys
import platform
import os
import cv2
import time
import json
import datetime
import numpy as np
from tsanpr.tsanpr import TSANPR
from PIL import ImageFont, ImageDraw, Image

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

def get_engine_file_name(): 
    arch = platform.machine()
    if sys.platform.startswith("win"):
        if arch in ("AMD64", "x86_64"):
            return os.path.join(BASE_DIR, "bin/windows-x86_64/tsanpr.dll")
        elif arch in ("x86", "i386"):
            return os.path.join(BASE_DIR, "bin/windows-x86/tsanpr.dll")
    elif sys.platform.startswith("linux"):
        if arch in ("x86_64", "amd64"):
            return os.path.join(BASE_DIR, "bin/linux-x86_64/libtsanpr.so")
        elif arch == "aarch64":
            return os.path.join(BASE_DIR, "bin/linux-aarch64/libtsanpr.so")
    return ""

def draw_korean_text(frame, text, position, font_path="/usr/share/fonts/truetype/nanum/NanumGothic.ttf", font_size=20, color=(0, 255, 255)):
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print("❗ 한글 폰트를 찾을 수 없습니다. 기본 폰트로 대체됩니다.")
        font = ImageFont.load_default()

    # y 좌표 음수 방지
    x, y = position
    y = max(0, y)
    draw.text((x, y), text, font=font, fill=color)
    return np.array(img_pil)


def webcam_read_license_plates(tsanpr, country_code):
    error = tsanpr.anpr_initialize(f"json;country={country_code};multi=true;func=vmsdr")
    if error:
        print(f"anpr_initialize() failed: {error}")
        return

    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print("❗ 카메라 장치를 열 수 없습니다.")
        return

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30
    delay = 1000 / fps

    save_dir = os.path.join(BASE_DIR, "examples/Python/anpr/img")
    os.makedirs(save_dir, exist_ok=True)

    while True:
        start = time.time()
        ret, frame = capture.read()
        if not ret:
            print("Camera read failed")
            break

        height, width = frame.shape[:2]
        stride = frame.strides[0]

        try:
            result = tsanpr.anpr_read_pixels(
                frame.ctypes.data_as(np.ctypeslib.ctypes.c_void_p),
                width, height, stride, "BGR", "json", ""
            )
        except Exception as e:
            print(f"ANPR 처리 오류: {e}")
            continue

        if result:
            try:
                plates = json.loads(result)
                print(f"Detected plates: {len(plates)}")
                print(json.dumps(plates, indent=10, ensure_ascii=False))
                for plate in plates:
                    area = plate.get("area", {})
                    x, y = int(area.get("x", 0)), int(area.get("y", 0))
                    w, h = int(area.get("width", 0)), int(area.get("height", 0))

                    # 바운딩 박스
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                    # 꼭짓점
                    vertices = plate.get("vertexes", [])
                    if isinstance(vertices, list) and len(vertices) == 4:
                        pts = [(int(v["x"]), int(v["y"])) for v in vertices]
                        for pt in pts:
                            cv2.circle(frame, pt, 4, (0, 255, 0), -1)
                        cv2.polylines(frame, [np.array(pts, dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)

                    # 텍스트 + EV
                    text = plate.get("text", "")
                    if plate.get("ev", False):
                        text += " (EV)"
                    frame = draw_korean_text(frame, text, (x, y - 25))

            except Exception as e:
                print(f"JSON parsing error: {e}")

        cv2.imshow('Webcam ANPR - Press [ESC] to exit, [F] to save.', frame)

        spent = (time.time() - start) * 1000
        key = cv2.waitKey(max(1, int(delay - spent)))

        if key == 27:  # ESC
            break
        elif key in (ord('f'), ord('F')):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(save_dir, f"capture_{timestamp}.png")
            cv2.imwrite(filepath, frame)
            print(f"📸 Saved: {filepath}")

    capture.release()
    cv2.destroyAllWindows()

def main():
    engine_file_name = get_engine_file_name()
    if not engine_file_name or not os.path.exists(engine_file_name):
        print("Unsupported OS or engine file not found")
        return

    try:
        tsanpr = TSANPR(engine_file_name)
    except Exception as ex:
        print(f"TSANPR initialization failed: {ex}")
        return
    
    # 민감도 및 최대 인식 개수 설정 (필요시)
    if hasattr(tsanpr, "set_params"):
        tsanpr.set_params({
            "min_plate_score": 0.0,
            "min_char_score": 0.0,
            "max_plate_per_image": 100,
            "allow_ev": True
        })

    webcam_read_license_plates(tsanpr, "KR")


if __name__ == "__main__":
    main()
