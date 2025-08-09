# -*- coding: utf-8 -*-

# 0. 필요한 라이브러리 임포트
import hailo_platform as hpf
import numpy as np
import cv2
import time
from PIL import ImageFont, ImageDraw, Image
import re # 정규 표현식을 사용하기 위해 임포트

# --- 1. 사용자 설정 ---
# 모델 및 추론 설정
YOLO_HEF_PATH = "yolov8m.hef"
LPR_HEF_PATH = "lprnet_test2.hef"
CONF_THRESHOLD = 0.6

# 카메라 설정
CAMERA_INDICES = [0, 2]
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# 번호판 문자셋
CHARS = [
    '가', '나', '다', '라', '마', '거', '너', '더', '러', '머', '버', '서', '어', '저',
    '고', '노', '도', '로', '모', '보', '소', '오', '조',
    '구', '누', '두', '루', '무', '부', '수', '우', '주',
    '아', '바', '사', '자', '배', '허', '하', '호', '국',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-'
]
CHARS_DICT = {char: i for i, char in enumerate(CHARS)}
# CTC Loss를 위해 blank 토큰을 마지막에 추가
NUM_CLASSES = len(CHARS) + 1

# 폰트 설정
try:
    font = ImageFont.truetype("fonts/NanumGothic.ttf", 20)
except IOError:
    print("나눔고딕 폰트 파일을 찾을 수 없습니다. 'fonts/NanumGothic.ttf' 경로에 폰트 파일을 위치시켜 주세요.")
    font = ImageFont.load_default()

# --- 2. 모델 로드 ---
print("[SETUP] Loading models...")
try:
    yolo_hef = hpf.HEF(YOLO_HEF_PATH)
    lpr_hef = hpf.HEF(LPR_HEF_PATH)
except Exception as e:
    print(f"[ERROR] Failed to load HEF files: {e}")
    exit()

yolo_input_vstream_info = yolo_hef.get_input_vstream_infos()[0]
yolo_output_vstream_info = yolo_hef.get_output_vstream_infos()[0]
YOLO_MODEL_INPUT_SHAPE = yolo_input_vstream_info.shape

lpr_input_vstream_info = lpr_hef.get_input_vstream_infos()[0]
lpr_output_vstream_info = lpr_hef.get_output_vstream_infos()[0]
LPR_MODEL_INPUT_SHAPE = lpr_input_vstream_info.shape

print(f"[INFO] YOLO Input Shape: {YOLO_MODEL_INPUT_SHAPE}")
print(f"[INFO] LPRNet Input Shape: {LPR_MODEL_INPUT_SHAPE}")


# --- 3. 전처리 및 후처리 함수 ---
def preprocess_yolo_image(image, input_shape):
    original_h, original_w, _ = image.shape
    input_height, input_width, _ = input_shape
    scale = min(input_height / original_h, input_width / original_w)
    resized_w, resized_h = int(original_w * scale), int(original_h * scale)
    resized_image = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    padded_image = np.full((input_height, input_width, 3), 128, np.uint8)
    dw, dh = (input_width - resized_w) // 2, (input_height - resized_h) // 2
    padded_image[dh:resized_h + dh, dw:resized_w + dw] = resized_image
    rgb_image = cv2.cvtColor(padded_image, cv2.COLOR_BGR2RGB)
    return np.expand_dims(rgb_image, axis=0)


def preprocess_lpr_image(image, input_shape):
    input_h, input_w, _ = input_shape
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_img = cv2.resize(gray_img, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
    rgb_replicated_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2RGB)
    return np.expand_dims(rgb_replicated_img, axis=0)


def extract_license_plates(detections, input_shape, frame_shape, conf_threshold):
    plates = []
    if detections is None or len(detections[0]) == 0:
        return plates

    detections_array = detections[0][0]
    if not isinstance(detections_array, np.ndarray):
        return plates

    original_h, original_w = frame_shape
    input_h, input_w, _ = input_shape
    scale = min(input_h / original_h, input_w / original_w)
    pad_x, pad_y = (input_w - original_w * scale) / 2, (input_h - original_h * scale) / 2

    for row in detections_array:
        score = row[4]
        if score >= conf_threshold:
            ymin, xmin, ymax, xmax = row[0:4]
            x1 = int((xmin * input_w - pad_x) / scale)
            y1 = int((ymin * input_h - pad_y) / scale)
            x2 = int((xmax * input_w - pad_x) / scale)
            y2 = int((ymax * input_h - pad_y) / scale)
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(original_w, x2), min(original_h, y2)

            if x2 > x1 and y2 > y1:
                plates.append({'box': (x1, y1, x2, y2), 'score': float(score), 'text': ''})
    return plates


def ctc_greedy_decode(raw_logits, characters):
    logits_map = np.squeeze(raw_logits, axis=0)
    logits_seq = np.transpose(logits_map, (1, 0, 2))
    best_path_indices = np.argmax(logits_seq, axis=2)
    
    decoded_indices = []
    prev_idx = -1
    for idx in best_path_indices.flatten():
        if idx != prev_idx:
            if idx != len(characters):
                decoded_indices.append(idx)
        prev_idx = idx
        
    final_text = "".join([characters[i] for i in decoded_indices])
    return final_text

# --- [수정된 함수] 엄격한 번호판 형식 검증 및 포맷팅 ---
def validate_and_format_plate_text(text):
    """
    CTC 디코딩 결과 텍스트를 엄격하게 검증하여,
    'xxx가xxxx' 또는 'xx가xxxx' 형식의 번호판만 반환합니다.
    유효한 형식을 찾지 못하면 빈 문자열("")을 반환합니다.
    """
    # 8자리 신형 번호판 형식 (예: 123가1234)
    pattern_8 = r'\d{3}[가-힣]\d{4}'
    # 7자리 구형 번호판 형식 (예: 12가1234)
    pattern_7 = r'\d{2}[가-힣]\d{4}'

    # 정규표현식 매칭을 쉽게 하기 위해 입력 텍스트에서 숫자와 한글만 남깁니다.
    cleaned_text = re.sub(r'[^0-9가-힣]', '', text)

    # 1. 8자리 형식을 먼저 찾습니다.
    match_8 = re.search(pattern_8, cleaned_text)
    if match_8:
        return match_8.group(0) # ex) '123가4567' 반환

    # 2. 8자리 형식이 없으면 7자리 형식을 찾습니다.
    match_7 = re.search(pattern_7, cleaned_text)
    if match_7:
        return match_7.group(0) # ex) '12가3456' 반환

    # 3. 두 형식 모두 찾지 못하면, 잘못된 인식이므로 빈 문자열을 반환합니다.
    return ""


def draw_results(frame, plates, display_fps, inference_fps):
    total_results = len(plates)
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    for plate in plates:
        x1, y1, x2, y2 = plate['box']
        text = plate['text']
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        
        # 텍스트가 비어있지 않은 경우에만 그립니다.
        if text:
            try:
                text_bbox = font.getbbox(text)
                text_w = text_bbox[2] - text_bbox[0]
                text_h = text_bbox[3] - text_bbox[1]
                draw.rectangle([x1, y1 - text_h - 10, x1 + text_w + 10, y1], fill="red")
                draw.text((x1 + 5, y1 - text_h - 10), text, font=font, fill=(255, 255, 255))
            except AttributeError: # getbbox 오류 발생 시 대처
                draw.text((x1, y1 - 20), text, font=font, fill=(255, 0, 0))

    fps_text = f"Display FPS: {display_fps:.2f} | Inference FPS: {inference_fps:.2f} | Detections: {total_results}"
    draw.text((10, 10), fps_text, font=font, fill=(255, 0, 0))

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# --- 4. 메인 로직 ---
def main():
    caps = [cv2.VideoCapture(idx) for idx in CAMERA_INDICES]
    for i, cap in enumerate(caps):
        if not cap.isOpened():
            print(f"카메라 {CAMERA_INDICES[i]}를 열 수 없습니다.")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', 'U', 'Y', 'V'))

    prev_time = 0
    inference_time = 0

    with hpf.VDevice() as target:
        yolo_network_group = target.configure(yolo_hef)[0]
        lpr_network_group = target.configure(lpr_hef)[0]

        yolo_input_params = hpf.InputVStreamParams.make_from_network_group(yolo_network_group, quantized=True, format_type=hpf.FormatType.UINT8)
        yolo_output_params = hpf.OutputVStreamParams.make_from_network_group(yolo_network_group, quantized=False, format_type=hpf.FormatType.FLOAT32)
        lpr_input_params = hpf.InputVStreamParams.make_from_network_group(lpr_network_group, quantized=True, format_type=hpf.FormatType.UINT8)
        lpr_output_params = hpf.OutputVStreamParams.make_from_network_group(lpr_network_group, quantized=False, format_type=hpf.FormatType.FLOAT32)

        while True:
            raw_frames = []
            for cap in caps:
                ret, frame = cap.read()
                if ret:
                    raw_frames.append(frame)
            
            if len(raw_frames) != len(CAMERA_INDICES):
                print("모든 카메라에서 프레임을 읽어오지 못했습니다. 건너뜁니다.")
                time.sleep(0.1)
                continue

            processed_frames = []
            inf_start_time = time.time()

            for frame in raw_frames:
                with yolo_network_group.activate():
                    with hpf.InferVStreams(yolo_network_group, yolo_input_params, yolo_output_params) as yolo_pipeline:
                        detections = yolo_pipeline.infer({yolo_input_vstream_info.name: preprocess_yolo_image(frame, YOLO_MODEL_INPUT_SHAPE)})[yolo_output_vstream_info.name]

                plates = extract_license_plates(detections, YOLO_MODEL_INPUT_SHAPE, (frame.shape[0], frame.shape[1]), CONF_THRESHOLD)

                for plate in plates:
                    x1, y1, x2, y2 = plate['box']
                    cropped_plate = frame[y1:y2, x1:x2]
                    if cropped_plate.size > 0:
                        with lpr_network_group.activate():
                            with hpf.InferVStreams(lpr_network_group, lpr_input_params, lpr_output_params) as lpr_pipeline:
                                raw_logits = lpr_pipeline.infer({lpr_input_vstream_info.name: preprocess_lpr_image(cropped_plate, LPR_MODEL_INPUT_SHAPE)})[lpr_output_vstream_info.name]
                        
                        # --- [핵심 수정 부분] ---
                        # 1. CTC로 텍스트를 디코딩합니다.
                        predicted_text = ctc_greedy_decode(raw_logits, CHARS)
                        # 2. 엄격한 검증 함수를 통과시켜 최종 번호판 텍스트를 얻습니다.
                        #    유효하지 않으면 이 결과는 빈 문자열이 됩니다.
                        plate['text'] = validate_and_format_plate_text(predicted_text)
                
                processed_frames.append({'frame': frame, 'plates': plates})

            inference_time = time.time() - inf_start_time
            current_time = time.time()
            display_fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
            inference_fps = 1 / inference_time if inference_time > 0 else 0
            prev_time = current_time

            final_frames = []
            for data in processed_frames:
                final_frames.append(draw_results(data['frame'], data['plates'], display_fps, inference_fps))

            if len(final_frames) == 2:
                combined_frame = np.hstack((final_frames[0], final_frames[1]))
                cv2.imshow("Hailo Inference - Dual Camera", combined_frame)
            elif len(final_frames) == 1:
                cv2.imshow("Hailo Inference - Single Camera", final_frames[0])

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()