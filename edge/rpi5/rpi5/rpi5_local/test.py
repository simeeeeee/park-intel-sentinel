# -*- coding: utf-8 -*-

# 0. 필요한 라이브러리 임포트
import hailo_platform as hpf
import numpy as np
import cv2
import time
import os
import logging

# --- 로깅 기본 설정 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 1. 사용자 설정 ---
# 모델 및 추론 설정
YOLO_HEF_PATH = "yolov8m.hef"
LPR_HEF_PATH = "lprnet_test2.hef"
CONF_THRESHOLD = 0.6
IMAGE_DATA_PATH = "./data"  # 이미지가 저장된 디렉토리

# 번호판 문자셋
LPR_CHARACTERS = [
    '가', '나', '다', '라', '마', '거', '너', '더', '러', '머', '버', '서', '어', '저',
    '고', '노', '도', '로', '모', '보', '소', '오', '조',
    '구', '누', '두', '루', '무', '부', '수', '우', '주',
    '아', '바', '사', '자', '배', '허', '하', '호', '국',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-'
]

# --- 2. 모델 로드 ---
logging.info("[SETUP] Loading models...")
try:
    yolo_hef = hpf.HEF(YOLO_HEF_PATH)
    lpr_hef = hpf.HEF(LPR_HEF_PATH)
except Exception as e:
    logging.error(f"[ERROR] Failed to load HEF files: {e}")
    exit()

yolo_input_vstream_info = yolo_hef.get_input_vstream_infos()[0]
yolo_output_vstream_info = yolo_hef.get_output_vstream_infos()[0]
YOLO_MODEL_INPUT_SHAPE = yolo_input_vstream_info.shape

lpr_input_vstream_info = lpr_hef.get_input_vstream_infos()[0]
lpr_output_vstream_info = lpr_hef.get_output_vstream_infos()[0]
LPR_MODEL_INPUT_SHAPE = lpr_input_vstream_info.shape

logging.info(f"[INFO] YOLO Input Shape: {YOLO_MODEL_INPUT_SHAPE}")
logging.info(f"[INFO] LPRNet Input Shape: {LPR_MODEL_INPUT_SHAPE}")

# --- 3. 전처리 및 후처리 함수 (기존 코드와 동일) ---
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
    equalized_img = cv2.equalizeHist(gray_img)
    resized_img = cv2.resize(equalized_img, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
    rgb_replicated_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2RGB)
    return np.expand_dims(rgb_replicated_img, axis=0)

def extract_license_plates(detections, input_shape, frame_shape):
    plates = []
    if detections is None or len(detections[0]) == 0: return plates
    detections_array = detections[0][0]
    if not isinstance(detections_array, np.ndarray): return plates
    
    original_h, original_w = frame_shape
    input_h, input_w, _ = input_shape
    scale = min(input_h / original_h, input_w / original_w)
    pad_x, pad_y = (input_w - original_w * scale) / 2, (input_h - original_h * scale) / 2
    
    for row in detections_array:
        score = row[4]
        if score >= CONF_THRESHOLD:
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
    logits_seq = np.mean(logits_map, axis=0)
    best_path_indices = np.argmax(logits_seq, axis=1)
    blank_idx = len(characters)
    decoded_indices = []
    prev_idx = -1
    for idx in best_path_indices:
        if idx != prev_idx:
            if idx != blank_idx:
                decoded_indices.append(idx)
        prev_idx = idx
    final_text = "".join([characters[i] for i in decoded_indices])
    return final_text

# --- 4. 성능 평가를 위한 메인 함수 ---
def main():
    # 이미지 파일 목록 가져오기
    if not os.path.exists(IMAGE_DATA_PATH):
        logging.error(f"이미지 디렉토리 '{IMAGE_DATA_PATH}'를 찾을 수 없습니다.")
        return
        
    image_files = [f for f in os.listdir(IMAGE_DATA_PATH) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        logging.error(f"'{IMAGE_DATA_PATH}' 디렉토리에서 이미지를 찾을 수 없습니다.")
        return

    inference_times = []

    # Hailo VDevice 초기화 및 네트워크 그룹 설정
    with hpf.VDevice() as target:
        logging.info("[SETUP] Configuring Hailo network groups...")
        yolo_network_group = target.configure(yolo_hef)[0]
        lpr_network_group = target.configure(lpr_hef)[0]

        # VStreams 파라미터 설정
        yolo_input_params = hpf.InputVStreamParams.make_from_network_group(yolo_network_group, quantized=True, format_type=hpf.FormatType.UINT8)
        yolo_output_params = hpf.OutputVStreamParams.make_from_network_group(yolo_network_group, quantized=False, format_type=hpf.FormatType.FLOAT32)
        lpr_input_params = hpf.InputVStreamParams.make_from_network_group(lpr_network_group, quantized=True, format_type=hpf.FormatType.UINT8)
        lpr_output_params = hpf.OutputVStreamParams.make_from_network_group(lpr_network_group, quantized=False, format_type=hpf.FormatType.FLOAT32)

        logging.info("\n[INFO] 성능 평가를 시작합니다...")
        
        # 각 이미지에 대해 추론 수행
        for image_name in image_files:
            image_path = os.path.join(IMAGE_DATA_PATH, image_name)
            frame = cv2.imread(image_path)
            if frame is None:
                logging.warning(f"이미지 '{image_name}'를 로드할 수 없습니다. 건너뜁니다.")
                continue

            try:
                # --- 성능 측정 시작 ---
                start_time = time.perf_counter()

                # 1. YOLOv8 추론
                with yolo_network_group.activate():
                    with hpf.InferVStreams(yolo_network_group, yolo_input_params, yolo_output_params) as yolo_pipeline:
                        detections = yolo_pipeline.infer({yolo_input_vstream_info.name: preprocess_yolo_image(frame, YOLO_MODEL_INPUT_SHAPE)})[yolo_output_vstream_info.name]
                
                # 2. 번호판 추출
                detected_plates = extract_license_plates(detections, YOLO_MODEL_INPUT_SHAPE, (frame.shape[0], frame.shape[1]))
                
                # 3. 각 번호판에 대해 LPRNet 추론
                recognized_texts = []
                if detected_plates:
                    for plate in detected_plates:
                        x1, y1, x2, y2 = plate['box']
                        cropped_plate = frame[y1:y2, x1:x2]

                        if cropped_plate.size > 0:
                            with lpr_network_group.activate():
                                with hpf.InferVStreams(lpr_network_group, lpr_input_params, lpr_output_params) as lpr_pipeline:
                                    raw_logits = lpr_pipeline.infer({lpr_input_vstream_info.name: preprocess_lpr_image(cropped_plate, LPR_MODEL_INPUT_SHAPE)})[lpr_output_vstream_info.name]
                            
                            predicted_text = ctc_greedy_decode(raw_logits, LPR_CHARACTERS)
                            recognized_texts.append(predicted_text)

                # --- 성능 측정 종료 ---
                end_time = time.perf_counter()
                duration = (end_time - start_time) * 1000  # 밀리초(ms) 단위로 변환
                inference_times.append(duration)
                
                logging.info(f"'{image_name}' 처리 완료. | 인식된 번호판: {recognized_texts} | 소요 시간: {duration:.2f} ms")

            except Exception as e:
                logging.error(f"'{image_name}' 처리 중 오류 발생: {e}", exc_info=True)

    # --- 최종 결과 집계 및 출력 ---
    if inference_times:
        average_time = sum(inference_times) / len(inference_times)
        total_images = len(inference_times)
        
        print("\n" + "="*50)
        print("성능 평가 결과 요약")
        print("="*50)
        print(f"총 처리된 이미지 수: {total_images} 개")
        print(f"평균 추론 시간: {average_time:.2f} ms")
        print(f"초당 평균 처리 이미지 수 (FPS): {1000 / average_time:.2f} FPS")
        print("="*50)
    else:
        logging.warning("처리된 이미지가 없어 성능을 평가할 수 없습니다.")

    logging.info("[INFO] Program finished.")

if __name__ == "__main__":
    main()