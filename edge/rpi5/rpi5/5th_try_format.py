# -*- coding: utf-8 -*-

# 0. 필요한 라이브러리 임포트
import hailo_platform as hpf
import numpy as np
import cv2
import time
import re
import requests
import json
import logging
from collections import Counter
import threading

# --- 로깅 기본 설정 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 서버 통신 함수 ---
def send_results_to_server(result_data):
    """최종 JSON 데이터를 서버로 전송합니다."""
    server_url = "https://222.234.38.97:8443/api/robot/status"
    try:
        r = requests.post(server_url, json=result_data, timeout=5, verify=False)
        if r.status_code == 200:
            logging.info(f"서버 전송 성공 (RFID: {result_data.get('rfid')})")
        else:
            logging.error(f"서버 응답 오류: {r.status_code}, URL: {server_url}, 응답 내용: {r.text}")
    except requests.exceptions.RequestException as e:
        logging.error(f"서버 전송 실패: {e}")


# --- 1. 사용자 설정 ---
YOLO_HEF_PATH = "yolov8m.hef"
LPR_HEF_PATH = "lprnet_test.hef"
CONF_THRESHOLD = 0.6

CAMERA_INDICES = [0, 2]
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 15

ROI_CONFIG = {
    0: {
        "ZONE1": [0, 0, CAMERA_WIDTH // 2, CAMERA_HEIGHT],
        "ZONE2": [CAMERA_WIDTH // 2, 0, CAMERA_WIDTH, CAMERA_HEIGHT]
    },
    2: {
        "ZONE1": [0, 0, CAMERA_WIDTH // 2, CAMERA_HEIGHT],
        "ZONE2": [CAMERA_WIDTH // 2, 0, CAMERA_WIDTH, CAMERA_HEIGHT]
    }
}

LPR_CHARACTERS = [
    '가', '나', '다', '라', '마', '거', '너', '더', '러', '머', '버', '서', '어', '저',
    '고', '노', '도', '로', '모', '보', '소', '오', '조',
    '구', '누', '두', '루', '무', '부', '수', '우', '주',
    '아', '바', '사', '자', '배', '허', '하', '호', '국',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-'
]


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
                plates.append({'box': (x1, y1, x2, y2), 'text': ''})
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

# --- [핵심 수정] 띄어쓰기를 추가하지 않도록 함수 변경 ---
def format_plate_text(text):
    """단순히 하이픈(-)과 모든 공백을 제거하여 반환합니다."""
    return text.replace('-', '').replace(' ', '')

def is_electric_vehicle(plate_image):
    hsv = cv2.cvtColor(plate_image, cv2.COLOR_BGR2HSV)
    total_pixels = hsv.shape[0] * hsv.shape[1]
    if total_pixels == 0:
        return "NORMAL"

    lower_blue = np.array([90, 70, 50])
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    blue_ratio = cv2.countNonZero(blue_mask) / total_pixels

    lower_white_grey = np.array([0, 0, 70])
    upper_white_grey = np.array([180, 40, 255])
    white_grey_mask = cv2.inRange(hsv, lower_white_grey, upper_white_grey)
    white_grey_ratio = cv2.countNonZero(white_grey_mask) / total_pixels
    
    is_ev = blue_ratio > white_grey_ratio and blue_ratio > 0.05
    decision = "EV" if is_ev else "NORMAL"
    return decision


# --- 4. 메인 추론 및 시스템 로직 ---
def run_inference_session(caps, yolo_network_group, lpr_network_group):
    start_time = time.time()
    
    results_aggregator = {f"CAM{cam_idx}_{zone_name}": [] 
                          for cam_idx, zones in ROI_CONFIG.items() 
                          for zone_name in zones}

    yolo_input_params = hpf.InputVStreamParams.make_from_network_group(yolo_network_group, quantized=True, format_type=hpf.FormatType.UINT8)
    yolo_output_params = hpf.OutputVStreamParams.make_from_network_group(yolo_network_group, quantized=False, format_type=hpf.FormatType.FLOAT32)
    lpr_input_params = hpf.InputVStreamParams.make_from_network_group(lpr_network_group, quantized=True, format_type=hpf.FormatType.UINT8)
    lpr_output_params = hpf.OutputVStreamParams.make_from_network_group(lpr_network_group, quantized=False, format_type=hpf.FormatType.FLOAT32)

    logging.info("2초간의 추론 세션을 시작합니다...")
    while time.time() - start_time < 2.0:
        for cam_idx, cap in caps.items():
            ret, frame = cap.read()
            if not ret: continue

            with yolo_network_group.activate():
                with hpf.InferVStreams(yolo_network_group, yolo_input_params, yolo_output_params) as yolo_pipeline:
                    yolo_input = {yolo_input_vstream_info.name: preprocess_yolo_image(frame, YOLO_MODEL_INPUT_SHAPE)}
                    detections = yolo_pipeline.infer(yolo_input)[yolo_output_vstream_info.name]
            
            frame_shape = (frame.shape[0], frame.shape[1])
            detected_plates = extract_license_plates(detections, YOLO_MODEL_INPUT_SHAPE, frame_shape)
            
            for zone_name, (x1_roi, y1_roi, x2_roi, y2_roi) in ROI_CONFIG[cam_idx].items():
                unique_zone_key = f"CAM{cam_idx}_{zone_name}"
                for plate in detected_plates:
                    (x1_p, y1_p, x2_p, y2_p) = plate['box']
                    center_x, center_y = (x1_p + x2_p) // 2, (y1_p + y2_p) // 2
                    if x1_roi < center_x < x2_roi and y1_roi < center_y < y2_roi:
                        cropped_plate = frame[y1_p:y2_p, x1_p:x2_p]
                        if cropped_plate.size == 0: continue

                        ev_status = is_electric_vehicle(cropped_plate)
                        
                        with lpr_network_group.activate():
                            with hpf.InferVStreams(lpr_network_group, lpr_input_params, lpr_output_params) as lpr_pipeline:
                                lpr_input = {lpr_input_vstream_info.name: preprocess_lpr_image(cropped_plate, LPR_MODEL_INPUT_SHAPE)}
                                raw_logits = lpr_pipeline.infer(lpr_input)[lpr_output_vstream_info.name]
                        
                        predicted_text = ctc_greedy_decode(raw_logits, LPR_CHARACTERS)
                        # 수정된 format_plate_text 함수가 여기서 호출됩니다.
                        formatted_text = format_plate_text(predicted_text)
                        
                        if formatted_text:
                            results_aggregator[unique_zone_key].append((formatted_text, ev_status))

    final_results = {}
    for key in results_aggregator.keys():
        final_results[key] = {"text": "", "ev": ""}
    
    for zone_key, results in results_aggregator.items():
        if results:
            plate_texts = [res[0] for res in results]
            most_common_text = Counter(plate_texts).most_common(1)[0][0]
            
            ev_statuses_for_common_text = [res[1] for res in results if res[0] == most_common_text]
            most_common_ev_status = Counter(ev_statuses_for_common_text).most_common(1)[0][0]

            final_results[zone_key] = {"text": most_common_text, "ev": most_common_ev_status}
    
    logging.info(f"추론 세션 완료. 최종 결과: {final_results}")
    return final_results


def main():
    with hpf.VDevice() as target:
        print("[SETUP] Configuring Hailo network groups...")
        yolo_network_group = target.configure(yolo_hef)[0]
        lpr_network_group = target.configure(lpr_hef)[0]

        caps = {}
        for idx in CAMERA_INDICES:
            cap = cv2.VideoCapture(idx)
            if not cap.isOpened():
                print(f"[ERROR] Cannot open camera {idx}. It will be skipped.")
                continue
            
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
            print(f"[SETUP] Camera {idx} opened and configured successfully.")
            caps[idx] = cap
        
        print(f"\n[INFO] Camera setup finished. {len(caps)} out of {len(CAMERA_INDICES)} cameras are active.")
        if not caps:
            print("[FATAL] No cameras could be opened. Exiting program.")
            return

        print("\n[INFO] Live view starting... Press 't' to trigger inference, 'q' to quit.")
        while True:
            for cam_idx, cap in caps.items():
                ret, frame = cap.read()
                if not ret:
                    print(f"[WARNING] Failed to grab frame from camera {cam_idx}.")
                    continue
                
                if cam_idx in ROI_CONFIG:
                    for zone_name, (x1, y1, x2, y2) in ROI_CONFIG[cam_idx].items():
                        color = (0, 255, 0) if zone_name == "ZONE1" else (255, 0, 0)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, zone_name, (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
                window_name = f"Camera {cam_idx} - Live View"
                cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                logging.info("'q' 키 입력됨. 프로그램을 종료합니다.")
                break
            elif key == ord('t'):
                rfid_id = "0643E69B"
                logging.info(f"테스트 키 't' 입력됨. RFID '{rfid_id}' 신호로 간주하여 추론 및 서버 전송을 시작합니다.")
                
                inference_results = run_inference_session(caps, yolo_network_group, lpr_network_group)

                zone_mapping = {
                    "CAM0_ZONE1": "ZONE1",
                    "CAM0_ZONE2": "ZONE2",
                    "CAM2_ZONE1": "ZONE3",
                    "CAM2_ZONE2": "ZONE4"
                }
                
                vehicle_data = {
                    "ZONE1": {"text": "", "ev": ""},
                    "ZONE2": {"text": "", "ev": ""},
                    "ZONE3": {"text": "", "ev": ""},
                    "ZONE4": {"text": "", "ev": ""}
                }
                
                for internal_key, result in inference_results.items():
                    if internal_key in zone_mapping:
                        server_key = zone_mapping[internal_key]
                        vehicle_data[server_key] = result
            
                final_json_payload = {
                    "rfid": rfid_id,
                    "vehicles": vehicle_data
                }
                
                logging.info("서버로 데이터 전송을 시도합니다...")
                logging.info(f"전송할 데이터: {json.dumps(final_json_payload, indent=2, ensure_ascii=False)}")
                send_results_to_server(final_json_payload)

    for cap in caps.values():
        cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Program finished.")

if __name__ == "__main__":
    main()