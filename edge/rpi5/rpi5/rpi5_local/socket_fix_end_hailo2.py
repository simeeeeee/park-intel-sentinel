# -*- coding: utf-8 -*-

# 0. 필요한 라이브러리 임포트
import hailo_platform as hpf
import numpy as np
import cv2
import time
import requests
import json
import logging
from collections import Counter
import threading
import socket
import queue
import os
from datetime import datetime

# --- SSL 경고 메시지 비활성화 ---
from urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

# --- 로깅 기본 설정 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 1. 사용자 설정 ---
# 모델 및 추론 설정
YOLO_HEF_PATH = "yolov8m.hef"
LPR_HEF_PATH = "lprnet_test2.hef"
CONF_THRESHOLD = 0.6
INFERENCE_FRAME_COUNT = 5
FRAME_SAVE_PATH = "./frame"

# 카메라 설정
CAMERA_INDICES = [0, 2]
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 15

# 소켓 서버 설정
SOCKET_HOST = '0.0.0.0'
SOCKET_PORT = 10004

# 번호판 문자셋
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

def format_plate_text(text):
    return text.replace('-', '').replace(' ', '')

def is_electric_vehicle(plate_image):
    hsv = cv2.cvtColor(plate_image, cv2.COLOR_BGR2HSV)
    total_pixels = hsv.shape[0] * hsv.shape[1]
    if total_pixels == 0: return "NORMAL"
    lower_blue = np.array([90, 70, 50])
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    blue_ratio = cv2.countNonZero(blue_mask) / total_pixels
    lower_white_grey = np.array([0, 0, 70])
    upper_white_grey = np.array([180, 40, 255])
    white_grey_mask = cv2.inRange(hsv, lower_white_grey, upper_white_grey)
    white_grey_ratio = cv2.countNonZero(white_grey_mask) / total_pixels
    is_ev = blue_ratio > white_grey_ratio and blue_ratio > 0.05
    return "EV" if is_ev else "NORMAL"

def send_results_to_server(result_data):
    server_url = "https://222.234.38.97:8443/api/robot/status"
    try:
        r = requests.post(server_url, json=result_data, timeout=10, verify=False)
        if r.status_code == 200:
            logging.info(f"서버 전송 성공 (RFID: {result_data.get('rfid')})")
        else:
            logging.error(f"서버 응답 오류: {r.status_code}, URL: {server_url}, 응답 내용: {r.text}")
    except requests.exceptions.RequestException as e:
        logging.error(f"서버 전송 실패: {e}")

# --- 4. 시스템 로직 ---

class CameraThread(threading.Thread):
    def __init__(self, camera_index):
        super().__init__()
        self.camera_index = camera_index
        self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.running = False
        if not self.cap.isOpened():
            logging.error(f"카메라 {self.camera_index}를 열 수 없습니다.")
        else:
            self.running = True
            logging.info(f"카메라 {self.camera_index} 활성화.")

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.frame_lock:
                    self.latest_frame = frame.copy()
            else:
                logging.warning(f"카메라 {self.camera_index}에서 프레임을 읽지 못했습니다. 1초 후 재시도합니다.")
                time.sleep(1)
        self.cap.release()
        logging.info(f"카메라 {self.camera_index} 비활성화.")

    def get_frame(self):
        with self.frame_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def stop(self):
        self.running = False

# [수정됨] perform_inference 함수는 이제 오직 추론 로직에만 집중합니다. (lock 관리 제거)
def perform_inference(rfid_data, camera_threads, yolo_network_group, lpr_network_group):
    try:
        logging.info(f"RFID '{rfid_data['id']}' 처리 시작 (총 {rfid_data['count']}번째 신호).")
        os.makedirs(FRAME_SAVE_PATH, exist_ok=True)

        yolo_input_params = hpf.InputVStreamParams.make_from_network_group(yolo_network_group, quantized=True, format_type=hpf.FormatType.UINT8)
        yolo_output_params = hpf.OutputVStreamParams.make_from_network_group(yolo_network_group, quantized=False, format_type=hpf.FormatType.FLOAT32)
        lpr_input_params = hpf.InputVStreamParams.make_from_network_group(lpr_network_group, quantized=True, format_type=hpf.FormatType.UINT8)
        lpr_output_params = hpf.OutputVStreamParams.make_from_network_group(lpr_network_group, quantized=False, format_type=hpf.FormatType.FLOAT32)
        
        all_frames_results = []
        best_frames_per_camera = {
            cam_idx: {'frame': None, 'score': -1}
            for cam_idx in camera_threads.keys()
        }

        for i in range(INFERENCE_FRAME_COUNT):
            frame_summary = {"frame_index": i, "cameras": {}}
            for cam_idx, cam_thread in camera_threads.items():
                frame = cam_thread.get_frame()
                if frame is None:
                    continue

                with yolo_network_group.activate():
                    with hpf.InferVStreams(yolo_network_group, yolo_input_params, yolo_output_params) as yolo_pipeline:
                        detections = yolo_pipeline.infer({yolo_input_vstream_info.name: preprocess_yolo_image(frame, YOLO_MODEL_INPUT_SHAPE)})[yolo_output_vstream_info.name]
                
                detected_plates = extract_license_plates(detections, YOLO_MODEL_INPUT_SHAPE, (frame.shape[0], frame.shape[1]))
                valid_plates = [p for p in detected_plates if (p['box'][2] - p['box'][0]) >= 150]

                cam_results = []
                if valid_plates:
                    best_plate_in_frame = max(valid_plates, key=lambda p: p['score'])
                    if best_plate_in_frame['score'] > best_frames_per_camera[cam_idx]['score']:
                        best_frames_per_camera[cam_idx].update({
                            'frame': frame.copy(),
                            'score': best_plate_in_frame['score']
                        })
                    
                    for plate in valid_plates:
                        x1, y1, x2, y2 = plate['box']
                        cropped_plate = frame[y1:y2, x1:x2]

                        if cropped_plate.size > 0:
                            with lpr_network_group.activate():
                                with hpf.InferVStreams(lpr_network_group, lpr_input_params, lpr_output_params) as lpr_pipeline:
                                    raw_logits = lpr_pipeline.infer({lpr_input_vstream_info.name: preprocess_lpr_image(cropped_plate, LPR_MODEL_INPUT_SHAPE)})[lpr_output_vstream_info.name]
                            
                            predicted_text = ctc_greedy_decode(raw_logits, LPR_CHARACTERS)
                            formatted_text = format_plate_text(predicted_text)
                            
                            if formatted_text:
                                center_x = (x1 + x2) / 2.0
                                frame_midpoint_x = CAMERA_WIDTH / 2.0
                                zone_position = "LEFT" if center_x < frame_midpoint_x else "RIGHT"
                                
                                cam_results.append({
                                    "zone": f"CAM{cam_idx}_{zone_position}", 
                                    "text": formatted_text, 
                                    "ev": is_electric_vehicle(cropped_plate)
                                })
                
                frame_summary["cameras"][cam_idx] = cam_results
            all_frames_results.append(frame_summary)
            time.sleep(0.05)

        # --- 최종 결과 집계 ---
        final_vehicle_data = {f"ZONE{z}": {"text": "", "ev": ""} for z in range(1, 5)}
        zone_mapping = {"CAM0_LEFT": "ZONE1", "CAM0_RIGHT": "ZONE2", "CAM2_LEFT": "ZONE3", "CAM2_RIGHT": "ZONE4"}
        aggregated_results = {}

        for frame_res in all_frames_results:
            for cam_res_list in frame_res["cameras"].values():
                for res in cam_res_list:
                    zone = res['zone']
                    if zone not in aggregated_results: aggregated_results[zone] = []
                    aggregated_results[zone].append((res['text'], res['ev']))

        for zone_key, results_list in aggregated_results.items():
            if results_list:
                most_common_text = Counter(res[0] for res in results_list).most_common(1)[0][0]
                most_common_ev = Counter(res[1] for res in results_list if res[0] == most_common_text).most_common(1)[0][0]
                if zone_key in zone_mapping:
                    final_vehicle_data[zone_mapping[zone_key]] = {"text": most_common_text, "ev": most_common_ev}

        # --- 파일 저장 및 서버 전송 ---
        timestamp_str = datetime.fromtimestamp(rfid_data['arrival_time']).strftime('%Y%m%d_%H%M%S')
        base_filename = f"{timestamp_str}_rfid_{rfid_data['id']}"

        saved_image_files = {}
        for cam_idx, best_info in best_frames_per_camera.items():
            if best_info['frame'] is not None:
                img_filename = os.path.join(FRAME_SAVE_PATH, f"{base_filename}_cam{cam_idx}_best.jpg")
                cv2.imwrite(img_filename, best_info['frame'])
                logging.info(f"카메라 {cam_idx}의 선명한 프레임 저장: {img_filename} (Score: {best_info['score']:.2f})")
                saved_image_files[f"camera_{cam_idx}"] = img_filename

        local_json_output = {
            "rfid": rfid_data['id'],
            "arrival_time": datetime.fromtimestamp(rfid_data['arrival_time']).isoformat(),
            "time_since_last_signal_sec": f"{rfid_data['interval']:.2f}",
            "total_signal_count": rfid_data['count'],
            "saved_image_files": saved_image_files,
            "inference_results": final_vehicle_data,
            "raw_frame_data": all_frames_results
        }
        json_filename = os.path.join(FRAME_SAVE_PATH, f"{base_filename}_results.json")
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(local_json_output, f, ensure_ascii=False, indent=4)
        logging.info(f"상세 추론 결과 JSON 저장: {json_filename}")
        
        server_payload = {"rfid": rfid_data['id'], "vehicles": final_vehicle_data}
        send_results_to_server(server_payload)

    except Exception as e:
        logging.error(f"추론 처리 중 예외 발생: {e}", exc_info=True)
    finally:
        # [수정됨] 이 함수는 더 이상 lock을 해제하지 않습니다.
        logging.info(f"RFID '{rfid_data['id']}' 처리 완료.")


# 수정된 코드
def socket_listener(rfid_queue):
    # 서버 소켓 생성 (프로그램이 실행되는 동안 계속 유지)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((SOCKET_HOST, SOCKET_PORT))
        s.listen()
        logging.info(f"[Socket] {SOCKET_HOST}:{SOCKET_PORT}에서 클라이언트 연결을 기다립니다...")

        # 프로그램이 실행되는 동안 계속해서 클라이언트의 연결을 받아 처리
        while True:
            try:
                # 1. 클라이언트가 연결될 때까지 여기서 대기합니다.
                conn, addr = s.accept()
                logging.info(f"[Socket] 클라이언트 {addr}가 연결되었습니다. 데이터 수신을 시작합니다.")

                # 2. 일단 연결되면, 해당 연결이 유지되는 동안 계속 데이터를 받습니다.
                while True:
                    data = conn.recv(1024)
                    
                    # 3. 만약 받은 데이터가 비어있다면, 클라이언트가 연결을 정상적으로 끊었다는 신호입니다.
                    if not data:
                        logging.warning(f"[Socket] 클라이언트 {addr}와의 연결이 끊어졌습니다. 새로운 연결을 기다립니다.")
                        break # 데이터 수신 루프를 탈출하여 다시 accept() 상태로 돌아갑니다.

                    rfid_id = data.decode('utf-8').strip()
                    if rfid_id:
                        logging.info(f"[Socket] 수신된 RFID: {rfid_id}. 작업 큐에 추가합니다.")
                        rfid_queue.put(rfid_id)

            # 4. 클라이언트의 강제 종료 등 비정상적인 연결 끊김에 대비한 예외 처리
            except (ConnectionResetError, BrokenPipeError) as e:
                logging.warning(f"[Socket] 클라이언트와의 연결이 비정상적으로 종료되었습니다: {e}")
            # 5. 그 외 소켓 관련 에러 처리
            except Exception as e:
                logging.error(f"[Socket] 소켓 오류 발생: {e}")
                time.sleep(5)

# [신규] Worker 스레드가 실행할 함수
def worker(rfid_queue, shared_state, camera_threads, yolo_network_group, lpr_network_group, inference_lock):
    """
    큐에서 작업을 가져와 순차적으로 처리하는 작업자 함수.
    추론 작업(Hailo 하드웨어 사용)이 동시에 실행되지 않도록 inference_lock으로 제어.
    """
    logging.info(f"작업자 스레드 {threading.get_ident()} 시작됨. 작업 대기 중...")
    while True:
        # 큐에서 RFID ID를 가져옴. 큐가 비어있으면 여기서 대기함.
        rfid_id = rfid_queue.get()
        
        logging.info(f"'{rfid_id}' 신호 수신. 이전 작업이 끝날 때까지 대기합니다...")
        
        # inference_lock을 획득할 때까지 여기서 대기 (Blocking).
        # 이렇게 함으로써 추론 작업은 항상 하나씩만 실행됨.
        inference_lock.acquire()
        
        try:
            logging.info(f"잠금 획득. '{rfid_id}' 처리를 시작합니다.")

            # 추론에 필요한 데이터 구성
            current_time = time.time()
            shared_state['rfid_counter'] += 1
            rfid_data = {
                'id': rfid_id,
                'arrival_time': current_time,
                'interval': current_time - shared_state['last_rfid_time'],
                'count': shared_state['rfid_counter']
            }
            shared_state['last_rfid_time'] = current_time

            # 실제 추론 함수 호출 (이제 lock을 전달하지 않음)
            perform_inference(rfid_data, camera_threads, yolo_network_group, lpr_network_group)
        
        finally:
            # 작업 완료 후 반드시 잠금 해제하여 다음 작업이 실행될 수 있도록 함
            inference_lock.release()
            logging.info(f"잠금 해제. 시스템 대기 상태로 복귀.")
            rfid_queue.task_done()

def main():
    rfid_queue = queue.Queue()
    shared_state = {'last_rfid_time': time.time(), 'rfid_counter': 0}
    inference_lock = threading.Lock() # 추론 하드웨어에 대한 접근을 제어하기 위한 잠금

    camera_threads = {idx: CameraThread(idx) for idx in CAMERA_INDICES}
    for cam_thread in camera_threads.values():
        if cam_thread.running:
            cam_thread.start()

    # 소켓 리스너 스레드는 계속해서 큐에 작업을 추가(생산)
    listener_thread = threading.Thread(target=socket_listener, args=(rfid_queue,), daemon=True)
    listener_thread.start()

    with hpf.VDevice() as target:
        print("[SETUP] Configuring Hailo network groups...")
        yolo_network_group = target.configure(yolo_hef)[0]
        lpr_network_group = target.configure(lpr_hef)[0]
        
        # [수정됨] 큐의 작업을 처리(소비)할 작업자 스레드 생성
        # Hailo 리소스는 하나이므로 작업자 스레드도 하나면 충분합니다.
        processing_thread = threading.Thread(
            target=worker,
            args=(
                rfid_queue, 
                shared_state, 
                camera_threads, 
                yolo_network_group, 
                lpr_network_group, 
                inference_lock
            ),
            daemon=True # 메인 스레드 종료 시 함께 종료
        )
        processing_thread.start()

        print("\n[INFO] 시스템이 준비되었습니다. 작업자 스레드가 RFID 신호를 기다립니다...")

        try:
            # 메인 스레드는 이제 프로그램 종료를 기다리기만 함
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logging.info("Ctrl+C 입력됨. 프로그램을 종료합니다.")
        finally:
            for cam_thread in camera_threads.values():
                cam_thread.stop()
            for cam_thread in camera_threads.values():
                if cam_thread.is_alive():
                    cam_thread.join()
            print("[INFO] Program finished.")

if __name__ == "__main__":
    main()