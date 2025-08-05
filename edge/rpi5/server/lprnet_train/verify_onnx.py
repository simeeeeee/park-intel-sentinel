import onnxruntime as ort
import numpy as np
import cv2
import argparse
import os
from imutils import paths
from tqdm import tqdm

# --- 1. 상수 및 전/후처리 함수 정의 (HEF 평가 스크립트와 동일) ---
LPR_CHARACTERS = [
    '가', '나', '다', '라', '마', '거', '너', '더', '러', '머', '버', '서', '어', '저',
    '고', '노', '도', '로', '모', '보', '소', '오', '조',
    '구', '누', '두', '루', '무', '부', '수', '우', '주',
    '아', '바', '사', '자', '배', '허', '하', '호', '국',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-'
]

def preprocess_lpr_image(image, input_shape):
    # input_shape: (height, width)
    input_h, input_w = input_shape
    resized_img = cv2.resize(image, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
    rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    # 정규화 후 채널 순서 변경 (H, W, C) -> (C, H, W)
    normalized_img = (rgb_img.astype(np.float32) - 127.5) * 0.0078125
    transposed_img = np.transpose(normalized_img, (2, 0, 1))
    # 배치 차원 추가
    return np.expand_dims(transposed_img, axis=0)

def ctc_greedy_decode(raw_logits, characters):
    # ONNX Runtime 출력은 (Batch, C, H, W) -> 예: (1, 52, 5, 19)
    logits_map = np.squeeze(raw_logits, axis=0) # (C, H, W)
    # (C, H, W) -> (H, W, C) -> (W, C)
    logits_seq = np.mean(np.transpose(logits_map, (1, 2, 0)), axis=0)
    
    best_path_indices = np.argmax(logits_seq, axis=1)
    blank_idx = len(characters) - 1
    decoded_indices = []
    prev_idx = -1
    for idx in best_path_indices:
        if idx != prev_idx:
            if idx != blank_idx:
                decoded_indices.append(idx)
        prev_idx = idx
    final_text = "".join([characters[i] for i in decoded_indices])
    return final_text

# --- 2. ONNX 모델 평가 메인 함수 ---
def verify_onnx_model(args):
    print(f"[SETUP] Loading ONNX model from {args.onnx_path}...")
    try:
        # ONNX Runtime 세션 생성
        session = ort.InferenceSession(args.onnx_path, providers=['CPUExecutionProvider'])
        print("✅ ONNX model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading ONNX model: {e}")
        return

    # 모델의 입력/출력 이름 자동 감지
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    # ONNX는 (Batch, C, H, W) 순서이므로, H, W를 가져옴
    _, _, input_h, input_w = session.get_inputs()[0].shape

    image_paths = list(paths.list_images(args.data_dir))
    if not image_paths:
        print(f"❌ No images found in directory: {args.data_dir}")
        return
        
    print(f"Found {len(image_paths)} images. Starting verification...")

    correct_count = 0
    
    for image_path in tqdm(image_paths, desc="Verifying ONNX model"):
        # 파일명에서 정답 추출
        basename = os.path.basename(image_path)
        imgname, _ = os.path.splitext(basename)
        ground_truth = imgname.split("-")[0].split("_")[0]
        
        # 이미지 로드 및 전처리
        image = cv2.imread(image_path)
        input_tensor = preprocess_lpr_image(image, (input_h, input_w))
        
        # ONNX Runtime으로 추론 실행
        raw_logits_map = session.run([output_name], {input_name: input_tensor})[0]
        
        # 후처리 (디코딩) 및 비교
        predicted_text = ctc_greedy_decode(raw_logits_map, LPR_CHARACTERS)
        if predicted_text == ground_truth:
            correct_count += 1

    # 최종 결과 출력
    total_images = len(image_paths)
    accuracy = (correct_count / total_images) * 100 if total_images > 0 else 0
    print("\n--- 📊 ONNX Model Verification Results ---")
    print(f"Total Images:    {total_images}")
    print(f"Correct:         {correct_count}")
    print(f"Accuracy:        {accuracy:.2f}%")
    print("------------------------------------------")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Verify LPRNet ONNX model performance")
    parser.add_argument('--onnx_path', type=str, required=True, help='Path to the LPRNet .onnx file')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the evaluation data directory (e.g., ./data)')
    
    args = parser.parse_args()
    verify_onnx_model(args)