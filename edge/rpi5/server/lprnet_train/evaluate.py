import torch
import torch.nn as nn  # <<< 빠졌던 핵심 import 문
import torch.nn.functional as F
import numpy as np
import cv2
import argparse
import os
from imutils import paths
from tqdm import tqdm

# =================================================================================
# 1. 모델 및 상수 정의 (학습 스크립트와 동일하게 유지)
# =================================================================================
# 이 부분은 .pt 파일(모델 구조 포함)을 올바르게 불러오기 위해 반드시 필요합니다.

CHARS = [
    '가', '나', '다', '라', '마', '거', '너', '더', '러', '머', '버', '서', '어', '저',
    '고', '노', '도', '로', '모', '보', '소', '오', '조',
    '구', '누', '두', '루', '무', '부', '수', '우', '주',
    '아', '바', '사', '자', '배', '허', '하', '호', '국',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-'
]

class small_basic_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(small_basic_block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out // 4, kernel_size=1), nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(3, 1), padding=(1, 0)), nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(1, 3), padding=(0, 1)), nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out, kernel_size=1),
        )
    def forward(self, x): return self.block(x)

class LPRNet_Hailo(nn.Module):
    def __init__(self, lpr_max_len, class_num, dropout_rate):
        super(LPRNet_Hailo, self).__init__()
        self.lpr_max_len = lpr_max_len; self.class_num = class_num
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=(3, 5)), 
            small_basic_block(64, 128), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=(2, 3)),
            small_basic_block(128, 256), nn.BatchNorm2d(256), nn.ReLU(),
            small_basic_block(256, 256), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4, 2), stride=(2, 1)),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, 256, kernel_size=(4, 1), stride=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, len(CHARS), kernel_size=(1, 13), stride=1, padding=(0, 6)),
        )
    def forward(self, x): return self.backbone(x)

# =================================================================================
# 2. 평가 로직 (이하 코드는 변경 없음)
# =================================================================================

def ctc_greedy_decode(logits):
    """
    Greedy CTC 디코더. 모델의 로짓 출력을 최종 텍스트로 변환합니다.
    Args:
        logits (torch.Tensor): 모델의 로짓 출력 (Width, Num_Classes)
    """
    best_path = torch.argmax(logits, dim=1)
    blank_idx = len(CHARS) - 1
    
    # 1. 연속된 중복 문자 제거
    collapsed_indices = []
    prev_idx = -1
    for idx in best_path:
        if idx.item() != prev_idx:
            collapsed_indices.append(idx.item())
        prev_idx = idx.item()
        
    # 2. 'blank' 토큰 제거
    final_text = "".join([CHARS[idx] for idx in collapsed_indices if idx != blank_idx])
            
    return final_text

def evaluate_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model from {args.model_path}...")
    try:
        model = torch.load(args.model_path, map_location=device, weights_only=False)
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    model.eval()

    image_paths = list(paths.list_images(args.data_dir))
    if not image_paths:
        print(f"❌ No images found in directory: {args.data_dir}")
        return
        
    print(f"Found {len(image_paths)} images. Starting evaluation...")

    correct_count = 0
    incorrect_examples = []

    for image_path in tqdm(image_paths, desc="Evaluating"):
        basename = os.path.basename(image_path)
        
        # <<< 수정된 부분: 파일 확장자를 먼저 제거합니다. >>>
        imgname, _ = os.path.splitext(basename)
        ground_truth = imgname.split("-")[0].split("_")[0]
        # ---------------------------------------------

        img = cv2.imread(image_path)
        # ... (이하 코드는 변경 없음) ...
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (args.img_width, args.img_height))
        img = img.astype('float32')
        img -= 127.5
        img *= 0.0078125
        img = np.transpose(img, (2, 0, 1))
        
        img_tensor = torch.from_numpy(img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits_map = model(img_tensor)
            logits_seq = torch.mean(logits_map, dim=2)
            logits_for_decode = logits_seq.permute(2, 0, 1).squeeze(1)
            predicted_text = ctc_greedy_decode(logits_for_decode)
        
        if predicted_text == ground_truth:
            correct_count += 1
        else:
            if len(incorrect_examples) < 10:
                incorrect_examples.append(f"  - File: {basename} | GT: '{ground_truth}' | Pred: '{predicted_text}'")

    # 6. 최종 결과 출력
    total_images = len(image_paths)
    accuracy = (correct_count / total_images) * 100 if total_images > 0 else 0

    print("\n--- 📊 Evaluation Results ---")
    print(f"Total Images:    {total_images}")
    print(f"Correct:         {correct_count}")
    print(f"Incorrect:       {total_images - correct_count}")
    print(f"Accuracy:        {accuracy:.2f}%")
    print("----------------------------")
    
    if accuracy > 95:
        print("🏆 Excellent performance!")
    elif accuracy > 90:
        print("👍 Good performance.")
    elif accuracy > 80:
        print("🙂 Decent performance, but could be improved.")
    else:
        print("🤔 Needs more training or data augmentation.")

    if incorrect_examples:
        print("\n📝 Examples of Incorrect Predictions:")
        for example in incorrect_examples:
            print(example)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate LPRNet model performance")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model .pt file')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the evaluation data directory')
    # 모델 학습 시 사용한 이미지 크기와 동일하게 설정
    parser.add_argument('--img_width', type=int, default=300, help='Image width for resize')
    parser.add_argument('--img_height', type=int, default=75, help='Image height for resize')
    
    args = parser.parse_args()
    evaluate_model(args)