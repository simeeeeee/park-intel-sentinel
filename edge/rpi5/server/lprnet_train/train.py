import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import random_split

import os
import numpy as np
import random
import cv2
from imutils import paths
import argparse
from tqdm import tqdm

# =================================================================================
# 1. 상수 및 데이터 로더 정의
# =================================================================================

CHARS = [
    '가', '나', '다', '라', '마', '거', '너', '더', '러', '머', '버', '서', '어', '저',
    '고', '노', '도', '로', '모', '보', '소', '오', '조',
    '구', '누', '두', '루', '무', '부', '수', '우', '주',
    '아', '바', '사', '자', '배', '허', '하', '호', '국',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-'
]
CHARS_DICT = {char:i for i, char in enumerate(CHARS)}
# CTC Loss를 위해 blank 토큰을 마지막에 추가
NUM_CLASSES = len(CHARS) + 1

class LPRDataLoader(Dataset):
    def __init__(self, img_dir, imgSize, lpr_max_len):
        self.img_dir = img_dir; self.img_paths = []
        if not isinstance(img_dir, list): img_dir = [img_dir]
        for i in range(len(img_dir)): self.img_paths += [el for el in paths.list_images(img_dir[i])]
        random.shuffle(self.img_paths); self.img_size = imgSize; self.lpr_max_len = lpr_max_len
    def __len__(self): return len(self.img_paths)
    def __getitem__(self, index):
        filename = self.img_paths[index]
        try:
            Image = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
            if Image.shape[0:2] != (self.img_size[1], self.img_size[0]): Image = cv2.resize(Image, self.img_size)
            Image = self.transform(Image)
            basename = os.path.basename(filename); imgname, _ = os.path.splitext(basename)
            imgname = imgname.split("-")[0].split("_")[0]
            label = [CHARS_DICT[c] for c in imgname if c in CHARS_DICT]
            if len(label) == 0: return self.__getitem__(random.randint(0, self.__len__() - 1))
            return Image, label, len(label)
        except Exception as e:
            print(f"Error loading image {filename}: {e}"); return self.__getitem__(random.randint(0, self.__len__() - 1))
    def transform(self, img):
        img = img.astype('float32'); img -= 127.5; img *= 0.0078125
        return np.transpose(img, (2, 0, 1))

def collate_fn(batch):
    images, labels, label_lengths = [], [], []
    for sample in batch:
        if sample is None: continue
        images.append(torch.from_numpy(sample[0])); labels.extend(sample[1]); label_lengths.append(sample[2])
    if not images: return None, None, None
    return torch.stack(images, 0), torch.IntTensor(labels), torch.IntTensor(label_lengths)

# =================================================================================
# 2. LPRNet 모델 정의
# =================================================================================

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
            nn.Conv2d(256, self.class_num, kernel_size=(1, 13), stride=1, padding=(0, 6)),
        )
    
    def forward(self, x):
        # <<< 수정된 부분 >>>
        # Hailo 컴파일을 위해 모델은 (Batch, Class, Height, Width) 3D 특징 맵을 그대로 반환합니다.
        # CTC Loss 계산에 필요한 차원 축소(torch.mean)는 학습/검증 루프에서 처리합니다.
        return self.backbone(x)

# =================================================================================
# 3. 학습/검증/변환 함수
# =================================================================================

def ctc_decode(logits):
    """Greedy CTC Decoder"""
    predictions = []
    # logits: (batch, seq_len, num_classes)
    for logit_seq in logits: 
        best_path = torch.argmax(logit_seq, dim=1)
        blank_idx = NUM_CLASSES - 1
        decoded_chars = []
        prev_char_idx = -1
        for idx in best_path:
            if idx.item() != prev_char_idx:
                if idx.item() != blank_idx: decoded_chars.append(CHARS[idx.item()])
            prev_char_idx = idx.item()
        predictions.append("".join(decoded_chars))
    return predictions

def label_to_string(labels, label_lengths):
    """정수 레이블 배치를 문자열 리스트로 변환"""
    strings = []
    start = 0
    for length in label_lengths:
        end = start + length
        strings.append("".join([CHARS[c] for c in labels[start:end]]))
        start = end
    return strings

def validate(model, val_loader, criterion, device):
    """모델 검증 및 정확도 계산"""
    model.eval()
    total_val_loss, total_correct, total_samples = 0, 0, 0
    pbar_val = tqdm(val_loader, desc="Validating", leave=False)
    with torch.no_grad():
        for i, data in enumerate(pbar_val):
            if data[0] is None: continue
            images, labels, label_lengths = data
            images, labels, label_lengths = images.to(device), labels.to(device), label_lengths.to(device)
            total_samples += images.size(0)
            
            # <<< 수정된 부분 >>>
            # 모델에서 3D 특징 맵을 받은 후, CTC Loss 계산을 위해 높이(H) 차원을 평균냅니다.
            logits_map = model(images)  # (B, C, H, W)
            logits_seq = torch.mean(logits_map, dim=2) # (B, C, W)
            
            # Loss 계산
            log_probs = F.log_softmax(logits_seq.permute(2, 0, 1), dim=2) # CTC Loss 입력 형태: (W, B, C)
            input_lengths = torch.IntTensor([log_probs.size(0)] * images.size(0))
            loss = criterion(log_probs, labels, input_lengths, label_lengths)
            if torch.isfinite(loss): total_val_loss += loss.item()

            # 정확도 계산
            decoded_preds = ctc_decode(logits_seq.permute(0, 2, 1)) # Decoder 입력 형태: (B, W, C)
            true_strings = label_to_string(labels.cpu().numpy(), label_lengths.cpu().numpy())
            for pred, true in zip(decoded_preds, true_strings):
                if pred == true: total_correct += 1
            
            # 첫 배치 샘플 출력
            if i == 0:
                print("\n--- Validation Samples ---")
                for j in range(min(4, len(true_strings))):
                    print(f"Pred: {decoded_preds[j]:<10} | True: {true_strings[j]}")
                print("------------------------")

    return total_val_loss / len(val_loader), total_correct / total_samples

def train_and_validate(args):
    """모델 학습 및 검증 메인 함수"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LPRNet_Hailo(lpr_max_len=18, class_num=NUM_CLASSES, dropout_rate=args.dropout_rate).to(device)

    if args.pretrained_model and os.path.exists(args.pretrained_model):
        print(f"Loading pre-trained weights from {args.pretrained_model}")
        model.load_state_dict(torch.load(args.pretrained_model, map_location=device), strict=False)

    full_dataset = LPRDataLoader(args.data_dir, (args.img_width, args.img_height), 18)
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

    criterion = nn.CTCLoss(blank=NUM_CLASSES - 1, reduction='mean', zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)

    print("--- Training Start ---")
    best_val_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0
        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for data in pbar_train:
            if data[0] is None: continue
            images, labels, label_lengths = data
            images, labels, label_lengths = images.to(device), labels.to(device), label_lengths.to(device)
            
            # <<< 수정된 부분 >>>
            # 검증 로직과 동일하게 학습 시에도 높이 차원을 평균내어 Loss를 계산합니다.
            logits_map = model(images)
            logits_seq = torch.mean(logits_map, dim=2)
            
            log_probs = F.log_softmax(logits_seq.permute(2, 0, 1), dim=2)
            input_lengths = torch.IntTensor([log_probs.size(0)] * images.size(0))
            
            optimizer.zero_grad()
            loss = criterion(log_probs, labels, input_lengths, label_lengths)
            if torch.isfinite(loss):
                loss.backward(); optimizer.step(); total_train_loss += loss.item()
            pbar_train.set_postfix(loss=f"{loss.item():.4f}")
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step(val_acc)

        print(f"Epoch [{epoch+1}/{args.epochs}] -> Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if not os.path.exists(args.save_dir): os.makedirs(args.save_dir)
            save_path = os.path.join(args.save_dir, "lprnet_hailo_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"✅ Best model saved to {save_path} (Val Acc: {best_val_acc:.4f})")

    print("--- Training finished ---")

def export_to_onnx(args):
    """.pth 파일을 ONNX 형식으로 변환"""
    if not args.weight_to_export: raise ValueError("--weight_to_export 경로를 지정해야 합니다.")
    if not os.path.exists(args.weight_to_export): raise FileNotFoundError(f"Weight file not found at {args.weight_to_export}")
    
    device = torch.device("cpu")
    model = LPRNet_Hailo(lpr_max_len=18, class_num=NUM_CLASSES, dropout_rate=0).to(device)
    
    print(f"Loading model from {args.weight_to_export}")
    model.load_state_dict(torch.load(args.weight_to_export, map_location=device))
    model.eval()

    dummy_input = torch.randn(1, 3, args.img_height, args.img_width, device=device)
    if not args.onnx_file:
        onnx_filename = os.path.splitext(os.path.basename(args.weight_to_export))[0] + ".onnx"
        args.onnx_file = os.path.join(args.save_dir, onnx_filename)

    print(f"Exporting model to {args.onnx_file}...")
    torch.onnx.export(model, dummy_input, args.onnx_file, export_params=True, opset_version=11, do_constant_folding=True,
                      input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
    print(f"✅ Export successful. Final ONNX output shape will be (batch, {NUM_CLASSES}, H, W)")


# =================================================================================
# 4. 메인 실행 함수
# =================================================================================

def main():
    parser = argparse.ArgumentParser(description="LPRNet Training and ONNX Export for Hailo")
    parser.add_argument("--train", action='store_true', help="Train the model")
    parser.add_argument("--export", action='store_true', help="Export the model to ONNX")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory containing all image data")
    parser.add_argument("--save_dir", type=str, default="./weights", help="Directory to save checkpoints and ONNX file")
    parser.add_argument("--pretrained_model", type=str, default="", help="Path to pre-trained .pth file for fine-tuning")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--dropout_rate", type=float, default=0.5, help="Dropout rate during training")
    parser.add_argument("--val_split", type=float, default=0.1, help="Proportion of the dataset to use for validation")
    parser.add_argument("--weight_to_export", type=str, default="", help="Path to the .pth file to be exported to ONNX")
    parser.add_argument("--onnx_file", type=str, default="", help="Output path for the ONNX file")
    parser.add_argument("--img_width", type=int, default=300, help="Image width (fixed for Hailo)")
    parser.add_argument("--img_height", type=int, default=75, help="Image height (fixed for Hailo)")
    args = parser.parse_args()

    if args.train:
        train_and_validate(args)
    if args.export:
        if not args.weight_to_export and args.train:
            args.weight_to_export = os.path.join(args.save_dir, "lprnet_hailo_best.pth")
        export_to_onnx(args)
    if not args.train and not args.export:
        print("Please specify a mode: --train or --export")

if __name__ == '__main__':
    main()