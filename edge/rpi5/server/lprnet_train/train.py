import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F

import os
import numpy as np
import random
import cv2
from imutils import paths
import argparse

# =================================================================================
# 1. 상수 및 모델/데이터 로더 정의 (이 부분은 변경 없음)
# =================================================================================

CHARS = [
    '가', '나', '다', '라', '마', '거', '너', '더', '러', '머', '버', '서', '어', '저',
    '고', '노', '도', '로', '모', '보', '소', '오', '조',
    '구', '누', '두', '루', '무', '부', '수', '우', '주',
    '아', '바', '사', '자', '배', '허', '하', '호', '국',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-'
]
CHARS_DICT = {char:i for i, char in enumerate(CHARS)}
NUM_CLASSES = len(CHARS)

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
    def forward(self, x): return self.backbone(x)

class LPRDataLoader(Dataset):
    def __init__(self, img_dir, imgSize, lpr_max_len):
        self.img_dir = img_dir; self.img_paths = [];
        if not isinstance(img_dir, list): img_dir = [img_dir]
        for i in range(len(img_dir)): self.img_paths += [el for el in paths.list_images(img_dir[i])]
        random.shuffle(self.img_paths); self.img_size = imgSize; self.lpr_max_len = lpr_max_len
    def __len__(self): return len(self.img_paths)
    def __getitem__(self, index):
        filename = self.img_paths[index]; Image = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
        if Image.shape[0:2] != (self.img_size[1], self.img_size[0]): Image = cv2.resize(Image, self.img_size)
        Image = self.transform(Image)
        basename = os.path.basename(filename); imgname, _ = os.path.splitext(basename)
        imgname = imgname.split("-")[0].split("_")[0]
        label = [CHARS_DICT[c] for c in imgname if c in CHARS_DICT]
        return Image, label, len(label)
    def transform(self, img):
        img = img.astype('float32'); img -= 127.5; img *= 0.0078125
        return np.transpose(img, (2, 0, 1))

def collate_fn(batch):
    images, labels, label_lengths = [], [], []
    for sample in batch:
        images.append(torch.from_numpy(sample[0])); labels.extend(sample[1]); label_lengths.append(sample[2])
    return torch.stack(images, 0), torch.IntTensor(labels), torch.IntTensor(label_lengths)


# =================================================================================
# 2. 메인 함수 수정
# =================================================================================

def main():
    parser = argparse.ArgumentParser(description="Hailo-Ready LPRNet Training and Export")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'export'], help='Run mode')
    parser.add_argument('--train_dir', type=str, default='', help='(train) Path to dataset')
    parser.add_argument('--weight_dir', type=str, default='./weights', help='Directory for model files')
    parser.add_argument('--pt_file', type=str, default='', help='(export) Path to the .pt model file')
    
    # <<< 수정된 부분: '--pretrained_model' 인자 추가 >>>
    parser.add_argument('--pretrained_model', type=str, default='', help='(train) Path to pre-trained model weights for transfer learning.')
    
    parser.add_argument('--onnx_file', type=str, default='lprnet_hailo.onnx', help='(export) Output ONNX path')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--img_width', type=int, default=300, help='Image width (fixed for Hailo)')
    parser.add_argument('--img_height', type=int, default=75, help='Image height (fixed for Hailo)')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate')
    
    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'export':
        export_to_onnx(args)

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LPRNet_Hailo(lpr_max_len=18, class_num=NUM_CLASSES, dropout_rate=args.dropout_rate).to(device)
    
    # <<< 수정된 부분: 사전 학습된 가중치를 불러오는 로직 추가 >>>
    if args.pretrained_model:
        print(f"Loading pre-trained weights from {args.pretrained_model}")
        try:
            # .pth 파일은 state_dict이므로 load_state_dict를 사용합니다.
            # strict=False는 불러올 가중치와 모델 구조가 완벽히 일치하지 않아도 에러를 내지 않습니다.
            model.load_state_dict(torch.load(args.pretrained_model, map_location=device), strict=False)
            print("✅ Pre-trained weights loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading pre-trained weights: {e}")
            return # 가중치 로드 실패 시 종료

    train_dataset = LPRDataLoader(args.train_dir, (args.img_width, args.img_height), 18)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    
    criterion = nn.CTCLoss(blank=CHARS_DICT['-'], reduction='mean', zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print(f"--- Training Hailo-Ready LPRNet (Input: {args.img_height}x{args.img_width}) ---")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for i, (images, labels, label_lengths) in enumerate(train_loader):
            images, labels, label_lengths = images.to(device), labels.to(device), label_lengths.to(device)
            logits_map = model(images)
            logits_seq = torch.mean(logits_map, dim=2)
            log_probs = F.log_softmax(logits_seq.permute(2, 0, 1), dim=2)
            input_lengths = torch.IntTensor([log_probs.size(0)] * images.size(0))
            
            optimizer.zero_grad()
            loss = criterion(log_probs, labels, input_lengths, label_lengths)
            if torch.isfinite(loss): loss.backward(); optimizer.step(); total_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{args.epochs}] - Avg Loss: {total_loss / len(train_loader):.4f}")

        if not os.path.exists(args.weight_dir): os.makedirs(args.weight_dir)
        save_path = os.path.join(args.weight_dir, f"lprnet_hailo_epoch_{epoch+1}.pt")
        # 모델 전체를 저장합니다.
        torch.save(model, save_path)
        print(f"Model saved to {save_path}")

def export_to_onnx(args):
    """ .pt 파일을 로드하여 ONNX 형식으로 변환 """
    if not args.pt_file: raise ValueError("--pt_file must be specified.")
    
    device = torch.device("cpu")
    print(f"Loading model from {args.pt_file}")
    
    # <<< 수정된 부분: weights_only=False 옵션 추가 >>>
    try:
        model = torch.load(args.pt_file, map_location=device, weights_only=False)
    except AttributeError:
        # 이전 버전 PyTorch에서 저장된 파일과의 호환성을 위해
        model = torch.load(args.pt_file, map_location=device)

    model.eval()

    dummy_input = torch.randn(1, 3, args.img_height, args.img_width, device=device)

    print(f"Exporting model to {args.onnx_file}...")
    torch.onnx.export(
        model,
        dummy_input,
        args.onnx_file,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("Export successful.")


if __name__ == '__main__':
    main()