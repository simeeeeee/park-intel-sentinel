import os
import argparse
import random
import cv2
from PIL import Image
from imutils import paths
from tqdm import tqdm

def parse_arguments():
    """명령줄 인자를 파싱합니다."""
    parser = argparse.ArgumentParser(description="LPRNet 모델을 위한 Hailo 보정 데이터셋 생성 스크립트")
    
    parser.add_argument("--data_dir", type=str, required=True, 
                        help="보정 데이터를 만들 원본 이미지들이 있는 디렉토리 경로 (e.g., ./data)")
    parser.add_argument("--target_dir", type=str, default="lpr_calib_rgb", 
                        help="처리된 보정 이미지들을 저장할 디렉토리 경로")
    parser.add_argument("--image_size", nargs=2, type=int, default=[300, 75], 
                        help="모델 입력 크기 (너비 높이 순)")
    parser.add_argument("--num_images", type=int, default=1024, 
                        help="생성할 보정 이미지의 개수")
    parser.add_argument("--image_extensions", nargs='+', type=str, default=['.jpg', '.jpeg', '.png'],
                        help="찾을 이미지 파일 확장자 목록")

    return parser.parse_args()

def main():
    args = parse_arguments()
    target_w, target_h = args.image_size

    # 타겟 디렉토리 생성
    os.makedirs(args.target_dir, exist_ok=True)
    print(f"결과물이 저장될 폴더: '{os.path.abspath(args.target_dir)}'")

    # 모든 이미지 파일 경로 탐색
    all_image_paths = list(paths.list_images(args.data_dir))
    if not all_image_paths:
        print(f"오류: '{args.data_dir}' 에서 이미지를 찾을 수 없습니다.")
        return
    print(f"총 {len(all_image_paths)}개의 이미지를 찾았습니다.")

    # 필요한 개수만큼 이미지 랜덤 선택
    random.shuffle(all_image_paths)
    images_to_process = all_image_paths[:args.num_images]
    
    if len(images_to_process) < args.num_images:
        print(f"경고: 요청하신 {args.num_images}개보다 적은 {len(images_to_process)}개의 이미지만 사용합니다.")

    print(f"총 {len(images_to_process)}개의 이미지로 RGB 순서의 보정 데이터셋을 생성합니다...")
    
    for idx, filepath in tqdm(enumerate(images_to_process), total=len(images_to_process)):
        try:
            # 1. OpenCV로 이미지 읽기 (BGR 순서)
            img_bgr = cv2.imread(filepath)
            if img_bgr is None:
                print(f"경고: 파일을 읽을 수 없습니다 - {filepath}")
                continue
            
            # 2. RGB 순서로 변환 (핵심 단계!)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            # 3. LPRNet 학습 방식과 동일하게 강제 리사이즈
            resized_img = cv2.resize(img_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            
            # 4. Pillow를 사용하여 RGB 순서 그대로 저장
            pil_img = Image.fromarray(resized_img)
            output_filename = f"calib_img_{idx:04d}.png"
            output_filepath = os.path.join(args.target_dir, output_filename)
            pil_img.save(output_filepath, format="PNG")

        except Exception as e:
            print(f"오류 발생 ({filepath}): {e}")
            continue
    
    print("\n보정 데이터셋 생성이 완료되었습니다!")
    print(f"이제 '{args.target_dir}' 폴더를 HEF 컴파일 시 --calib-path 인자로 사용하세요.")

if __name__ == "__main__":
    main()