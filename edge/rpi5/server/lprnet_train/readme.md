python train.py --pretrained_model ./path/to/official_weights.pth
python inference.py --weights ./weights/lprnet_epoch_50.pth --image_path ./test_images/02라3170.jpg


      
# 기본값으로 50 에포크, 배치 사이즈 128로 학습
python train.py --train_dir ./lpr_dataset

# 에포크와 배치 사이즈를 직접 지정하여 학습
python train.py --train_dir ./data --epochs 100 --batch_size 16

# 사전 학습된 가중치(state_dict)로 전이 학습을 시작할 경우
python train.py --train_dir ./data --epochs 100 --pretrained_model ./Final_LPRNet_model.pth


python inference.py --weights ./weights/lprnet_epoch_100.pt --image_path ./test_images/02라3170.jpg


      
# --train_dir에 실제 데이터셋 경로를 지정합니다.
# 이미지 크기는 자동으로 300x75로 설정됩니다.
python train.py --mode train --train_dir ./lpr_dataset --epochs 50 --batch_size 32

# --pt_file에 방금 생성된 모델 파일 경로를 지정합니다.
python train.py --mode export --pt_file ./weights/lprnet_epoch_50.pt --onnx_file my_lprnet.onnx 


      
# --train_dir에 실제 데이터셋 경로를 지정합니다.
# 이미지 크기는 300x75로 고정됩니다.
python train.py --mode train --train_dir ./lpr_dataset --epochs 50 --batch_size 32

python train.py --mode export --pt_file ./weights/lprnet_hailo_epoch_50.pt


# YAML 파일의 network_path를 'lprnet_hailo.onnx'로 수정한 후 실행
hailomz compile --ckpt lprnet_hailo.onnx --calib-path /path/to/calib/ --yaml /path/to/your/lpr.yaml --classes 53 --hw-arch hailo8



------
nvidia-smi
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118