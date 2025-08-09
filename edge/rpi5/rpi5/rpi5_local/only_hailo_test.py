# -*- coding: utf-8 -*-

# 0. 필요한 라이브러리 임포트
import hailo_platform as hpf
import numpy as np
import time
import logging
import os

# --- 로깅 기본 설정 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 1. 사용자 설정 ---
# TODO: 여기에 사용하실 커스텀 모델 경로를 입력하세요.
MODEL_1_HEF_PATH = "yolov8m.hef"
MODEL_2_HEF_PATH = "lprnet_test.hef"

# 성능 평가에 사용할 추론 횟수
NUM_INFERENCES = 200
# 본 측정 전, 하드웨어 예열을 위한 추론 횟수
NUM_WARMUP_INFERENCES = 10

def benchmark_model(target: hpf.VDevice, hef_path: str):
    """
    주어진 HEF 모델의 성능을 측정하고 평균 추론 시간과 FPS를 반환합니다.
    """
    if not os.path.exists(hef_path):
        logging.error(f"[ERROR] 모델 파일 '{hef_path}'를 찾을 수 없습니다. 경로를 확인해주세요.")
        return None, None

    try:
        logging.info(f"--- 모델 '{hef_path}' 성능 측정을 시작합니다. ---")

        # 1. 모델 로드 및 정보 확인
        hef = hpf.HEF(hef_path)
        input_vstream_info = hef.get_input_vstream_infos()[0]
        output_vstream_info = hef.get_output_vstream_infos()[0]
        input_shape = input_vstream_info.shape
        input_dtype = np.uint8

        logging.info(f"모델 입력 형태 (배치 제외): {input_shape}, 데이터 타입: {input_dtype}")

        # 2. 네트워크 그룹 및 VStream 파라미터 설정
        network_group = target.configure(hef)[0]
        
        # === 오류 수정된 부분 1: 출력 파라미터 수정 ===
        # 입력은 양자화된 UINT8, 출력은 비양자화된 FLOAT32로 설정합니다.
        # 이것이 대부분의 모델에서 사용하는 일반적인 설정입니다.
        input_params = hpf.InputVStreamParams.make_from_network_group(network_group, quantized=True, format_type=hpf.FormatType.UINT8)
        output_params = hpf.OutputVStreamParams.make_from_network_group(network_group, quantized=False, format_type=hpf.FormatType.FLOAT32)

        # === 오류 수정된 부분 2: 입력 데이터 형태 수정 ===
        # 배치 차원(1)을 추가하여 4D 텐서로 만듭니다. e.g., (75, 300, 3) -> (1, 75, 300, 3)
        single_frame_data = (np.random.rand(*input_shape) * 255).astype(input_dtype)
        batched_input_data = np.expand_dims(single_frame_data, axis=0)
        
        infer_payload = {input_vstream_info.name: batched_input_data}

        # 4. 전체 벤치마크 동안 네트워크를 활성화
        with network_group.activate():
            # 5. 추론 파이프라인(InferVStreams) 생성
            with hpf.InferVStreams(network_group, input_params, output_params) as infer_pipeline:

                # 6. 워밍업(Warm-up) 추론 실행
                logging.info(f"워밍업 추론을 {NUM_WARMUP_INFERENCES}회 실행합니다...")
                for _ in range(NUM_WARMUP_INFERENCES):
                    infer_pipeline.infer(infer_payload)

                # 7. 성능 측정 시작
                inference_times = []
                logging.info(f"성능 측정을 위한 추론을 {NUM_INFERENCES}회 실행합니다...")
                for _ in range(NUM_INFERENCES):
                    start_time = time.perf_counter()
                    infer_pipeline.infer(infer_payload)
                    end_time = time.perf_counter()
                    duration_ms = (end_time - start_time) * 1000
                    inference_times.append(duration_ms)

        # 8. 결과 계산
        avg_time_ms = sum(inference_times) / len(inference_times)
        fps = 1000 / avg_time_ms

        logging.info(f"--- 모델 '{hef_path}' 성능 측정 완료 ---")
        return avg_time_ms, fps

    except Exception as e:
        logging.error(f"[ERROR] '{hef_path}' 처리 중 예외 발생: {e}", exc_info=True)
        return None, None

def main():
    """메인 실행 함수"""
    results = {}

    try:
        with hpf.VDevice() as target:
            logging.info("Hailo 디바이스를 활성화하고 벤치마크를 시작합니다.")

            # 모델 1 성능 측정
            avg_time_1, fps_1 = benchmark_model(target, MODEL_1_HEF_PATH)
            if avg_time_1 is not None:
                results[MODEL_1_HEF_PATH] = {'avg_time_ms': avg_time_1, 'fps': fps_1}

            # 모델 2 성능 측정
            avg_time_2, fps_2 = benchmark_model(target, MODEL_2_HEF_PATH)
            if avg_time_2 is not None:
                results[MODEL_2_HEF_PATH] = {'avg_time_ms': avg_time_2, 'fps': fps_2}

    except Exception as e:
        logging.error(f"Hailo 디바이스 초기화 또는 접근 중 오류 발생: {e}")

    # 최종 결과 출력
    print("\n" + "="*60)
    print("           Hailo 모델 성능 벤치마크 최종 결과")
    print("="*60)

    if not results:
        print("측정된 성능 결과가 없습니다.")
        print("HEF 파일 경로와 Hailo 장치 연결 상태를 확인해주세요.")
    else:
        print(f"{'모델 파일':<30} | {'평균 시간 (ms)':<15} | {'성능 (FPS)':<10}")
        print("-" * 60)
        for model_path, data in results.items():
            print(f"{model_path:<30} | {data['avg_time_ms']:<15.2f} | {data['fps']:<10.2f}")

    print("="*60)
    logging.info("벤치마크 프로그램을 종료합니다.")


if __name__ == "__main__":
    main()