#!/usr/bin/env python3
# encoding: utf-8
# @data:2023/03/11
# @author:aiden
# lane detection for autonomous driving
import os
import cv2
import math
import queue
import time
import threading
import numpy as np
from cv_bridge import CvBridge
import sdk.common as common
import rclpy
from sensor_msgs.msg import Image
from rclpy.executors import SingleThreadedExecutor

# CvBridge 인스턴스
bridge = CvBridge()
lab_data = common.get_yaml_data("/home/ubuntu/software/lab_tool/lab_config.yaml")
# --- 고정된 HSV 값 (사용자 요청에 따라 설정) ---
h_min, s_min, v_min = 1, 17, 57
h_max, s_max, v_max = 57, 255, 255

class LaneDetector(object):
    def __init__(self, color):
        self.target_color = color # 현재 코드에서는 이 변수를 직접 색상 필터링에 사용하지 않습니다.
        
        # 다중 ROI 설정: (y_start, y_end, x_start, x_end, weight)
        # 이 값들은 카메라 및 환경에 따라 조정이 필요할 수 있습니다.
        # 640x480 해상도를 가정하며, 이에 맞춰 ROI 범위 조정.
        if os.environ.get('DEPTH_CAMERA_TYPE') == 'ascamera':
            # a 카메라에 맞는 ROI 예시 (차선이 보일 것으로 예상되는 하단 영역에 집중)
            self.rois = (
                (380, 480, 0, 640, 0.7),  # 가장 하단 (가장 중요)
                (300, 370, 0, 640, 0.2),  # 중간
                (200, 290, 0, 640, 0.1)   # 상단 (가장 가중치 낮음)
            )
        else:
            # 일반 카메라에 맞는 ROI 예시
            self.rois = (
                (450, 480, 0, 640, 0.7),  # 가장 하단 (가장 중요)
                (390, 420, 0, 640, 0.2),  # 중간
                (330, 360, 0, 640, 0.1)   # 상단 (가장 가중치 낮음)
            )
        
        self.weight_sum = sum([roi[4] for roi in self.rois]) # 가중치 합계 계산

    # ROI 설정 함수 (외부에서 ROI 변경 시 사용)
    def set_roi(self, rois_tuple):
        self.rois = rois_tuple
        self.weight_sum = sum([roi[4] for roi in self.rois])

    @staticmethod
    def get_area_max_contour(contours, threshold=30):
        """
        주어진 윤곽선들 중에서 면적이 가장 큰 윤곽선을 반환합니다.
        threshold보다 작은 면적의 윤곽선은 무시합니다.
        """
        contour_area = zip(contours, tuple(map(lambda c: math.fabs(cv2.contourArea(c)), contours)))
        contour_area = tuple(filter(lambda c_a: c_a[1] > threshold, contour_area))
        if len(contour_area) > 0:
            max_c_a = max(contour_area, key=lambda c_a: c_a[1])
            return max_c_a
        return None

    def __call__(self, image,result_image): 
        """
        주어진 이미지에서 다중 ROI를 사용하여 차선 중심 X 좌표를 감지하고,
        결과를 원본 이미지에 시각화한 후 최종 차선 중심 X 좌표를 반환합니다.
        
        Args:
            image (numpy.array): 입력 이미지 (BGR 형식).
                                 rclpy image_callback에서 bgr8로 받으므로 BGR입니다.

        Returns:
            tuple: (result_image, None, final_cx)
                   result_image (numpy.array): 시각화 결과가 그려진 이미지 (BGR 형식)
                   None: 차선 각도 (현재 사용하지 않으므로 None)
                   final_cx (int): 감지된 차선 중심의 X 좌표 (-1이면 감지 실패)
        """
        result_image = image.copy() # 원본 이미지를 복사하여 시각화에 사용
        h_orig, w_orig = image.shape[:2] 
        img_lab = cv2.cvtColor(result_image, cv2.COLOR_RGB2LAB)

        # --- 카메라 이미지 반전 처리 (필요에 따라 주석 해제하여 테스트) ---
        # self_driving.py와 연동 시 카메라 방향에 따라 필요할 수 있음
        # 0: 수직 반전, 1: 수평 반전, -1: 수직+수평 반전
        # result_image = cv2.flip(result_image, 0) # 시각화에도 적용해야 하므로 result_image에 바로 적용
        # image = cv2.flip(image, 0) # 실제 이미지 처리에도 적용
        # result_image = cv2.flip(result_image, 1) 
        # image = cv2.flip(image, 1)
        # result_image = cv2.flip(result_image, -1) 
        # image = cv2.flip(image, -1)
        
        # --- HSV 색상 공간 변환 및 마스크 생성 (요청에 따라 HSV 값 사용) ---
        # image_callback에서 BGR 이미지를 받으므로 BGR -> HSV로 변환
        hsv_image = cv2.cvtColor(img_lab, cv2.COLOR_BGR2HSV) 
        
        lower_hsv = np.array([h_min, s_min, v_min])
        upper_hsv = np.array([h_max, s_max, v_max])
        color_mask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)
        
        # 모폴로지 연산 (잡음 제거 및 영역 연결)
        kernel = np.ones((5,5), np.uint8)
        mask_processed = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
        mask_processed = cv2.morphologyEx(mask_processed, cv2.MORPH_CLOSE, kernel)

        centroid_x_sum = 0
        total_valid_weights = 0
        
        # 각 ROI에 대해 차선 중심 찾기 및 가중 평균
        for roi_y_start, roi_y_end, roi_x_start, roi_x_end, weight in self.rois:
            # ROI 영역 유효성 검사 및 클리핑
            roi_x_start = max(0, min(roi_x_start, w_orig))
            roi_x_end = max(0, min(roi_x_end, w_orig))
            roi_y_start = max(0, min(roi_y_start, h_orig))
            roi_y_end = max(0, min(roi_y_end, h_orig))

            if roi_y_start >= roi_y_end or roi_x_start >= roi_x_end:
                continue

            roi_mask = mask_processed[roi_y_start:roi_y_end, roi_x_start:roi_x_end].copy()
            
            # ROI 영역 시각화 (원본 이미지에 파란색 사각형)
            cv2.rectangle(result_image, (roi_x_start, roi_y_start), (roi_x_end, roi_y_end), (255, 0, 0), 2)

            # 윤곽선 찾기
            contours_tuple = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = contours_tuple[0] if len(contours_tuple) == 2 else contours_tuple[1]
            
            if len(contours) > 0: 
                max_c_a = self.get_area_max_contour(contours)
                if max_c_a is not None:
                    c = max_c_a[0]
                    M = cv2.moments(c)
                      
                    if M['m00'] != 0: 
                        cx_roi = int(M['m10'] / M['m00']) 

                        # 원본 이미지 좌표계로 변환
                        cx_on_full_image = cx_roi + roi_x_start
                        cy_on_full_image = (roi_y_start + roi_y_end) // 2 # ROI 중앙 Y 좌표
                        
                        # 원본 이미지에 ROI 내의 중심점 그리기 (초록색 원)
                        cv2.circle(result_image, (cx_on_full_image, cy_on_full_image), 5, (0, 255, 0), -1) 
                        
                        centroid_x_sum += cx_on_full_image * weight
                        total_valid_weights += weight
        
        final_cx = -1
        if total_valid_weights > 0:
            final_cx = int(centroid_x_sum / total_valid_weights)
            # 최종 중심선 그리기 (원본 이미지에 빨간색 선과 원)
            cv2.line(result_image, (final_cx, 0), (final_cx, h_orig), (0, 0, 255), 2) 
            cv2.circle(result_image, (final_cx, h_orig // 2), 7, (0, 0, 255), -1) 
        
        # angle은 현재 계산하지 않으므로 None 반환
        return result_image, 0, final_cx
    
    def get_binary(self, image):
        # recognize color through LAB space
        img_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)  # convert RGB to LAB
        img_blur = cv2.GaussianBlur(img_lab, (3, 3), 3)  # Gaussian blur denoising
        mask = cv2.inRange(img_blur, tuple(lab_data['lab']['Stereo'][self.target_color]['min']), tuple(lab_data['lab']['Stereo'][self.target_color]['max']))  # 二值化
        eroded = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))  # erode
        dilated = cv2.dilate(eroded, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))  # dilate

        return dilated


# --- FPS 계산 클래스 (변동 없음) ---
class FPS:
    def __init__(self):
        self._start_time = None
        self._num_frames = 0

    def start(self):
        self._start_time = time.time()
        self._num_frames = 0

    def update(self):
        self._num_frames += 1

    def fps(self):
        if self._start_time is None:
            return 0.0
        elapsed_time = time.time() - self._start_time
        if elapsed_time == 0:
            return 0.0
        return self._num_frames / elapsed_time

    def show_fps(self, img):
        if self._start_time is not None:
            fps_val = self.fps()
            cv2.putText(img, f"FPS: {int(fps_val)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return img

# ROS 이미지 큐 및 콜백
image_queue = queue.Queue(2) # 큐 사이즈를 2로 설정

# ROS 이미지 콜백 함수
# 이 함수는 ROS 이미지 메시지를 구독하여 OpenCV 이미지로 변환하고 큐에 넣습니다.
def image_callback(ros_image):
    # 'bgr8' 대신 'rgb8'을 시도해 보세요. '/ascamera/camera_publisher/rgb0/image'는 보통 rgb8입니다.
    # 만약 색상이 이상하게 나온다면 이 부분을 변경해야 합니다.
    try:
        cv_image = bridge.imgmsg_to_cv2(ros_image, "bgr8") 
        bgr_image = np.array(cv_image, dtype=np.uint8)
        if image_queue.full():
            image_queue.get_nowait() # 큐가 가득 찼으면 가장 오래된 항목을 제거
        image_queue.put(bgr_image)
    except Exception as e:
        # 노드 로거는 여기서는 직접 접근할 수 없으므로 print로 출력
        print(f"Error in image_callback: {e}")

# --- ROS 노드 스핀 함수 ---
def ros_spin_thread(node):
    """
    별도의 스레드에서 ROS 노드를 스핀하는 함수.
    이 함수는 ROS 이벤트 처리(콜백, 타이머 등)를 담당합니다.
    """
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass # 메인 스레드에서 종료 처리
    finally:
        executor.shutdown()
        node.destroy_node()

# --- main 함수 (ROS 노드 초기화 및 시각화 담당) ---
def main():
    # rclpy.init()은 main 함수 시작 부분에서 단 한 번만 호출되어야 합니다.
    rclpy.init() 
    
    node = rclpy.create_node('lane_detect_visualizer') # 메인 시각화 노드
    lane_detector_instance = LaneDetector('yellow') # LaneDetector 인스턴스 생성

    # ROS 이미지 토픽 구독
    # '/ascamera/camera_publisher/rgb0/image'가 실제 토픽 이름이라고 가정합니다.
    node.create_subscription(Image, '/ascamera/camera_publisher/rgb0/image', image_callback, 10) # 큐 크기 10으로 늘림
    
    # ROS 스핀을 위한 별도의 스레드 시작
    # 이렇게 하면 메인 스레드(GUI)가 ROS 콜백에 의해 블록되지 않습니다.
    spin_thread = threading.Thread(target=ros_spin_thread, args=(node,), daemon=True)
    spin_thread.start()

    # FPS 카운터 시작
    fps_counter = FPS()
    fps_counter.start()

    # OpenCV 윈도우 생성
    cv2.namedWindow('Lane Detection Visualizer (Press Q or ESC to exit)', cv2.WINDOW_NORMAL)

    node.get_logger().info("Lane detection visualizer is running.")
    node.get_logger().info(f"HSV values: H=[{h_min}-{h_max}], S=[{s_min}-{s_max}], V=[{v_min}-{v_max}]")
    node.get_logger().info("Press 'q' or ESC to exit.")

    running = True
    while rclpy.ok() and running: # rclpy가 실행 중이고, 프로그램이 종료되지 않았을 때만 루프 실행
        try:
            # 큐에서 이미지 가져오기, 짧은 타임아웃
            image_from_queue = image_queue.get(block=True, timeout=0.1)
        except queue.Empty:
            # 큐가 비어있으면 계속 시도
            # node.get_logger().debug("Image queue is empty. Waiting for images...")
            continue
        except Exception as e:
            node.get_logger().error(f"Error getting image from queue: {e}")
            running = False
            break
        
        # LaneDetector의 __call__ 메서드를 호출하여 이미지 처리 및 시각화
        # image_from_queue는 bgr8로 변환된 이미지이므로, __call__에서 BGR->HSV 변환을 수행합니다.
        processed_image, _, final_detected_cx = lane_detector_instance(image_from_queue,image_from_queue) 

        # FPS 업데이트 및 표시
        fps_counter.update()
        frame_with_fps = fps_counter.show_fps(processed_image) 

        # 터미널에 최종 차선 중심 X 좌표 출력
        if final_detected_cx != -1:
            node.get_logger().info(f"Final detected lane center X: {final_detected_cx}")
        else:
            node.get_logger().info("No lane detected.")

        try:
            # 처리된 이미지 (시각화 결과 포함) 표시
            cv2.imshow('Lane Detection Visualizer (Press Q or ESC to exit)', frame_with_fps)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27: # 'q' 또는 ESC 키 눌리면 종료
                node.get_logger().info("User pressed 'q' or ESC. Exiting.")
                running = False
                break

        except cv2.error as e:
            node.get_logger().error(f"Error in OpenCV display or key processing: {e}")
            running = False # 오류 발생 시 루프 종료
            break

    node.get_logger().info("Visualizer loop finished.")
    cv2.destroyAllWindows()
    # ROS 컨텍스트가 정리될 때까지 기다림 (선택 사항, 깔끔한 종료를 위해)
    spin_thread.join(timeout=2) 
    rclpy.shutdown()
    node.get_logger().info("ROS2 context shut down.")

if __name__ == '__main__':
    # 이 부분에서는 rclpy.init()을 호출하지 않습니다.
    # main() 함수 내부에서 단 한번만 호출됩니다.
    main()