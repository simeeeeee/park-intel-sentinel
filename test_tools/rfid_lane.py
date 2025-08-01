import os
import cv2
import math
import queue
import threading
import numpy as np
import sdk.common as common
from cv_bridge import CvBridge
import rclpy
from sensor_msgs.msg import Image # main 함수에 필요하므로 명시적으로 임포트

bridge = CvBridge()

# lab_config.yaml 로드 부분 (기존 코드와 동일)
try:
    lab_data = common.get_yaml_data("/home/ubuntu/software/lab_tool/lab_config.yaml")
except Exception as e:
    print(f"Error loading lab_config.yaml: {e}. Please ensure the file exists and is accessible.")
    # 오류 발생 시 기본값 설정 (예시)
    lab_data = {'lab': {'Stereo': {'yellow': {'min': [20, 100, 100], 'max': [255, 140, 140]}}}}

class LaneDetector(object):
    def __init__(self, color):
        # lane color
        self.target_color = color
        # ROI for lane detection
        if os.environ.get('DEPTH_CAMERA_TYPE') == 'ascamera': # .get() 사용으로 안전하게 환경 변수 접근
            self.rois = ((338, 360, 0, 320, 0.7), (292, 315, 0, 320, 0.2), (248, 270, 0, 320, 0.1))
        else:
            self.rois = ((450, 480, 0, 320, 0.7), (390, 480, 0, 320, 0.2), (330, 480, 0, 320, 0.1))
        self.weight_sum = 1.0

    def set_roi(self, roi):
        self.rois = roi

    # 기존 __call__ 메서드 및 기타 차선 감지 관련 메서드 생략 (원본 코드를 참고해주세요)
    # 다만, 아래는 분석을 위해 최소한으로 포함했습니다.

    @staticmethod
    def get_area_max_contour(contours, threshold=100):
        contour_area = zip(contours, tuple(map(lambda c: math.fabs(cv2.contourArea(c)), contours)))
        contour_area = tuple(filter(lambda c_a: c_a[1] > threshold, contour_area))
        if len(contour_area) > 0:
            max_c_a = max(contour_area, key=lambda c_a: c_a[1])
            return max_c_a
        return None
    
    def get_binary(self, image):
        img_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        img_blur = cv2.GaussianBlur(img_lab, (3, 3), 3)
        lab_min = lab_data['lab']['Stereo'].get(self.target_color, {}).get('min', [0, 0, 0])
        lab_max = lab_data['lab']['Stereo'].get(self.target_color, {}).get('max', [255, 255, 255])
        mask = cv2.inRange(img_blur, tuple(lab_min), tuple(lab_max))
        eroded = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        dilated = cv2.dilate(eroded, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        return dilated

    def __call__(self, image, result_image):
        centroid_sum = 0
        h, w = image.shape[:2]
        max_center_x = -1
        center_x = []
        for roi in self.rois:
            blob = image[roi[0]:roi[1], roi[2]:roi[3]]
            if blob.shape[0] == 0 or blob.shape[1] == 0:
                center_x.append(-1)
                continue
            contours_tuple = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
            contours = contours_tuple[0] if len(contours_tuple) == 2 else contours_tuple[1]
            
            max_contour_area = self.get_area_max_contour(contours, 30)
            if max_contour_area is not None:
                rect = cv2.minAreaRect(max_contour_area[0])
                box = np.intp(cv2.boxPoints(rect))
                for j in range(4):
                    box[j, 1] = box[j, 1] + roi[0]
                cv2.drawContours(result_image, [box], -1, (255, 255, 0), 2)

                pt1_x, pt1_y = box[0, 0], box[0, 1]
                pt3_x, pt3_y = box[2, 0], box[2, 1]
                line_center_x, line_center_y = (pt1_x + pt3_x) / 2, (pt1_y + pt3_y) / 2

                cv2.circle(result_image, (int(line_center_x), int(line_center_y)), 5, (0, 0, 255), -1)
                center_x.append(line_center_x)
            else:
                center_x.append(-1)
        
        for i in range(len(center_x)):
            if center_x[i] != -1:
                if center_x[i] > max_center_x:
                    max_center_x = center_x[i]
                if i < len(self.rois) and len(self.rois[i]) > 4:
                    centroid_sum += center_x[i] * self.rois[i][-1]
                else:
                    centroid_sum += center_x[i]

        if centroid_sum == 0:
            return result_image, None, max_center_x
        
        if self.weight_sum == 0: self.weight_sum = 1.0 # 0으로 나누는 것 방지
        
        center_pos = centroid_sum / self.weight_sum
        
        denominator = (h / 2.0)
        angle = math.degrees(-math.atan((center_pos - (w / 2.0)) / denominator)) if denominator != 0 else 0.0
        
        return result_image, angle, max_center_x


    def middle(self, image):
        # ROI 크기 정의 (원본 코드와 동일)
        crop_y_start, crop_y_end = 60, 120
        crop_x_start, crop_x_end = 0, 160

        # 이미지 크기 검증
        h_orig, w_orig = image.shape[:2]
        if crop_y_end > h_orig or crop_x_end > w_orig:
            print("Warning: ROI for middle() is out of image bounds. Skipping.")
            return

        # 이미지 자르기
        crop_image = image[crop_y_start:crop_y_end, crop_x_start:crop_x_end].copy() # .copy()를 사용하여 원본 이미지에 영향 방지

        # 잘라낸 이미지의 높이와 너비
        h_crop, w_crop = crop_image.shape[:2]

        gray = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY) # 이미지 회색으로 변경
        blur = cv2.GaussianBlur(gray, (5, 5), 0) # 가우시안 필터링 블러처리
        
        # OTSU를 사용할 경우 임계값은 0으로 설정하는 것이 일반적입니다.
        ret, thresh1 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # 임계점 처리
        
        mask = cv2.erode(thresh1, None, iterations=2)  
        mask = cv2.dilate(mask, None, iterations=2)
        cv2.imshow('mask', mask) # 처리된 마스크 이미지 표시
    
        # 이미지의 윤곽선을 검출
        contours_tuple = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # OpenCV 버전 호환성을 위해 윤곽선 추출 결과 확인
        contours = contours_tuple[0] if len(contours_tuple) == 2 else contours_tuple[1]
        
        # 윤곽선이 있다면, max(가장큰값)을 반환, 모멘트를 계산한다.
        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
            
            # 0으로 나누는 에러 방지
            if M['m00'] != 0:
                # X축과 Y축의 무게중심을 구한다.
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                
                # X축의 무게중심을 crop_image에 출력한다.
                # 선의 길이가 crop_image의 높이, 너비에 맞게 조정
                cv2.line(crop_image, (cx, 0), (cx, h_crop), (255, 0, 0), 1)
                cv2.line(crop_image, (0, cy), (w_crop, cy), (255, 0, 0), 1)
            
                cv2.drawContours(crop_image, contours, -1, (0,255,0), 1) # 모든 윤곽선 그리기

                print(f"Detected center X: {cx}") # 출력값을 print 한다.
        
        cv2.imshow('cropped_with_line', crop_image) # 처리된 잘린 이미지 표시


# image_queue는 ROS 메시지를 비동기적으로 처리하기 위해 필요합니다.
image_queue = queue.Queue(2)

def image_callback(ros_image):
    global bgr_image_for_main # main 함수에서 접근할 이미지 변수 (필요시 사용)
    try:
        cv_image = bridge.imgmsg_to_cv2(ros_image, "bgr8")
        bgr_image = np.array(cv_image, dtype=np.uint8)
        if image_queue.full():
            image_queue.get() # 큐가 가득 차면 가장 오래된 이미지 제거
        image_queue.put(bgr_image)
    except Exception as e:
        print(f"Error in image_callback: {e}")

def main(args=None):
    # 웹캠 비디오 캡처 초기화 (ROS 카메라 토픽 대신)
    cap = cv2.VideoCapture(0) 
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    rclpy.init(args=args)
    # ROS 노드 초기화
    node = rclpy.create_node('lane_detector_node') # 노드 이름 변경
    
    # LaneDetector 인스턴스 초기화 (노란색 차선 감지)
    lane_detector = LaneDetector(color='yellow')

    # ROS 토픽 구독 (웹캠 사용 시 주석 처리하거나, 웹캠과 ROS 둘 다 사용 시 유지)
    # node.create_subscription(Image, '/ascamera/camera_publisher/rgb0/image', image_callback, 1)

    running = True
    while running:
        # 웹캠에서 프레임 읽기 (ROS 토픽 대신)
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from camera. Exiting.")
            running = False
            continue
        
        # 'middle' 메서드 호출
        lane_detector.middle(frame)

        # q 또는 Esc 키를 누르면 종료
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            running = False

    # 리소스 해제
    cap.release()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()