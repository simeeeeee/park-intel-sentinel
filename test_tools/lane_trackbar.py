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
import rclpy
from sensor_msgs.msg import Image

bridge = CvBridge()

# --- HSV 값 저장 및 로드할 파일 경로 ---
COLOR_FILE_PATH = './color.txt'

# --- 초기 HSV 값 (파일 로드 실패 시 사용될 기본값) ---
# 전역 변수로 선언하여 load_hsv_from_file 및 save_hsv_to_file에서 접근
h_min, s_min, v_min = 14, 100, 100
h_max, s_max, v_max = 30, 255, 255

# 트랙바 콜백 함수 (아무것도 하지 않음)
def nothing(x):
    pass

# --- HSV 값 파일에서 로드 함수 ---
def load_hsv_from_file():
    global h_min, s_min, v_min, h_max, s_max, v_max
    try:
        with open(COLOR_FILE_PATH, 'r') as f:
            lines = f.readlines()
            if len(lines) >= 2:
                min_vals = list(map(int, lines[0].strip().split(',')))
                max_vals = list(map(int, lines[1].strip().split(',')))
                if len(min_vals) == 3 and len(max_vals) == 3:
                    h_min, s_min, v_min = min_vals
                    h_max, s_max, v_max = max_vals
                    print(f"Loaded HSV from {COLOR_FILE_PATH}: Min=[{h_min},{s_min},{v_min}], Max=[{h_max},{s_max},{v_max}]")
                else:
                    print(f"Warning: Invalid format in {COLOR_FILE_PATH}. Using default HSV values.")
            else:
                print(f"Warning: Not enough lines in {COLOR_FILE_PATH}. Using default HSV values.")
    except FileNotFoundError:
        print(f"'{COLOR_FILE_PATH}' not found. Using default HSV values.")
    except Exception as e:
        print(f"Error loading HSV from {COLOR_FILE_PATH}: {e}. Using default HSV values.")

# --- 현재 트랙바의 HSV 값을 파일에 저장 함수 ---
def save_hsv_to_file():
    current_h_min = cv2.getTrackbarPos('H_Min', 'Color Mask (Yellow)')
    current_s_min = cv2.getTrackbarPos('S_Min', 'Color Mask (Yellow)')
    current_v_min = cv2.getTrackbarPos('V_Min', 'Color Mask (Yellow)')
    current_h_max = cv2.getTrackbarPos('H_Max', 'Color Mask (Yellow)')
    current_s_max = cv2.getTrackbarPos('S_Max', 'Color Mask (Yellow)')
    current_v_max = cv2.getTrackbarPos('V_Max', 'Color Mask (Yellow)')

    try:
        with open(COLOR_FILE_PATH, 'w') as f:
            f.write(f"{current_h_min},{current_s_min},{current_v_min}\n")
            f.write(f"{current_h_max},{current_s_max},{current_v_max}\n")
        print(f"Successfully saved HSV values to: {COLOR_FILE_PATH}")
    except Exception as e:
        print(f"Error saving HSV to {COLOR_FILE_PATH}: {e}")

class LaneDetector(object):
    def __init__(self, color):
        self.target_color = color
        if os.environ.get('DEPTH_CAMERA_TYPE') == 'ascamera':
            self.rois = ((338, 360, 0, 320, 0.7), (292, 315, 0, 320, 0.2), (248, 270, 0, 320, 0.1))
        else:
            self.rois = ((450, 480, 0, 320, 0.7), (390, 480, 0, 320, 0.2), (330, 480, 0, 320, 0.1))
        self.weight_sum = 1.0

    def set_roi(self, roi):
        self.rois = roi

    @staticmethod
    def get_area_max_contour(contours, threshold=100):
        contour_area = zip(contours, tuple(map(lambda c: math.fabs(cv2.contourArea(c)), contours)))
        contour_area = tuple(filter(lambda c_a: c_a[1] > threshold, contour_area))
        if len(contour_area) > 0:
            max_c_a = max(contour_area, key=lambda c_a: c_a[1])
            return max_c_a
        return None
    
    def add_horizontal_line(self, image):
        h, w = image.shape[:2]
        roi_w_min = int(w/2)
        roi_w_max = w
        roi_h_min = 0
        roi_h_max = h
        roi = image[roi_h_min:roi_h_max, roi_w_min:roi_w_max]
        flip_binary = cv2.flip(roi, 0)
        if len(flip_binary.shape) > 2:
            flip_binary = cv2.cvtColor(flip_binary, cv2.COLOR_BGR2GRAY)
        max_y = cv2.minMaxLoc(flip_binary)[-1][1]

        return h - max_y

    def add_vertical_line_far(self, image):
        h, w = image.shape[:2]
        roi_w_min = int(w/8)
        roi_w_max = int(w/2)
        roi_h_min = 0
        roi_h_max = h
        roi = image[roi_h_min:roi_h_max, roi_w_min:roi_w_max]
        flip_binary = cv2.flip(roi, -1)
        if len(flip_binary.shape) > 2:
            flip_binary = cv2.cvtColor(flip_binary, cv2.COLOR_BGR2GRAY)
        (x_0, y_0) = cv2.minMaxLoc(flip_binary)[-1]
        y_center = y_0 + 55
        roi = flip_binary[y_center:, :]
        if len(roi.shape) > 2:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        (x_1, y_1) = cv2.minMaxLoc(roi)[-1]
        down_p = (roi_w_max - x_1, roi_h_max - (y_1 + y_center))
        
        y_center = y_0 + 65
        roi = flip_binary[y_center:, :]
        if len(roi.shape) > 2:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        (x_2, y_2) = cv2.minMaxLoc(roi)[-1]
        up_p = (roi_w_max - x_2, roi_h_max - (y_2 + y_center))

        up_point = (0, 0)
        down_point = (0, 0)
        if up_p[1] - down_p[1] != 0 and up_p[0] - down_p[0] != 0:
            up_point = (int(-down_p[1]/((up_p[1] - down_p[1])/(up_p[0] - down_p[0])) + down_p[0]), 0)
            down_point = (int((h - down_p[1])/((up_p[1] - down_p[1])/(up_p[0] - down_p[0])) + down_p[0]), h)

        return up_point, down_point

    def add_vertical_line_near(self, image):
        h, w = image.shape[:2]
        roi_w_min = 0
        roi_w_max = int(w/2)
        roi_h_min = int(h/2)
        roi_h_max = h
        roi = image[roi_h_min:roi_h_max, roi_w_min:roi_w_max]
        flip_binary = cv2.flip(roi, -1)
        if len(flip_binary.shape) > 2:
            flip_binary = cv2.cvtColor(flip_binary, cv2.COLOR_BGR2GRAY)
        (x_0, y_0) = cv2.minMaxLoc(flip_binary)[-1]
        down_p = (roi_w_max - x_0, roi_h_max - y_0)

        if len(roi.shape) > 2:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        (x_1, y_1) = cv2.minMaxLoc(roi)[-1]
        y_center = int((roi_h_max - roi_h_min - y_1 + y_0)/2)
        roi = flip_binary[y_center:, :] 
        if len(roi.shape) > 2:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        (x, y) = cv2.minMaxLoc(roi)[-1]
        up_p = (roi_w_max - x, roi_h_max - (y + y_center))

        up_point = (0, 0)
        down_point = (0, 0)
        if up_p[1] - down_p[1] != 0 and up_p[0] - down_p[0] != 0:
            up_point = (int(-down_p[1]/((up_p[1] - down_p[1])/(up_p[0] - down_p[0])) + down_p[0]), 0)
            down_point = down_p

        return up_point, down_point, y_center

    def get_binary(self, image):
        # 이 함수는 __call__ 에서 사용되므로 기존 LAB 공간 이진화 로직 유지
        img_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        img_blur = cv2.GaussianBlur(img_lab, (3, 3), 3)
        
        # NOTE: 이 부분은 main 루프에서 트랙바 값을 직접 읽어서 사용하지 않고
        # 이곳에서 target_color에 따라 고정된 lab_data를 사용합니다.
        # 만약 이 get_binary 함수에서도 트랙바 값을 사용하고 싶다면,
        # main에서 트랙바 값을 읽어와 이 함수에 인자로 전달하거나,
        # 이 함수 내에서 cv2.getTrackbarPos를 직접 호출해야 합니다.
        # 현재는 middle 함수만 트랙바 값으로 yellow 마스크를 제어합니다.
        
        # 이 함수는 현재 color.txt의 HSV 값에 영향을 받지 않고,
        # 기존 lab_config.yaml의 로직(target_color에 따른 LAB 이진화)을 따릅니다.
        # lab_data 변수가 로드되지 않은 경우를 대비하여 추가 방어 코드
        # 실제 환경에서는 sdk.common을 사용하므로 이 부분이 필요 없을 수 있습니다.
        # 여기서는 최소한의 오류 방지를 위해 임시 lab_data 구조를 가정합니다.
        
        # 안전한 lab_data 접근을 위해 수정 (global lab_data 사용)
        h_min_trackbar = cv2.getTrackbarPos('H_Min', 'Color Mask (Yellow)')
        s_min_trackbar = cv2.getTrackbarPos('S_Min', 'Color Mask (Yellow)')
        v_min_trackbar = cv2.getTrackbarPos('V_Min', 'Color Mask (Yellow)')
        h_max_trackbar = cv2.getTrackbarPos('H_Max', 'Color Mask (Yellow)')
        s_max_trackbar = cv2.getTrackbarPos('S_Max', 'Color Mask (Yellow)')
        v_max_trackbar = cv2.getTrackbarPos('V_Max', 'Color Mask (Yellow)')

        lower_bound = np.array([h_min_trackbar, s_min_trackbar, v_min_trackbar])
        upper_bound = np.array([h_max_trackbar, s_max_trackbar, v_max_trackbar])

        # BGR을 HSV로 변환 후 마스크 적용
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
        
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
                centroid_sum += center_x[i] * self.rois[i][-1]
        
        if all(x == -1 for x in center_x):
             return result_image, None, max_center_x
        
        center_pos = centroid_sum / self.weight_sum
        angle = math.degrees(-math.atan((center_pos - (w / 2.0)) / (h / 2.0)))
        
        return result_image, angle, max_center_x

    def middle(self, image):
        crop_x_start = 100
        crop_x_end = 500
        crop_y_start = 180
        crop_y_end = 480

        h_orig, w_orig = image.shape[:2]

        if crop_y_end > h_orig or crop_x_end > w_orig or \
           crop_y_start < 0 or crop_x_start < 0 or \
           crop_y_start >= crop_y_end or crop_x_start >= crop_x_end:
            print(f"Warning: Invalid or out-of-bounds ROI for middle(). Image shape: {image.shape}, ROI: [{crop_y_start}:{crop_y_end}, {crop_x_start}:{crop_x_end}]")
            crop_image = image.copy()
            crop_y_start, crop_y_end = 0, h_orig
            crop_x_start, crop_x_end = 0, w_orig
            print("Using full image as ROI due to invalid settings.")
        else:
            crop_image = image[crop_y_start:crop_y_end, crop_x_start:crop_x_end].copy()
        
        # --- 트랙바에서 Use_Color_Mask 값 읽기 ---
        # NOTE: 이 트랙바는 main 함수에서 생성되어야 합니다.
        use_color_mask = cv2.getTrackbarPos('Use_Color_Mask', 'Control Panel') # 창 이름을 'Control Panel'로 통일

        if use_color_mask == 1:
            # 트랙바에서 현재 HSV 값 읽어오기
            current_h_min = cv2.getTrackbarPos('H_Min', 'Control Panel')
            current_s_min = cv2.getTrackbarPos('S_Min', 'Control Panel')
            current_v_min = cv2.getTrackbarPos('V_Min', 'Control Panel')
            current_h_max = cv2.getTrackbarPos('H_Max', 'Control Panel')
            current_s_max = cv2.getTrackbarPos('S_Max', 'Control Panel')
            current_v_max = cv2.getTrackbarPos('V_Max', 'Control Panel')

            yellow_min_hsv = np.array([current_h_min, current_s_min, current_v_min])
            yellow_max_hsv = np.array([current_h_max, current_s_max, current_v_max])
            
            hsv_image = cv2.cvtColor(crop_image, cv2.COLOR_BGR2HSV)
            color_mask = cv2.inRange(hsv_image, yellow_min_hsv, yellow_max_hsv)
            
            kernel = np.ones((5,5), np.uint8)
            mask_for_contour = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
            mask_for_contour = cv2.morphologyEx(mask_for_contour, cv2.MORPH_CLOSE, kernel)

            try:
                cv2.imshow('Color Mask (Yellow Only)', color_mask)
            except cv2.error as e:
                print(f"Error displaying OpenCV color mask: {e}. Check if display is available.")

        else: # use_color_mask == 0:
            print("Using grayscale thresholding for lane detection.")
            gray = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            mask_for_contour = cv2.erode(thresh, None, iterations=2)   
            mask_for_contour = cv2.dilate(mask_for_contour, None, iterations=2) 
            
            try:
                # 색상 마스크가 꺼졌을 때는 해당 창을 닫거나 다른 이미지 표시
                if cv2.getWindowProperty('Color Mask (Yellow Only)', cv2.WND_PROP_VISIBLE) >= 1:
                    cv2.destroyWindow('Color Mask (Yellow Only)')
            except cv2.error:
                pass
        
        try:
            cv2.imshow('Final Mask for Contour', mask_for_contour)
        except cv2.error as e:
            print(f"Error displaying OpenCV final mask: {e}. Check if display is available.")

        contours_tuple = cv2.findContours(mask_for_contour.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = contours_tuple[0] if len(contours_tuple) == 2 else contours_tuple[1]
         
        cx = -1
        cy = -1

        if len(contours) > 0: 
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
              
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                
                h_crop, w_crop = crop_image.shape[:2]
                cv2.line(crop_image, (cx, 0), (cx, h_crop), (255, 0, 0), 1)
                cv2.line(crop_image, (0, cy), (w_crop, cy), (255, 0, 0), 1)
             
                cv2.drawContours(crop_image, [c], -1, (0, 255, 0), 1)

                print(f"Detected cx in ROI: {cx}")
        
        try:
            cv2.imshow('Processed ROI Image', crop_image)
        except cv2.error as e:
            print(f"Error displaying Processed ROI Image: {e}. Check if display is available.")
        
        cv2.rectangle(image, (crop_x_start, crop_y_start), (crop_x_end, crop_y_end), (0, 255, 255), 2)
        if cx != -1:
            cx_on_full_image = cx + crop_x_start
            cy_on_full_image = crop_y_start + (crop_y_end - crop_y_start) // 2
            cv2.circle(image, (cx_on_full_image, cy_on_full_image), 7, (0, 0, 255), -1)
            cv2.line(image, (cx_on_full_image, 0), (cx_on_full_image, h_orig), (0, 255, 255), 2)

        return cx

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


image_queue = queue.Queue(2)
def image_callback(ros_image):
    cv_image = bridge.imgmsg_to_cv2(ros_image, "bgr8")
    bgr_image = np.array(cv_image, dtype=np.uint8)
    if image_queue.full():
        image_queue.get()
    image_queue.put(bgr_image)

def main():
    global h_min, s_min, v_min, h_max, s_max, v_max

    running = True
    fps_counter = FPS()
    fps_counter.start()

    # --- HSV 값 로드 ---
    load_hsv_from_file()

    # --- OpenCV 윈도우 및 트랙바 생성 ---
    # 하나의 제어 패널 창에 모든 트랙바를 배치
    cv2.namedWindow('Control Panel')
    cv2.createTrackbar('Use_Color_Mask', 'Control Panel', 1, 1, nothing) # 0: Grayscale, 1: Color Mask
    cv2.createTrackbar('H_Min', 'Control Panel', h_min, 179, nothing)
    cv2.createTrackbar('S_Min', 'Control Panel', s_min, 255, nothing)
    cv2.createTrackbar('V_Min', 'Control Panel', v_min, 255, nothing)
    cv2.createTrackbar('H_Max', 'Control Panel', h_max, 179, nothing)
    cv2.createTrackbar('S_Max', 'Control Panel', s_max, 255, nothing)
    cv2.createTrackbar('V_Max', 'Control Panel', v_max, 255, nothing)
    
    cv2.namedWindow('Original Camera Feed with Detections')
    cv2.namedWindow('Final Mask for Contour')
    cv2.namedWindow('Processed ROI Image')
    # Use_Color_Mask가 1일 때만 보이도록 'Color Mask (Yellow Only)' 창은 필요에 따라 생성/삭제되게 함
    # cv2.namedWindow('Color Mask (Yellow Only)') # 이 부분은 middle 함수 내에서 동적으로 제어

    print("Adjust HSV trackbars in 'Control Panel' window to refine yellow detection.")
    print("Toggle 'Use_Color_Mask' (0 for Grayscale, 1 for Color Mask).")
    print("Press 's' to save current HSV values to color.txt.")
    print("Press 'q' or ESC to exit.")

    while running:
        try:
            image = image_queue.get(block=True, timeout=1)
        except queue.Empty:
            if not running:
                break
            else:
                continue
        
        # 이미지 복사본을 전달하여 원본 이미지 훼손 방지
        img_copy = image.copy() 
        detected_x = lane_detect.middle(img_copy) # 중앙선 감지 (여기서 HSV 트랙바 값 사용)

        # get_binary는 현재 middle에서 사용되지 않지만, 만약 외부에서 필요하다면
        # 트랙바 값을 반영하도록 수정해야 합니다. 여기서는 __call__에서 트랙바 값 사용.
        binary_image = lane_detect.get_binary(image.copy()) # 이진화 이미지 생성
        
        # NOTE: 이 부분은 차선 감지 로직과 중복될 수 있으므로,
        # 실제 주행 로직에서는 middle() 함수의 결과만 활용하는 것이 좋습니다.
        # 여기서는 시각화 및 디버깅을 위해 남겨둡니다.
        h, w = image.shape[:2]
        
        # cv2.imshow('binary', binary_image) # 이진화된 이미지 표시
        
        # add_horizontal_line 및 add_vertical_line_far/near는 현재 사용되지 않음
        # y = lane_detect.add_horizontal_line(binary_image)
        # roi = [(0, y), (w, y), (w, 0), (0, 0)] # roi_w_min, roi_w_max를 w로 수정
        # cv2.fillPoly(binary_image, [np.array(roi)], [0, 0, 0])
        # min_x = cv2.minMaxLoc(binary_image)[-1][0]
        # cv2.line(img_copy, (min_x, y), (w, y), (255, 255, 255), 50)
        
        result_image, angle, x = lane_detect(binary_image, image.copy()) # 여기서 다시 이진화된 이미지와 원본 이미지를 사용
        print(f"Detected lane center X in ROI: {detected_x}")

        fps_counter.update()
        frame_with_fps = fps_counter.show_fps(image) # 원본 이미지에 FPS 표시

        try:
            cv2.imshow('Original Camera Feed with Detections', frame_with_fps)
            # cv2.imshow('result', result_image) # 이전에 main에서 사용되던 결과 이미지. 필요시 다시 활성화

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("User pressed 'q' or ESC. Exiting.")
                running = False
                break
            elif key == ord('s'):
                save_hsv_to_file()
                print("Current HSV values saved to color.txt.")

        except cv2.error as e:
            print(f"Error in OpenCV display or key processing: {e}")
            running = False # 오류 발생 시 루프 종료

    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    rclpy.init()
    node = rclpy.create_node('lane_detect')
    lane_detect = LaneDetector('yellow') # 'yellow'는 __init__에서 target_color로 사용됨

    node.create_subscription(Image, '/ascamera/camera_publisher/rgb0/image', image_callback, 1)
    
    # main 함수를 별도의 스레드에서 실행
    # ROS2의 rclpy.spin(node)는 메인 스레드에서 메시지 콜백 등을 처리해야 합니다.
    # OpenCV GUI는 메인 스레드에서 실행하는 것이 일반적이지만,
    # 여기서는 기존 구조를 유지하기 위해 main 함수를 별도 스레드에서 실행하고,
    # rclpy.spin(node)를 메인 스레드에서 호출합니다.
    # 만약 GUI가 응답하지 않는 문제가 발생하면, GUI와 ROS2 spin을 분리하는 방식을 고려해야 합니다.
    threading.Thread(target=main, daemon=True).start()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("ROS2 node interrupted. Shutting down...")
    finally:
        node.destroy_node()
        print("ROS2 node destroyed.")