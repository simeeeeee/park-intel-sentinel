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
import sdk.common as common
from cv_bridge import CvBridge

bridge = CvBridge()

lab_data = common.get_yaml_data("/home/ubuntu/software/lab_tool/lab_config.yaml")

class LaneDetector(object):
    def __init__(self, color):
        # lane color
        self.target_color = color
        # ROI for lane detection
        if os.environ['DEPTH_CAMERA_TYPE'] == 'ascamera':
            self.rois = ((338, 360, 0, 320, 0.7), (292, 315, 0, 320, 0.2), (248, 270, 0, 320, 0.1))
        else:
            self.rois = ((450, 480, 0, 320, 0.7), (390, 480, 0, 320, 0.2), (330, 480, 0, 320, 0.1))
        self.weight_sum = 1.0

    def set_roi(self, roi):
        self.rois = roi

    @staticmethod
    def get_area_max_contour(contours, threshold=100):
        '''
        obtain the contour corresponding to the maximum area
        :param contours:
        :param threshold:
        :return:
        '''
        contour_area = zip(contours, tuple(map(lambda c: math.fabs(cv2.contourArea(c)), contours)))
        contour_area = tuple(filter(lambda c_a: c_a[1] > threshold, contour_area))
        if len(contour_area) > 0:
            max_c_a = max(contour_area, key=lambda c_a: c_a[1])
            return max_c_a
        return None
    
    def add_horizontal_line(self, image):
        #   |____  --->   |————   ---> ——
        h, w = image.shape[:2]
        roi_w_min = int(w/2)
        roi_w_max = w
        roi_h_min = 0
        roi_h_max = h
        roi = image[roi_h_min:roi_h_max, roi_w_min:roi_w_max]  # crop the right half
        flip_binary = cv2.flip(roi, 0)  # flip upside down
        max_y = cv2.minMaxLoc(flip_binary)[-1][1]  # extract the coordinates of the top-left point with a value of 255

        return h - max_y

    def add_vertical_line_far(self, image):
        h, w = image.shape[:2]
        roi_w_min = int(w/8)
        roi_w_max = int(w/2)
        roi_h_min = 0
        roi_h_max = h
        roi = image[roi_h_min:roi_h_max, roi_w_min:roi_w_max]
        flip_binary = cv2.flip(roi, -1)  # flip the image horizontally and vertically
        #cv2.imshow('1', flip_binary)
        # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(ret)
        # minVal：the minimum value
        # maxVal：the maximum value
        # minLoc：the location of the minimum value
        # maxLoc：the location of the maximum value
        # the order of traversal is: first rows, then columns, with rows from left to right and columns from top to bottom
        (x_0, y_0) = cv2.minMaxLoc(flip_binary)[-1]  # extract the coordinates of the top-left point with a value of 255
        y_center = y_0 + 55
        roi = flip_binary[y_center:, :]
        (x_1, y_1) = cv2.minMaxLoc(roi)[-1]
        down_p = (roi_w_max - x_1, roi_h_max - (y_1 + y_center))
        
        y_center = y_0 + 65
        roi = flip_binary[y_center:, :]
        (x_2, y_2) = cv2.minMaxLoc(roi)[-1]
        up_p = (roi_w_max - x_2, roi_h_max - (y_2 + y_center))

        up_point = (0, 0)
        down_point = (0, 0)
        if up_p[1] - down_p[1] != 0 and up_p[0] - down_p[0] != 0:
            up_point = (int(-down_p[1]/((up_p[1] - down_p[1])/(up_p[0] - down_p[0])) + down_p[0]), 0)
            down_point = (int((h - down_p[1])/((up_p[1] - down_p[1])/(up_p[0] - down_p[0])) + down_p[0]), h)

        return up_point, down_point

    def add_vertical_line_near(self, image):
        # ——|         |——        |
        #   |   --->  |     --->
        h, w = image.shape[:2]
        roi_w_min = 0
        roi_w_max = int(w/2)
        roi_h_min = int(h/2)
        roi_h_max = h
        roi = image[roi_h_min:roi_h_max, roi_w_min:roi_w_max]
        flip_binary = cv2.flip(roi, -1)  # flip the image horizontally and vertically
        cv2.imshow('1', flip_binary)
        (x_0, y_0) = cv2.minMaxLoc(flip_binary)[-1]  # extract the coordinates of the top-left point with a value of 255
        down_p = (roi_w_max - x_0, roi_h_max - y_0)

        (x_1, y_1) = cv2.minMaxLoc(roi)[-1]
        y_center = int((roi_h_max - roi_h_min - y_1 + y_0)/2)
        roi = flip_binary[y_center:, :] 
        (x, y) = cv2.minMaxLoc(roi)[-1]
        up_p = (roi_w_max - x, roi_h_max - (y + y_center))

        up_point = (0, 0)
        down_point = (0, 0)
        if up_p[1] - down_p[1] != 0 and up_p[0] - down_p[0] != 0:
            up_point = (int(-down_p[1]/((up_p[1] - down_p[1])/(up_p[0] - down_p[0])) + down_p[0]), 0)
            down_point = down_p

        return up_point, down_point, y_center

    def get_binary(self, image):
        # recognize color through LAB space
        img_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)  # convert RGB to LAB
        img_blur = cv2.GaussianBlur(img_lab, (3, 3), 3)  # Gaussian blur denoising
        mask = cv2.inRange(img_blur, tuple(lab_data['lab']['Stereo'][self.target_color]['min']), tuple(lab_data['lab']['Stereo'][self.target_color]['max']))  # 二值化
        eroded = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))  # erode
        dilated = cv2.dilate(eroded, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))  # dilate

        return dilated

    def __call__(self, image, result_image):
        # extract the center point based on the proportion
        centroid_sum = 0
        h, w = image.shape[:2]
        max_center_x = -1
        center_x = []
        for roi in self.rois:
            blob = image[roi[0]:roi[1], roi[2]:roi[3]]  # crop ROI
            contours = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)[-2]  # find contours
            max_contour_area = self.get_area_max_contour(contours, 30)  # obtain the contour with the largest area
            if max_contour_area is not None:
                rect = cv2.minAreaRect(max_contour_area[0])  # the minimum bounding rectangle
                box = np.intp(cv2.boxPoints(rect))  # four corners
                for j in range(4):
                    box[j, 1] = box[j, 1] + roi[0]
                cv2.drawContours(result_image, [box], -1, (255, 255, 0), 2)  # draw the rectangle composed of the four points

                # obtain the diagonal points of the rectangle
                pt1_x, pt1_y = box[0, 0], box[0, 1]
                pt3_x, pt3_y = box[2, 0], box[2, 1]
                # the center point of the line
                line_center_x, line_center_y = (pt1_x + pt3_x) / 2, (pt1_y + pt3_y) / 2

                cv2.circle(result_image, (int(line_center_x), int(line_center_y)), 5, (0, 0, 255), -1)  # draw the center point
                center_x.append(line_center_x)
            else:
                center_x.append(-1)
        for i in range(len(center_x)):
            if center_x[i] != -1:
                if center_x[i] > max_center_x:
                    max_center_x = center_x[i]
                centroid_sum += center_x[i] * self.rois[i][-1]
        if centroid_sum == 0:
            return result_image, None, max_center_x
        center_pos = centroid_sum / self.weight_sum  # calculate the center point based on the proportion
        angle = math.degrees(-math.atan((center_pos - (w / 2.0)) / (h / 2.0)))
        
        return result_image, angle, max_center_x
    #------------------------수정--------------------------

    def middle(self, image):
        """
        주어진 이미지의 특정 ROI에서 중앙 선을 감지하고, 그 선의 X축 무게 중심을 반환합니다.
        시각화 결과를 화면에 표시합니다.
        """
        # 중앙선 감지를 위한 이미지 ROI (Region of Interest) 정의
        # [y_start:y_end, x_start:x_end]
        # 사용자 요청에 따라 640x480 해상도를 기준으로 ROI 조정
        # 가로: 200~400 (중앙 200픽셀), 세로: 밑부분부터 200픽셀 (480-200=280부터 480까지)
        crop_x_start = 100
        crop_x_end = 500
        crop_y_start = 180 # 480 - 200 = 280
        crop_y_end = 480


        h_orig, w_orig = image.shape[:2] # 원본 이미지의 높이와 너비

        # ROI가 이미지 경계를 벗어나는지 확인 (안전성 확보)
        if crop_y_end > h_orig or crop_x_end > w_orig or \
           crop_y_start < 0 or crop_x_start < 0 or \
           crop_y_start >= crop_y_end or crop_x_start >= crop_x_end:
            print(f"Warning: Invalid or out-of-bounds ROI for middle(). Image shape: {image.shape}, ROI: [{crop_y_start}:{crop_y_end}, {crop_x_start}:{crop_x_end}]")
            # ROI가 유효하지 않으면 기본 ROI (전체 이미지)를 사용하거나, 에러 처리
            # 여기서는 ROI 대신 전체 이미지를 사용하도록 대체
            crop_image = image.copy()
            crop_y_start, crop_y_end = 0, h_orig
            crop_x_start, crop_x_end = 0, w_orig
            print("Using full image as ROI due to invalid settings.")
        else:
            # 이미지에서 정의된 ROI를 자름. .copy()를 사용하여 원본 이미지에 영향을 주지 않도록 함.
            crop_image = image[crop_y_start:crop_y_end, crop_x_start:crop_x_end].copy()
        
        # --- 노란색만 검출하는 로직 추가 (선택 사항) ---
        # lab_config.yaml에서 로드한 노란색 HSV 범위 사용
        if lab_data and 'lab' in lab_data and 'Stereo' in lab_data['lab'] and 'yellow' in lab_data['lab']['Stereo']:
            yellow_min_hsv = np.array([14, 100, 100])
            yellow_max_hsv = np.array([30, 255, 255])
            
            # BGR을 HSV로 변환
            hsv_image = cv2.cvtColor(crop_image, cv2.COLOR_BGR2HSV)
            
            # HSV 범위에 따른 마스크 생성
            color_mask = cv2.inRange(hsv_image, yellow_min_hsv, yellow_max_hsv)
            
            # --- 중요: 어떤 마스크를 사용할지 결정 ---
            # 1. 색상 기반 마스크를 사용할 경우:
            mask = color_mask
            # 2. 기존의 명암 대비 기반 마스크를 사용할 경우 (아래 코드 주석 처리):
            #    cv2.threshold(blur, ...)와 cv2.erode, cv2.dilate 사용
            
            # (선택 사항) 색상 마스크만으로 부족할 경우, 기존 명암 대비 마스크와 결합
            # gray = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
            # blur = cv2.GaussianBlur(gray, (5, 5), 0)
            # ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # mask_contrast = cv2.erode(thresh, None, iterations=2)   
            # mask_contrast = cv2.dilate(mask_contrast, None, iterations=2) 
            # mask = cv2.bitwise_and(color_mask, mask_contrast) # 두 마스크를 결합

            cv2.imshow('Color Mask (Yellow)', color_mask) # 노란색 마스크 표시
            
            # 현재는 색상 마스크를 기본 마스크로 사용
            mask_for_contour = mask # 윤곽선 검출에 사용할 마스크
        else:
            # lab_config 로드 실패 또는 yellow 설정 없을 경우 기존 명암 대비 방식 사용
            print("Using grayscale thresholding for lane detection (yellow color range not found/loaded).")
            gray = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY) # 이미지를 회색으로 변경
            blur = cv2.GaussianBlur(gray, (5, 5), 0) # 가우시안 필터링 블러 처리 (노이즈 감소)
            ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            mask_for_contour = cv2.erode(thresh, None, iterations=2)   
            mask_for_contour = cv2.dilate(mask_for_contour, None, iterations=2) 
        
        cv2.imshow('Final Mask for Contour', mask_for_contour) # 최종 마스크 이미지 표시

        # 이미지의 윤곽선을 검출
        contours_tuple = cv2.findContours(mask_for_contour.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # OpenCV 버전 호환성을 위해 튜플 언패킹 수정
        contours = contours_tuple[0] if len(contours_tuple) == 2 else contours_tuple[1]
         
        cx = -1 # X축 무게 중심 초기값 (선 감지 실패 시 -1 반환)
        cy = -1 # Y축 무게 중심 초기값

        # 윤곽선이 있다면, 가장 큰(면적이 넓은) 윤곽선을 선택하여 모멘트를 계산한다.
        if len(contours) > 0: 
            c = max(contours, key=cv2.contourArea) # 가장 큰 윤곽선 선택
            M = cv2.moments(c) # 선택된 윤곽선의 모멘트 계산
              
            # 모멘트의 'm00' (면적)이 0이 아닌 경우에만 무게 중심 계산 (0으로 나누는 오류 방지)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00']) # X축 무게 중심
                cy = int(M['m01'] / M['m00']) # Y축 무게 중심
                
                # 시각화를 위해 crop_image에 무게 중심 선 그리기
                h_crop, w_crop = crop_image.shape[:2]
                cv2.line(crop_image, (cx, 0), (cx, h_crop), (255, 0, 0), 1) # 파란색 수직선 (X축 무게 중심)
                cv2.line(crop_image, (0, cy), (w_crop, cy), (255, 0, 0), 1) # 파란색 수평선 (Y축 무게 중심)
             
                # 윤곽선을 crop_image에 그리기
                cv2.drawContours(crop_image, [c], -1, (0, 255, 0), 1) # 초록색 윤곽선

                # X축 무게 중심을 콘솔에 출력
                print(f"Detected cx in ROI: {cx}") 
        
        cv2.imshow('Processed ROI Image', crop_image) # 처리된 잘라낸 이미지 표시
        
        # 원본 이미지에도 ROI 표시 및 감지된 중앙선 표시
        # ROI 사각형 그리기 (원본 이미지 기준)
        cv2.rectangle(image, (crop_x_start, crop_y_start), (crop_x_end, crop_y_end), (0, 255, 255), 2) # 노란색 사각형
        if cx != -1:
            cx_on_full_image = cx + crop_x_start # 전체 이미지에서의 중앙점 X 좌표
            # 중앙점의 Y좌표는 ROI의 중간 높이를 사용
            cy_on_full_image = crop_y_start + (crop_y_end - crop_y_start) // 2 
            cv2.circle(image, (cx_on_full_image, cy_on_full_image), 7, (0, 0, 255), -1) # 원본 이미지에 빨간색 중앙점
            cv2.line(image, (cx_on_full_image, 0), (cx_on_full_image, h_orig), (0, 255, 255), 2) # 원본 이미지에 노란색 수직선

        return cx # 감지된 X축 무게 중심 반환
#------------------------수정--------------------------
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



#-----------------------------------------------------

image_queue = queue.Queue(2)
def image_callback(ros_image):
    cv_image = bridge.imgmsg_to_cv2(ros_image, "bgr8")
    bgr_image = np.array(cv_image, dtype=np.uint8)
    if image_queue.full():
        # if the queue is full, remove the oldest image
        image_queue.get()
        # put the image into the queue
    image_queue.put(bgr_image)

def main():
    running = True
    # self.get_logger().info('\033[1;32m%s\033[0m' % (*tuple(lab_data['lab']['Stereo'][self.target_color]['min']), tuple(lab_data['lab']['Stereo'][self.target_color]['max'])))
    fps_counter = FPS() # FPS 카운터 인스턴스
    fps_counter.start()
    while running:
        try:
            image = image_queue.get(block=True, timeout=1)
        except queue.Empty:
            if not running:
                break
            else:
                continue
        detected_x = lane_detect.middle(image.copy()) # 중앙선 감지
        binary_image = lane_detect.get_binary(image)
        x,h = image.shape[:2]

        cv2.imshow('binary', binary_image)
        img = image.copy()
        y = lane_detect.add_horizontal_line(binary_image)
        roi = [(0, y), (640, y), (640, 0), (0, 0)]
        cv2.fillPoly(binary_image, [np.array(roi)], [0, 0, 0])  # fill the top with black to avoid interference
        min_x = cv2.minMaxLoc(binary_image)[-1][0]
        cv2.line(img, (min_x, y), (640, y), (255, 255, 255), 50)  # draw a virtual line to guide the turning
        result_image, angle, x = lane_detect(binary_image, image.copy()) 
        print(f"가로 {x} 세로 {h} 이미지에서 중앙선 X축 무게 중심: {detected_x}")
        '''
        up, down = lane_detect.add_vertical_line_far(binary_image)
        #up, down, center = lane_detect.add_vertical_line_near(binary_image)
        cv2.line(img, up, down, (255, 255, 255), 10)
        '''
        #cv2.imshow('image', img)
        #cv2.imshow('result', result_image)
        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:  # press Q or Esc to quit
            break

    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    import rclpy
    from sensor_msgs.msg import Image
    rclpy.init()
    node = rclpy.create_node('lane_detect')
    lane_detect = LaneDetector('yellow')
    node.create_subscription(Image, '/ascamera/camera_publisher/rgb0/image', image_callback, 1)
    threading.Thread(target=main, daemon=True).start()
    rclpy.spin(node)




