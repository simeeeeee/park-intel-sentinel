import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
import message_filters
import cv2
import numpy as np
import json
import os
import time
import threading # threading 모듈 추가
import queue # queue 모듈 추가

# ===============================================
# 중요: GUI 표시 여부 플래그
SHOW_GUI = True # 테스트 환경에 따라 True/False 설정
# ===============================================

class ColorDepthTracker(Node):
    def __init__(self):
        super().__init__('color_depth_tracker')
        self.bridge = CvBridge()

        self.hsv_config = self.load_hsv_config()

        if SHOW_GUI:
            self.init_trackbars()
        
        self.mecanum_pub = self.create_publisher(Twist, 'cmd_vel', 10) 

        # 이미지 데이터를 저장할 큐
        self.image_queue = queue.Queue(maxsize=1) # 최신 이미지만 유지 (오래된 이미지 버림)

        # RGB + Depth 구독 (message_filters를 사용하여 이미지 동기화)
        rgb_sub = message_filters.Subscriber(self, Image, '/ascamera/camera_publisher/rgb0/image')
        depth_sub = message_filters.Subscriber(self, Image, '/ascamera/camera_publisher/depth0/image_raw')
        sync = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size=10, slop=0.05)
        sync.registerCallback(self.callback)

        self.get_logger().info("ColorDepthTracker 노드가 시작되었습니다.")

    def load_hsv_config(self, json_name="HSV-cal.json"):
        default = {
            "h_min": 20, "h_max": 40,
            "s_min": 100, "s_max": 255,
            "v_min": 100, "v_max": 255
        }
        file_path = os.path.join(os.path.dirname(__file__), json_name)

        if os.path.exists(file_path):
            try:
                with open(file_path, "r") as f:
                    config = json.load(f)
                    self.get_logger().info(f"'{json_name}' 파일이 성공적으로 로드되었습니다.")
                    return config
            except json.JSONDecodeError as e:
                self.get_logger().warn(f"'{json_name}' 파일 로드 중 JSON 디코딩 오류 발생: {e}. 기본값을 사용합니다.")
                return default
            except KeyError as e:
                self.get_logger().warn(f"'{json_name}' 파일에 필요한 키가 없습니다: {e}. 기본값을 사용합니다.")
                return default
        else:
            self.get_logger().warn(f"'{json_name}' 파일이 존재하지 않습니다. 기본값을 사용합니다.")
            return default

    def init_trackbars(self):
        def nothing(x): pass
        cv2.namedWindow("Trackbars")
        cv2.resizeWindow("Trackbars", 800, 260)
        
        cv2.createTrackbar("H Min", "Trackbars", self.hsv_config["h_min"], 179, nothing)
        cv2.createTrackbar("H Max", "Trackbars", self.hsv_config["h_max"], 179, nothing)
        cv2.createTrackbar("S Min", "Trackbars", self.hsv_config["s_min"], 255, nothing)
        cv2.createTrackbar("S Max", "Trackbars", self.hsv_config["s_max"], 255, nothing)
        cv2.createTrackbar("V Min", "Trackbars", self.hsv_config["v_min"], 255, nothing)
        cv2.createTrackbar("V Max", "Trackbars", self.hsv_config["v_max"], 255, nothing)

    def get_trackbar_values(self):
        h_min = cv2.getTrackbarPos("H Min", "Trackbars")
        h_max = cv2.getTrackbarPos("H Max", "Trackbars")
        s_min = cv2.getTrackbarPos("S Min", "Trackbars")
        s_max = cv2.getTrackbarPos("S Max", "Trackbars")
        v_min = cv2.getTrackbarPos("V Min", "Trackbars")
        v_max = cv2.getTrackbarPos("V Max", "Trackbars")
        return np.array([h_min, s_min, v_min]), np.array([h_max, s_max, v_max])

    def go_straight(self, speed):
        twist = Twist()
        twist.linear.x = speed if 0.0 < speed <= 1.0 else 0.1
        self.mecanum_pub.publish(twist)
        
    def stop(self):
        twist=Twist()
        twist.linear.x = 0.0
        twist.linear.y = 0.0
        twist.linear.z = 0.0
        twist.angular.x=0.0
        twist.angular.y=0.0
        twist.angular.z=0.0
        self.mecanum_pub.publish(twist)
        self.get_logger().info("로봇 정지 명령 발행.")
        
    def go_left(self, speed):
        twist=Twist()
        twist.linear.x = 0.0
        twist.linear.y = 0.0
        twist.linear.z = 0.0
        twist.angular.x=0.0
        twist.angular.y=0.0
        if speed > 0.0 and speed <1.0:
            twist.angular.z = speed
        else:
            twist.angular.z = 0.1
        self.mecanum_pub.publish(twist)
    
    def go_right(self, speed):
        twist=Twist()
        twist.linear.x = 0.0
        twist.linear.y = 0.0
        twist.linear.z = 0.0
        twist.angular.x=0.0
        twist.angular.y=0.0
        if speed > 0.0 and speed <1.0:
            twist.angular.z = -speed
        else:
            twist.angular.z = -0.1
        self.mecanum_pub.publish(twist)
            
    def destroy_node(self):
        self.get_logger().info("노드 종료 중... 로봇 정지 명령 발행.")
        self.stop()
        time.sleep(0.1)
        super().destroy_node()
        self.get_logger().info("노드 종료 완료.")

    def callback(self, rgb_msg, depth_msg):
        """동기화된 RGB 및 깊이 이미지 메시지를 처리하는 콜백 함수입니다."""
        try:
            rgb = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
            depth = self.bridge.imgmsg_to_cv2(depth_msg, '16UC1')
            
            # 큐에 이미지 데이터 넣기
            if not self.image_queue.full(): # 큐가 가득 차지 않았으면 추가
                self.image_queue.put((rgb, depth))
            else: # 가득 찼으면 오래된 것 버리고 새로 넣음
                try:
                    self.image_queue.get_nowait() # 가장 오래된 항목 제거
                except queue.Empty:
                    pass # 큐가 비어있으면 무시
                self.image_queue.put((rgb, depth))

        except Exception as e:
            self.get_logger().error(f"이미지 변환 오류: {e}")
            return
    
    def process_images_and_control_robot(self):
        """이미지 처리 및 로봇 제어 로직을 수행하는 함수."""
        while rclpy.ok(): # 노드가 활성화되어 있는 동안 계속 실행
            try:
                rgb, depth = self.image_queue.get(timeout=0.1) # 큐에서 이미지 가져오기, 타임아웃 설정
            except queue.Empty:
                time.sleep(0.01) # 큐가 비어있으면 잠시 대기 후 다시 시도
                continue # 다음 루프로

            h, w = rgb.shape[:2]
            roi_y_start = int(h * 1 / 2) 
            vis = rgb.copy()
            
            if SHOW_GUI:
                lower_hsv, upper_hsv = self.get_trackbar_values()
            else:
                lower_hsv = np.array([self.hsv_config["h_min"], self.hsv_config["s_min"], self.hsv_config["v_min"]])
                upper_hsv = np.array([self.hsv_config["h_max"], self.hsv_config["s_max"], self.hsv_config["v_max"]])

            mask_combined = np.zeros((h, w), dtype=np.uint8) 
            filtered_hsv_result = np.zeros_like(rgb)

            centers = []
            areas = []
            
            for i in range(3):
                x_start = i * (w // 3)
                x_end = (i + 1) * (w // 3) if i < 2 else w
                
                roi = rgb[roi_y_start:h, x_start:x_end]
                cv2.rectangle(vis, (x_start, roi_y_start), (x_end, h), (0, 255, 0), 2)

                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
                
                kernel = np.ones((3,3), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                
                mask_combined[roi_y_start:h, x_start:x_end] = mask

                result_roi = cv2.bitwise_and(roi, roi, mask=mask)
                filtered_hsv_result[roi_y_start:h, x_start:x_end] = result_roi

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                center_x = None
                max_area = 0
                
                if contours:
                    biggest = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(biggest)
                    
                    if area > 200:
                        x, y, w_box, h_box = cv2.boundingRect(biggest)
                        cx_roi = x + w_box // 2
                        center_x = x_start + cx_roi 
                        max_area = area
                        
                        cv2.rectangle(vis, (x_start + x, roi_y_start + y), 
                                      (x_start + x + w_box, roi_y_start + y + h_box), (0, 0, 255), 2)
                        cv2.circle(vis, (center_x, roi_y_start + y + h_box // 2), 5, (255, 0, 0), -1)
                
                centers.append(center_x)
                areas.append(max_area)
            
            final_cx = w // 2
            if any(c is not None for c in centers):
                valid_indices = [i for i, c in enumerate(centers) if c is not None]
                if valid_indices:
                    max_area_idx = valid_indices[np.argmax([areas[i] for i in valid_indices])]
                    final_cx = centers[max_area_idx]

            frame_cx = w // 2
            offset = final_cx - frame_cx

            depth_val = 0
            depth_cy = roi_y_start + (h - roi_y_start) // 2
            depth_cx = final_cx
            
            if 0 <= depth_cy < depth.shape[0] and 0 <= depth_cx < depth.shape[1]:
                depth_val = depth[depth_cy, depth_cx]
                cv2.putText(vis, f"{depth_val} mm", (final_cx + 10, depth_cy - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            if depth_val > 0 and depth_val < 500:
                self.stop()
                self.get_logger().info(f"객체 거리 {depth_val}mm. 정지.")
            else:
                if offset < -40:
                    self.go_left(0.08)
                    self.get_logger().info(f"좌회전 (오프셋: {offset})")
                elif offset > 40:
                    self.go_right(0.08)
                    self.get_logger().info(f"우회전 (오프셋: {offset})")
                else:
                    self.go_straight(0.08)
                    self.get_logger().info(f"직진 (오프셋: {offset})")
            
            cv2.line(vis, (frame_cx, 0), (frame_cx, h), (0, 255, 0), 1, cv2.LINE_AA) 
            
            if SHOW_GUI:
                cv2.imshow("Lane ROI Tracking", vis)
                cv2.imshow("HSV Mask", mask_combined)
                cv2.imshow("HSV Filtered Result", filtered_hsv_result)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    # GUI 스레드에서 종료 요청 시 노드 종료 플래그 설정 (rclpy.ok()를 False로 만들 방법)
                    # 실제 ROS2에서는 rclpy.shutdown()을 호출해야 하지만,
                    # 다른 스레드에서 호출 시 문제가 생길 수 있으므로 주의.
                    # 여기서는 main 함수의 spin 루프를 종료하기 위한 플래그가 필요함.
                    pass # 메인 스레드에서 종료 처리되도록 spin이 끝날때까지 기다리거나,
                         # 노드 클래스에 종료 플래그를 두어 main 함수에 전달해야 함.

def main(args=None):
    rclpy.init(args=args)
    node = ColorDepthTracker()

    # ROS2 spin을 위한 별도 스레드 시작
    # 노드 객체를 바로 rclpy.spin()에 전달하면 됨
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    # 이미지 처리 및 로봇 제어를 위한 스레드 시작 (메인 스레드에서 실행해도 됨)
    # 여기서는 새로운 스레드에서 처리하도록 구성
    process_thread = threading.Thread(target=node.process_images_and_control_robot, daemon=True)
    process_thread.start()

    try:
        # 메인 스레드는 스핀 스레드가 종료될 때까지 기다리거나,
        # 다른 주기적인 작업을 수행할 수 있습니다.
        # 여기서는 단순히 스레드들이 작업을 마치기를 기다립니다.
        while rclpy.ok():
            time.sleep(0.1) # CPU 과부하 방지
            # 만약 GUI에서 'q'를 눌렀을 때 종료하고 싶다면,
            # process_images_and_control_robot 함수에서 노드의 종료를 알리는
            # 방법을 구현해야 합니다. (예: node.destroy_node() 호출 또는 플래그 설정 후 메인 루프에서 확인)

    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt received. Shutting down node.")
    except Exception as e:
        node.get_logger().error(f"Exception occurred: {e}")
    finally:
        # 노드 종료 및 리소스 해제
        if node:
            node.destroy_node() # 노드 종료 및 로봇 정지
        rclpy.shutdown()
        if SHOW_GUI:
            cv2.destroyAllWindows()
        node.get_logger().info("Shutdown complete.")

if __name__ == '__main__':
    main()