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
import threading
import queue
from MFRC522 import MFRC522
import signal
import subprocess
import socket
import traceback

# ===============================================
# 중요: GUI 표시 여부 플래그
SHOW_GUI = True # 테스트 환경에 따라 True/False 설정
# ===============================================

def uidToString(uid):
    mystring = ""
    for i in uid:
        mystring = format(i, '02X') + mystring
    return mystring

# RFID 리더 클래스는 원본과 동일하게 유지됩니다.
class myRFIDReader(MFRC522):
    def __init__(self, bus=0, dev=0):
        super().__init__(bus=bus, dev=dev)
        self.key = None
        self.keyIn = False
        self.keyValidCount = 0

    def Read(self):
        status, TagType = self.MFRC522_Request(super().PICC_REQIDL)
        if status == self.MI_OK:
            status, uid = self.MFRC522_SelectTagSN()
            if status == self.MI_OK:
                self.keyIn = True
                self.keyValidCount = 2
                if self.key != uid:
                    self.key = uid
                    if uid is None:
                        return False
                    return True
        else:
            if self.keyIn:
                if self.keyValidCount > 0:
                    self.keyValidCount -= 1
                else:
                    self.keyIn = False
                    self.key = None
        return False

class ColorDepthTracker(Node):
    def __init__(self):
        super().__init__('color_depth_tracker')
        self.bridge = CvBridge()
        self.paused = False

        self.hsv_config = self.load_hsv_config()

        if SHOW_GUI:
            self.init_trackbars()
        
        self.mecanum_pub = self.create_publisher(Twist, 'cmd_vel', 10) 
        self.image_queue = queue.Queue(maxsize=2)

        rgb_sub = message_filters.Subscriber(self, Image, '/ascamera/camera_publisher/rgb0/image')
        depth_sub = message_filters.Subscriber(self, Image, '/ascamera/camera_publisher/depth0/image_raw')
        sync = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size=10, slop=0.05)
        sync.registerCallback(self.callback)

        self.get_logger().info("ColorDepthTracker 노드가 시작되었습니다. (일시정지: q, 재개: w, 종료: ESC)")
        self.start_rfid_process()

        self.rfid_stop_flag = False
        self.rfid_stop_uid = None
        self.HOST = '127.0.0.1'
        self.PORT = 10004
        self.client_socket = None
        self.socket_lock = threading.Lock()
        self.connect_to_server()

    def connect_to_server(self):
        with self.socket_lock:
            if self.client_socket:
                self.client_socket.close()
                self.client_socket = None
            try:
                self.get_logger().info(f"서버({self.HOST}:{self.PORT})에 연결을 시도합니다...")
                new_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                new_socket.settimeout(3)
                new_socket.connect((self.HOST, self.PORT))
                new_socket.settimeout(None)
                self.client_socket = new_socket
                self.get_logger().info("✅ 서버 연결에 성공했습니다.")
                return True
            except (socket.timeout, ConnectionRefusedError, OSError) as e:
                self.get_logger().error(f"🚨 서버 연결 실패: {e}")
                self.client_socket = None
                return False

    def send_message(self, message, max_retries=1):
        if not message:
            return False
        for attempt in range(max_retries + 1):
            with self.socket_lock:
                if self.client_socket:
                    try:
                        self.client_socket.sendall(message.encode('utf-8'))
                        self.get_logger().info(f"메시지 전송 성공: {message}")
                        return True
                    except (BrokenPipeError, ConnectionResetError) as e:
                        self.get_logger().warn(f"소켓 연결이 끊어졌습니다: {e}. (시도 {attempt + 1})")
                        self.client_socket = None
                    except Exception as e:
                        self.get_logger().error(f"메시지 전송 중 예기치 않은 오류: {e}")
                        self.client_socket = None
            if attempt < max_retries:
                self.get_logger().info("재연결을 시도합니다...")
                if self.connect_to_server():
                    time.sleep(0.1)
                    continue
                else:
                    self.get_logger().warn("재연결 실패. 2초 후 다시 시도합니다.")
                    time.sleep(2)
        self.get_logger().error(f"최종적으로 메시지 전송에 실패했습니다: {message}")
        return False

    def load_hsv_config(self, json_name="HSV-cal.json"):
        default = {"h_min": 9, "h_max": 51, "s_min": 43, "s_max": 255, "v_min": 54, "v_max": 255}
        file_path = os.path.join(os.path.dirname(__file__), json_name)
        if os.path.exists(file_path):
            try:
                with open(file_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                self.get_logger().warn(f"'{json_name}' 파일 로드 오류: {e}. 기본값을 사용합니다.")
        else:
            self.get_logger().warn(f"'{json_name}' 파일이 존재하지 않습니다. 기본값을 사용합니다.")
        return default

    def init_trackbars(self):
        def nothing(x): pass
        cv2.namedWindow("Trackbars")
        cv2.resizeWindow("Trackbars", 800, 260)
        for key, value in self.hsv_config.items():
            max_val = 179 if 'h' in key else 255
            cv2.createTrackbar(key, "Trackbars", value, max_val, nothing)

    def get_trackbar_values(self):
        h_min = cv2.getTrackbarPos("h_min", "Trackbars")
        h_max = cv2.getTrackbarPos("h_max", "Trackbars")
        s_min = cv2.getTrackbarPos("s_min", "Trackbars")
        s_max = cv2.getTrackbarPos("s_max", "Trackbars")
        v_min = cv2.getTrackbarPos("v_min", "Trackbars")
        v_max = cv2.getTrackbarPos("v_max", "Trackbars")
        return np.array([h_min, s_min, v_min]), np.array([h_max, s_max, v_max])

    def stop(self):
        twist=Twist()
        twist.linear.x, twist.linear.y, twist.linear.z = 0.0, 0.0, 0.0
        twist.angular.x, twist.angular.y, twist.angular.z = 0.0, 0.0, 0.0
        self.mecanum_pub.publish(twist)

# ===================================================================
    # START: 차선 추종 주행 로직 (장애물 감지 제외)
    # ===================================================================
    def drive_logic(self, rgb, depth):
        h, w = rgb.shape[:2]
        vis = rgb.copy() # 시각화용 이미지 복사

        # 1. ROI(관심 영역) 설정 및 HSV 변환
        roi_y = h // 2
        roi_hsv = cv2.cvtColor(rgb[roi_y:h, :], cv2.COLOR_BGR2HSV)

        # 2. HSV 값으로 마스크 생성
        if SHOW_GUI:
            lower, upper = self.get_trackbar_values()
        else:
            lower = np.array([self.hsv_config["h_min"], self.hsv_config["s_min"], self.hsv_config["v_min"]])
            upper = np.array([self.hsv_config["h_max"], self.hsv_config["s_max"], self.hsv_config["v_max"]])
        
        mask = cv2.inRange(roi_hsv, lower, upper)
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # 3. 마스크의 무게 중심 계산
        M = cv2.moments(mask)
        
        # Twist 메시지 초기화
        twist = Twist()

        if M["m00"] > 200: # 노이즈를 걸러내기 위해 최소 면적 임계값 설정
            # 4. 차선이 감지되었을 경우: 차선 추종 주행
            cx = int(M["m10"] / M["m00"])
            
            # 화면 중앙과의 오차(offset) 계산
            offset = cx - (w // 2)
            
            # 주행 로직 결정
            twist.linear.x = 0.1  # 직진 속도 고정 0.08
            
            # 오차에 비례하여 회전 속도 조절 (비례 제어)
            max_angular = 0.5      # 최대 회전 속도 0.3
            max_offset = w // 2    # 최대 오차 (화면 너비의 절반)
            
            angular_z = -max_angular * (offset / max_offset)
            
            # 미세한 오차는 무시하여 직진 안정성 확보
            if abs(angular_z) < 0.02:
                angular_z = 0.0
            
            # 최대 회전 속도를 넘지 않도록 제한
            twist.angular.z = np.clip(angular_z, -max_angular, max_angular) 

            # [시각화용] 감지된 차선 중심점 업데이트
            depth_cx_display = cx

        else:
            # ===================================================================
            # START: 추가된 로직 - 차선을 놓쳤을 때의 복구 동작
            # ===================================================================
            # 차선이 감지되지 않으면 좌회전하여 차선을 다시 찾습니다.
            self.get_logger().warn('Line not detected, executing recovery behavior (turning left).') # 터미널에 경고 메시지 출력
            
            twist.linear.x = 0.1  # 탐색 시 안정성을 위해 직진 속도를 줄입니다. 0.05
            twist.angular.z = 0.5  # 좌회전을 위한 회전 속도 설정 (값은 환경에 맞게 조절) 0.4
            # ===================================================================
            # END: 추가된 로직
            # ===================================================================

            # [시각화용] 차선을 놓쳤을 때 중심점을 화면 중앙으로 표시
            depth_cx_display = w // 2

        # 최종적으로 계산된 주행 명령을 발행
        self.mecanum_pub.publish(twist)

        # 6. GUI 시각화 (깊이 값은 표시만 하고 제어에는 사용 안 함)
        if SHOW_GUI:
            # 깊이 값 측정 (디버깅/표시용)
            depth_cy = roi_y + (h - roi_y) // 2
            depth_val = 0
            if 0 <= depth_cx_display < w and 0 <= depth_cy < h:
                depth_val = depth[depth_cy, depth_cx_display]
            
            cv2.line(vis, (w // 2, 0), (w // 2, h), (0, 255, 0), 1) # 화면 중앙선
            cv2.circle(vis, (depth_cx_display, depth_cy), 5, (0, 0, 255), -1) # 감지된 차선 중심 또는 화면 중앙
            
            # offset 값은 차선이 감지될 때만 의미가 있으므로, 해당 상황에 맞게 표시
            offset_display = offset if M["m00"] > 200 else "N/A"
            cv2.putText(vis, f"Offset: {offset_display}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.putText(vis, f"Depth (display only): {depth_val} mm", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            
            # ROI 영역 표시
            cv2.rectangle(vis, (0, roi_y), (w-1, h-1), (255, 0, 0), 2)
            
            cv2.imshow("Lane ROI Tracking", vis)
            
            full_mask = np.zeros_like(rgb, dtype=np.uint8)
            full_mask[roi_y:h, :] = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            cv2.imshow("HSV Mask", full_mask)
    # ===================================================================
    # END: 주행 로직
    # ===================================================================
            
    def destroy_node(self):
        self.get_logger().info("===== 노드 종료 절차 시작 =====")
        self.get_logger().info("1. 로봇 정지 명령 발행...")
        self.stop()
        if self.client_socket:
            self.get_logger().info("2. 소켓 연결 닫는 중...")
            self.client_socket.close()
        if hasattr(self, 'rfid_proc') and self.rfid_proc.poll() is None:
            self.get_logger().info("3. RFID 자식 프로세스 종료 중...")
            self.rfid_proc.terminate()
            self.rfid_proc.wait(timeout=1) # 프로세스가 종료될 때까지 최대 1초 대기
        time.sleep(0.1)
        super().destroy_node()
        self.get_logger().info("===== 노드 종료 완료 =====")

    def callback(self, rgb_msg, depth_msg):
        try:
            rgb = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
            depth = self.bridge.imgmsg_to_cv2(depth_msg, '16UC1')
            if not self.image_queue.full():
                self.image_queue.put((rgb, depth))
        except Exception as e:
            self.get_logger().error(f"이미지 변환 오류: {e}")

    def process_images_and_control_robot(self):
        while rclpy.ok():
            key = -1
            if SHOW_GUI:
                key = cv2.waitKey(1) & 0xFF

            if self.paused:
                self.stop()
                if key == ord('w'):
                    self.paused = False
                    self.get_logger().info("--- 로봇 동작 재개 (w) ---")
                elif key == 27: # ESC
                    self.get_logger().info("--- 종료 (ESC) ---")
                    raise KeyboardInterrupt
                time.sleep(0.1) # CPU 사용량 줄이기
                continue

            if getattr(self, 'rfid_stop_flag', False):
                self.stop()
                self.get_logger().info(f"RFID 인식: 2초간 정지 (UID: {self.rfid_stop_uid})")
                self.send_message(self.rfid_stop_uid)
                time.sleep(2)
                self.rfid_stop_flag = False
                self.rfid_stop_uid = None
                continue # RFID 정지 후에는 바로 다음 루프로 넘어감

            try:
                rgb, depth = self.image_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # 메인 로직인 drive_logic 함수 호출
            self.drive_logic(rgb, depth)

            if SHOW_GUI:
                if key == ord('q'):
                    self.paused = True
                    self.get_logger().info("--- 로봇 동작 일시정지 (q) ---")
                elif key == 27: # ESC 키
                    self.get_logger().info("--- 종료 (ESC) ---")
                    raise KeyboardInterrupt

    def handle_rfid_stop(self, uid):
        self.rfid_stop_flag = True
        self.rfid_stop_uid = uid

    def start_rfid_process(self):
        script_path = os.path.join(os.path.dirname(__file__), "rfid_reader_process.py")
        if not os.path.exists(script_path):
             self.get_logger().error(f"RFID 스크립트 '{script_path}'를 찾을 수 없습니다!")
             return
        self.rfid_proc = subprocess.Popen(["python3", script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        threading.Thread(target=self.rfid_process_loop, daemon=True).start()

    def rfid_process_loop(self):
        for line in iter(self.rfid_proc.stdout.readline, ''):
            if not rclpy.ok(): break
            uid = line.strip()
            if uid:
                self.get_logger().info(f"RFID UID 수신: {uid}")
                self.handle_rfid_stop(uid)
        self.get_logger().warn("RFID 프로세스 루프가 종료되었습니다.")


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = ColorDepthTracker()
        spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
        spin_thread.start()
        node.process_images_and_control_robot()
    except KeyboardInterrupt:
        if node:
            node.get_logger().info("키보드 인터럽트 수신. 안전하게 종료합니다.")
    except Exception as e:
        if node:
            node.get_logger().error(f"메인 루프에서 처리되지 않은 예외 발생: {e}")
            traceback.print_exc()
    finally:
        if node:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if SHOW_GUI:
            cv2.destroyAllWindows()
        print("프로그램이 완전히 종료되었습니다.")

if __name__ == '__main__':
    main()
