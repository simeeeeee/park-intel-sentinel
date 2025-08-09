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

# --- 4번 개선 사항: GUI 로직 분리 ---
class Visualizer:
    """GUI 시각화와 관련된 모든 작업을 처리하는 클래스"""
    def __init__(self, hsv_config):
        self.hsv_config = hsv_config
        cv2.namedWindow("Lane ROI Tracking")
        cv2.namedWindow("HSV Mask")
        self.init_trackbars()

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

    def update(self, vis, mask, roi_y, depth_info, offset_info):
        """시각화 정보를 받아서 화면에 그려주는 메소드"""
        h, w = vis.shape[:2]
        depth_cx, depth_cy, depth_val = depth_info
        offset = offset_info

        # 화면 중앙선 및 감지된 차선 중심점 그리기
        cv2.line(vis, (w // 2, 0), (w // 2, h), (0, 255, 0), 1)
        cv2.circle(vis, (depth_cx, depth_cy), 5, (0, 0, 255), -1)

        # 정보 텍스트 표시
        offset_display = offset if offset != "N/A" else "N/A"
        cv2.putText(vis, f"Offset: {offset_display}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(vis, f"Depth (display only): {depth_val} mm", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        
        # ROI 영역 표시
        cv2.rectangle(vis, (0, roi_y), (w-1, h-1), (255, 0, 0), 2)
        
        # 마스크 이미지 생성
        full_mask = np.zeros_like(vis, dtype=np.uint8)
        full_mask[roi_y:h, :] = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        cv2.imshow("Lane ROI Tracking", vis)
        cv2.imshow("HSV Mask", full_mask)

        return cv2.waitKey(1) & 0xFF


class ColorDepthTracker(Node):
    def __init__(self):
        super().__init__('color_depth_tracker')
        self.bridge = CvBridge()
        self.paused = False

        self.hsv_config = self.load_hsv_config()

        # --- 4번 개선 사항: Visualizer 인스턴스 생성 ---
        self.visualizer = Visualizer(self.hsv_config) if SHOW_GUI else None
        
        self.mecanum_pub = self.create_publisher(Twist, 'cmd_vel', 10) 
        self.image_queue = queue.Queue(maxsize=2)

        # --- 3번, 5번 개선 사항 적용 ---
        self.state_lock = threading.Lock() # 5번: 스레드 잠금 장치
        self.rfid_stop_flag = False
        self.rfid_stop_uid = None
        
        # 3번: 비동기 메시지 전송을 위한 큐와 스레드
        self.message_queue = queue.Queue()
        self.socket_thread = threading.Thread(target=self._socket_worker, daemon=True)
        self.socket_thread.start()

        rgb_sub = message_filters.Subscriber(self, Image, '/ascamera/camera_publisher/rgb0/image')
        depth_sub = message_filters.Subscriber(self, Image, '/ascamera/camera_publisher/depth0/image_raw')
        sync = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size=10, slop=0.05)
        sync.registerCallback(self.callback)

        self.get_logger().info("ColorDepthTracker 노드가 시작되었습니다. (일시정지: q, 재개: w, 종료: ESC)")
        self.start_rfid_process()

    # --- 3번 개선 사항: 비동기 소켓 워커 스레드 ---
    def _socket_worker(self):
        """백그라운드에서 소켓 통신을 전담하는 워커 스레드"""
        HOST = '127.0.0.1'
        PORT = 10004
        client_socket = None
        
        while rclpy.ok():
            try:
                # 큐에서 메시지를 기다림 (블로킹)
                message = self.message_queue.get()
                if message is None: # 종료 신호
                    break

                # 소켓 연결 확인 및 시도
                if client_socket is None:
                    try:
                        self.get_logger().info(f"서버({HOST}:{PORT})에 연결을 시도합니다...")
                        new_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        new_socket.settimeout(2) # 연결 타임아웃
                        new_socket.connect((HOST, PORT))
                        new_socket.settimeout(None)
                        client_socket = new_socket
                        self.get_logger().info("✅ 서버 연결에 성공했습니다.")
                    except Exception as e:
                        self.get_logger().error(f"🚨 서버 연결 실패: {e}. 메시지를 큐에 다시 넣습니다.")
                        self.message_queue.put(message) # 실패 시 메시지 복구
                        time.sleep(5) # 5초 후 재시도
                        continue
                
                # 메시지 전송
                try:
                    client_socket.sendall(message.encode('utf-8'))
                    self.get_logger().info(f"메시지 전송 성공: {message}")
                except (BrokenPipeError, ConnectionResetError) as e:
                    self.get_logger().warn(f"소켓 연결이 끊어졌습니다: {e}. 재연결을 시도합니다.")
                    if client_socket:
                        client_socket.close()
                    client_socket = None
                    self.message_queue.put(message) # 실패 시 메시지 복구
                
            except Exception as e:
                self.get_logger().error(f"소켓 워커에서 예기치 않은 오류 발생: {e}")
                if client_socket:
                    client_socket.close()
                client_socket = None
                time.sleep(5) # 5초 후 재시도

        if client_socket:
            client_socket.close()
        self.get_logger().info("소켓 워커 스레드가 종료되었습니다.")
        
    def send_message_async(self, message):
        """메시지를 큐에 넣어 비동기 전송을 요청"""
        if message:
            self.message_queue.put(message)
            return True
        return False
        
    def load_hsv_config(self, json_name="HSV-cal.json"):
        # (이전과 동일)
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

    def stop(self):
        twist=Twist()
        twist.linear.x, twist.linear.y, twist.linear.z = 0.0, 0.0, 0.0
        twist.angular.x, twist.angular.y, twist.angular.z = 0.0, 0.0, 0.0
        self.mecanum_pub.publish(twist)

    def drive_logic(self, rgb, depth):
        h, w = rgb.shape[:2]
        vis = rgb.copy() if SHOW_GUI else None # GUI가 켜져 있을 때만 이미지 복사

        roi_y = h // 2
        roi_hsv = cv2.cvtColor(rgb[roi_y:h, :], cv2.COLOR_BGR2HSV)

        if self.visualizer:
            lower, upper = self.visualizer.get_trackbar_values()
        else:
            lower = np.array([self.hsv_config["h_min"], self.hsv_config["s_min"], self.hsv_config["v_min"]])
            upper = np.array([self.hsv_config["h_max"], self.hsv_config["s_max"], self.hsv_config["v_max"]])
        
        mask = cv2.inRange(roi_hsv, lower, upper)
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        M = cv2.moments(mask)
        twist = Twist()
        
        offset = "N/A" # 오프셋 기본값
        depth_cx_display = w // 2 # 중심점 기본값

        if M["m00"] > 200:
            cx = int(M["m10"] / M["m00"])
            offset_val = cx - (w // 2)
            offset = offset_val # 오프셋 값 업데이트
            depth_cx_display = cx # 중심점 업데이트
            
            twist.linear.x = 0.1
            max_angular = 0.5
            max_offset = w // 2
            angular_z = -max_angular * (offset_val / max_offset)
            
            if abs(angular_z) < 0.02: angular_z = 0.0
            twist.angular.z = np.clip(angular_z, -max_angular, max_angular) 
        else:
            self.get_logger().warn('Line not detected, executing recovery behavior (turning left).')
            twist.linear.x = 0.1
            twist.angular.z = 0.5

        self.mecanum_pub.publish(twist)

        # --- 4번 개선 사항: 시각화 데이터 전달 ---
        if self.visualizer:
            depth_cy = roi_y + (h - roi_y) // 2
            depth_val = 0
            if 0 <= depth_cx_display < w and 0 <= depth_cy < h:
                depth_val = depth[depth_cy, depth_cx_display]
            
            # Visualizer에 필요한 모든 정보를 딕셔너리로 묶어 전달
            depth_info = (depth_cx_display, depth_cy, depth_val)
            
            # Visualizer의 update 메소드 호출
            return self.visualizer.update(vis, mask, roi_y, depth_info, offset)
        
        return cv2.waitKey(1) & 0xFF # GUI가 없을 때도 루프를 위한 가짜 key 값 반환

    def destroy_node(self):
        self.get_logger().info("===== 노드 종료 절차 시작 =====")
        self.get_logger().info("1. 로봇 정지 명령 발행...")
        self.stop()
        if hasattr(self, 'rfid_proc') and self.rfid_proc.poll() is None:
            self.get_logger().info("2. RFID 자식 프로세스 종료 중...")
            self.rfid_proc.terminate()
            self.rfid_proc.wait(timeout=1)
        if hasattr(self, 'message_queue'):
             self.message_queue.put(None) # 3. 소켓 워커 스레드에 종료 신호 전송
             self.socket_thread.join(timeout=1) # 스레드가 끝날 때까지 대기
        time.sleep(0.1)
        super().destroy_node()
        self.get_logger().info("===== 노드 종료 완료 =====")

    def callback(self, rgb_msg, depth_msg):
        # (이전과 동일)
        try:
            rgb = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
            depth = self.bridge.imgmsg_to_cv2(depth_msg, '16UC1')
            if not self.image_queue.full():
                self.image_queue.put((rgb, depth))
        except Exception as e:
            self.get_logger().error(f"이미지 변환 오류: {e}")

    def process_images_and_control_robot(self):
        while rclpy.ok():
            key = -1 # 키 기본값
            
            # --- 5번 개선 사항: 공유 변수 접근 시 잠금 ---
            rfid_triggered = False
            uid_to_send = None
            with self.state_lock:
                if self.rfid_stop_flag:
                    rfid_triggered = True
                    uid_to_send = self.rfid_stop_uid
                    self.rfid_stop_flag = False # 확인 후 즉시 플래그 리셋
                    self.rfid_stop_uid = None

            if rfid_triggered:
                self.stop()
                self.get_logger().info(f"RFID 인식: 2초간 정지 (UID: {uid_to_send})")
                self.send_message_async(uid_to_send) # 3번: 비동기 메시지 전송
                time.sleep(2)
                continue

            if self.paused:
                self.stop()
                if self.visualizer:
                    key = cv2.waitKey(100) & 0xFF # CPU 사용량 줄이기
                else:
                    time.sleep(0.1)

                if key == ord('w'):
                    self.paused = False
                    self.get_logger().info("--- 로봇 동작 재개 (w) ---")
                elif key == 27:
                    raise KeyboardInterrupt
                continue

            try:
                rgb, depth = self.image_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            key = self.drive_logic(rgb, depth)

            if key == ord('q'):
                self.paused = True
                self.get_logger().info("--- 로봇 동작 일시정지 (q) ---")
            elif key == 27: # ESC 키
                self.get_logger().info("--- 종료 (ESC) ---")
                raise KeyboardInterrupt

    # --- 5번 개선 사항: 공유 변수 수정 시 잠금 ---
    def handle_rfid_stop(self, uid):
        with self.state_lock:
            self.rfid_stop_flag = True
            self.rfid_stop_uid = uid

    def start_rfid_process(self):
        # (이전과 동일)
        script_path = os.path.join(os.path.dirname(__file__), "rfid_reader_process.py")
        if not os.path.exists(script_path):
             self.get_logger().error(f"RFID 스크립트 '{script_path}'를 찾을 수 없습니다!")
             return
        self.rfid_proc = subprocess.Popen(["python3", script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        threading.Thread(target=self.rfid_process_loop, daemon=True).start()

    def rfid_process_loop(self):
        # (이전과 동일)
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