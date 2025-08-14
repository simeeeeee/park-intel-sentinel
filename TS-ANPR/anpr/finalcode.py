#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import cv2
import json
import time
import queue
import ctypes
import platform
import logging
import threading
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import message_filters
from tsanpr.tsanpr import TSANPR
from multiprocessing import Process, Queue
import requests

SHOW_GUI = True
SAVE_IMAGES = False
EXAMPLES_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s', datefmt='%H:%M:%S')


def get_engine_file_name():
    arch = platform.machine()
    if sys.platform.startswith("linux"):
        if arch == "aarch64":
            return os.path.join(EXAMPLES_BASE_DIR, "bin/linux-aarch64/libtsanpr.so")
        elif arch in ("x86_64", "amd64"):
            return os.path.join(EXAMPLES_BASE_DIR, "bin/linux-x86_64/libtsanpr.so")
    return ""


def get_pixel_format(img):
    channels = 1 if len(img.shape) == 2 else img.shape[2]
    return {1: "GRAY", 2: "BGR565", 3: "BGR", 4: "BGRA"}.get(channels, None)


def recognize_from_frame(tsanpr, frame, label):
    height, width = frame.shape[:2]
    stride = frame.strides[0]
    pixel_format = get_pixel_format(frame)
    if not pixel_format:
        logging.error(f"{label} 알 수 없는 픽셀 포맷")
        return []
    try:
        result_json = tsanpr.anpr_read_pixels(
            frame.ctypes.data_as(ctypes.c_void_p),
            width, height, stride, pixel_format, "json", "m"
        )
        plates = json.loads(result_json) if result_json else []
        results = []
        for plate in plates:
            area = plate.get("area", {})
            size = area.get("w", 0) * area.get("h", 0)
            results.append({
                "text": plate.get("text", ""),
                "ev": "EV" if plate.get("ev", False) else "일반",
                "size": size,
                "camera": label
            })
        return results
    except Exception as e:
        logging.error(f"{label} 처리 오류: {e}")
        return []


def rfid_process_main(uid_queue):
    from MFRC522 import MFRC522
    import time
    import logging

    logging.basicConfig(level=logging.INFO)
    logging.info("[RFID] RFID 프로세스 시작")

    reader = MFRC522()
    last_uid = None
    last_time = 0

    while True:
        status, tag_type = reader.MFRC522_Request(reader.PICC_REQIDL)
        if status == reader.MI_OK:
            status, uid = reader.MFRC522_SelectTagSN()
            if status == reader.MI_OK:
                uid_str = ''.join([f"{x:02X}" for x in uid[::-1]])
                current_time = time.time()

                if uid_str != last_uid or (current_time - last_time > 5):
                    logging.info(f"[RFID] UID 감지: {uid_str}")
                    uid_queue.put(uid_str)
                    last_uid = uid_str
                    last_time = current_time
        time.sleep(0.2)


class ColorDepthTSNode(Node):
    def __init__(self):
        super().__init__('color_depth_ts_node')
        self.bridge = CvBridge()
        self.hsv_config = self.load_hsv_config()
        self.tsanpr = self.init_tsanpr()
        self.image_queue = queue.Queue(maxsize=1)
        self.rfid_stop_uid = None
        self.left_frame = None
        self.right_frame = None

        self.last_plate_data = {}

        self.mecanum_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        rgb_sub = message_filters.Subscriber(self, Image, '/ascamera/camera_publisher/rgb0/image')
        depth_sub = message_filters.Subscriber(self, Image, '/ascamera/camera_publisher/depth0/image_raw')
        sync = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size=10, slop=0.05)
        sync.registerCallback(self.callback)

        self.stopped = False  # 주행 중지 상태 초기값 False
        self.is_stopped = False  # stop() 중복 호출 방지 플래그

        # Multiprocessing RFID 관련
        self.rfid_queue = Queue()
        self.rfid_proc = Process(target=rfid_process_main, args=(self.rfid_queue,))
        self.rfid_proc.start()

        # RFID 수신 대기 스레드
        threading.Thread(target=self.rfid_queue_listener, daemon=True).start()

        # 트랙바 및 윈도우 생성
        if SHOW_GUI:
            cv2.namedWindow("HSV Thresholds")
            for k in ["h_min", "h_max", "s_min", "s_max", "v_min", "v_max"]:
                max_val = 179 if 'h' in k else 255
                cv2.createTrackbar(k, "HSV Thresholds", self.hsv_config[k], max_val, lambda x: None)
            cv2.namedWindow("Left Camera")
            cv2.namedWindow("Right Camera")

        threading.Thread(target=self.image_processing_loop, daemon=True).start()
        threading.Thread(target=self.capture_camera, args=(0, 'left'), daemon=True).start()
        threading.Thread(target=self.capture_camera, args=(2, 'right'), daemon=True).start()
        threading.Thread(target=self.keyboard_monitor, daemon=True).start()

        self.get_logger().info("통합 노드 실행 시작")

    def load_hsv_config(self, json_name="HSV-cal.json"):
        default = {"h_min": 9, "h_max": 51, "s_min": 43, "s_max": 255, "v_min": 54, "v_max": 255}
        path = os.path.join(os.path.dirname(__file__), json_name)
        if os.path.exists(path):
            try:
                return json.load(open(path))
            except Exception:
                return default
        return default

    def init_tsanpr(self):
        path = get_engine_file_name()
        if not path or not os.path.exists(path):
            self.get_logger().error("TSANPR 엔진 경로 없음")
            return None
        tsanpr = TSANPR(path)
        if tsanpr.anpr_initialize("json;country=KR;multi=true;func=m"):
            self.get_logger().error("TSANPR 초기화 실패")
            return None
        return tsanpr

    def callback(self, rgb_msg, depth_msg):
        try:
            rgb = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
            depth = self.bridge.imgmsg_to_cv2(depth_msg, '16UC1')
            if not self.image_queue.empty():
                _ = self.image_queue.get_nowait()
            self.image_queue.put_nowait((rgb, depth))
        except Exception as e:
            self.get_logger().error(f"이미지 변환 실패: {e}")

    def keyboard_monitor(self):
        import select
        import termios
        import tty
        old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        self.get_logger().info("키보드 입력 대기 중... (q = 구동계 정지, w = 주행 재개, ESC = 종료)")
        try:
            while True:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1)
                    if key == 'q':
                        self.stopped = True
                        self.stop()  # 구동계 정지(0,0)
                        self.get_logger().info("q 키 입력: 구동계 정지")
                    elif key == 'w':
                        self.stopped = False
                        self.get_logger().info("w 키 입력: 주행 재개")
                    elif key == '\x1b':  # ESC 키
                        self.get_logger().info("ESC 키 입력: 노드 종료")
                        self.stop()
                        rclpy.shutdown()
                        break
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    def image_processing_loop(self):
        last_gui_time = time.time()
        while rclpy.ok():
            try:
                rgb, depth = self.image_queue.get(timeout=0.05)

                if SHOW_GUI:
                    for k in self.hsv_config:
                        self.hsv_config[k] = cv2.getTrackbarPos(k, "HSV Thresholds")

                if not self.stopped:
                    self.drive_logic(rgb, depth)
                else:
                    self.stop()

                if SHOW_GUI and (time.time() - last_gui_time > 0.1):
                    if self.left_frame is not None:
                        cv2.imshow("Left Camera", self.left_frame)
                    if self.right_frame is not None:
                        cv2.imshow("Right Camera", self.right_frame)

                    h, w = rgb.shape[:2]
                    roi_y = h // 2
                    roi = rgb[roi_y:h, :]
                    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    lower = np.array([self.hsv_config[k] for k in ("h_min", "s_min", "v_min")])
                    upper = np.array([self.hsv_config[k] for k in ("h_max", "s_max", "v_max")])
                    mask = cv2.inRange(hsv, lower, upper)
                    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                    vis = np.vstack((roi, mask_bgr))
                    cv2.imshow("HSV Thresholds", vis)
                    last_gui_time = time.time()

                key = cv2.waitKey(10) & 0xFF

            except queue.Empty:
                continue

        cv2.destroyAllWindows()

    def drive_logic(self, rgb, depth):
        h, w = rgb.shape[:2]
        roi_y = h // 2
        hsv = cv2.cvtColor(rgb[roi_y:h, :], cv2.COLOR_BGR2HSV)
        lower = np.array([self.hsv_config[k] for k in ("h_min", "s_min", "v_min")])
        upper = np.array([self.hsv_config[k] for k in ("h_max", "s_max", "v_max")])
        mask = cv2.inRange(hsv, lower, upper)
        M = cv2.moments(mask)
        cx = w // 2

        if M["m00"] == 0:
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.2
            self.mecanum_pub.publish(twist)
            self.get_logger().info("선이 안보임 -> 왼쪽 회전")
            return

        cx = int(M["m10"] / M["m00"])
        offset = cx - (w // 2)
        depth_cx, depth_cy = cx, roi_y + (h - roi_y) // 2
        depth_val = depth[depth_cy, depth_cx] if 0 <= depth_cx < w and 0 <= depth_cy < h else 0

        if depth_val > 0 and depth_val < 500:
            if not self.is_stopped:
                self.stop()
        else:
            self.is_stopped = False
            twist = Twist()
            twist.linear.x = 0.1
            max_angular = 0.5
            max_offset = w // 2
            angular_z = -max_angular * (offset / max_offset)
            if abs(angular_z) < 0.02:
                angular_z = 0.0
            twist.angular.z = angular_z
            self.mecanum_pub.publish(twist)

    def capture_camera(self, cam_id, side):
        cap = cv2.VideoCapture(cam_id, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 15)

        if not cap.isOpened():
            self.get_logger().error(f"[{side.upper()} CAMERA] 열기 실패 (ID: {cam_id})")
            return

        self.get_logger().info(f"[{side.upper()} CAMERA] 캡처 시작 (ID: {cam_id})")
        while True:
            ret, frame = cap.read()
            if not ret:
                self.get_logger().warning(f"[{side.upper()} CAMERA] 프레임 획득 실패")
                time.sleep(0.1)
                continue
            if side == 'left':
                self.left_frame = frame
            else:
                self.right_frame = frame
            time.sleep(0.03)

    def capture_and_infer(self):
        if self.left_frame is None or self.right_frame is None:
            self.get_logger().error("카메라 프레임 없음")
            return

        # left_frame : 라이트 카메라 영상 (왼쪽 카메라)
        # right_frame : 레프트 카메라 영상 (오른쪽 카메라)
        frame_left_cam = self.left_frame   # 라이트 카메라 (왼쪽 카메라)
        frame_right_cam = self.right_frame  # 레프트 카메라 (오른쪽 카메라)

        h, w = frame_left_cam.shape[:2]
        mid = w // 2

        zones = {
            "ZONE1": frame_left_cam[:, :mid],    # 왼쪽 카메라 왼쪽 영역
            "ZONE2": frame_left_cam[:, mid:],    # 왼쪽 카메라 오른쪽 영역
            "ZONE3": frame_right_cam[:, :mid],   # 오른쪽 카메라 왼쪽 영역
            "ZONE4": frame_right_cam[:, mid:]    # 오른쪽 카메라 오른쪽 영역
        }

        self.last_plate_data = {}
        for zone, img in zones.items():
            results = recognize_from_frame(self.tsanpr, img, zone)
            if results:
                top = sorted(results, key=lambda x: x["size"], reverse=True)[0]
                self.last_plate_data[zone] = {
                    "text": top["text"],
                    "ev": top["ev"]
                }
                self.get_logger().info(f"[{zone}] 번호: {top['text']} / {top['ev']}")
            else:
                self.last_plate_data[zone] = {
                    "text": "",
                    "ev": ""
                }
                self.get_logger().info(f"[{zone}] 번호 인식 실패")            

    def send_plate_data_to_server(self):
        if not self.rfid_stop_uid:
            return  # RFID UID 없으면 전송 안함

        payload = {
            "rfid": self.rfid_stop_uid,
            "vehicles": self.last_plate_data if self.last_plate_data else None  # 빈 dict 대신 None
        }
        try:
            res = requests.post("https://222.234.38.97:8443/api/robot/status", json=payload, timeout=5, verify=False)
            self.get_logger().info(f"서버 응답: {res.status_code} {res.text}")
        except Exception as e:
            self.get_logger().error(f"서버 전송 실패: {e}")
        
    def rfid_queue_listener(self):
        self.get_logger().info("[ROS] RFID 메시지 수신 대기")
        while True:
            try:
                uid = self.rfid_queue.get(timeout=1.0)
                if uid:
                    self.get_logger().info(f"[ROS] RFID 수신: {uid}")
                    self.stopped = True
                    self.stop()
                    self.rfid_stop_uid = uid
                    self.capture_and_infer()
                    self.send_plate_data_to_server()
                    self.stopped = False
                    self.rfid_stop_uid = None
            except queue.Empty:
                continue

    def stop(self):
        if self.is_stopped:
            return
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.mecanum_pub.publish(twist)
        self.get_logger().debug("구동계 정지: twist.linear.x = 0, twist.angular.z = 0")
        self.is_stopped = True


def main(args=None):
    node = None
    try:
        rclpy.init(args=args)
        node = ColorDepthTSNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"[ERROR] main 예제: {e}")
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if 'node' in locals() and hasattr(node, 'rfid_proc'):
            if node.rfid_proc.is_alive():
                node.rfid_proc.terminate()
                node.rfid_proc.join()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
