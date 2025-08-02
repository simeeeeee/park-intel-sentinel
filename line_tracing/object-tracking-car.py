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

class ColorDepthTracker(Node):
    def __init__(self):
        super().__init__('color_depth_tracker')
        self.bridge = CvBridge()

        # 트랙바용 LAB 설정값 불러오기
        self.lab_config = self.load_lab_config()

        # 트랙바 UI
        self.init_trackbars()
        
        
        # Twist 퍼블리셔
        self.mecanum_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # RGB + Depth 구독
        rgb_sub = message_filters.Subscriber(self, Image, '/ascamera/camera_publisher/rgb0/image')
        depth_sub = message_filters.Subscriber(self, Image, '/ascamera/camera_publisher/depth0/image_raw')
        sync = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size=10, slop=0.05)
        sync.registerCallback(self.callback)

    def load_lab_config(self):
        default = {
            "l_min": 0, "l_max": 255,
            "a_min": 0, "a_max": 255,
            "b_min": 0, "b_max": 255
        }
        if os.path.exists("LAB-cal.json"):
            try:
                with open("LAB-cal.json", "r") as f:
                    return json.load(f)
            except:
                return default
        return default

    def init_trackbars(self):
        def nothing(x): pass
        cv2.namedWindow("Trackbars")
        cv2.resizeWindow("Trackbars", 800, 260)
        cv2.createTrackbar("L Min", "Trackbars", self.lab_config["l_min"], 255, nothing)
        cv2.createTrackbar("L Max", "Trackbars", self.lab_config["l_max"], 255, nothing)
        cv2.createTrackbar("A Min", "Trackbars", self.lab_config["a_min"], 255, nothing)
        cv2.createTrackbar("A Max", "Trackbars", self.lab_config["a_max"], 255, nothing)
        cv2.createTrackbar("B Min", "Trackbars", self.lab_config["b_min"], 255, nothing)
        cv2.createTrackbar("B Max", "Trackbars", self.lab_config["b_max"], 255, nothing)

    def go_straight(self, speed):
        twist = Twist()
        if speed > 0.0 and speed <= 1.0:
            twist.linear.x = speed
        else:
            twist.linear.x = 0.1
        twist.linear.y = 0.0
        twist.linear.z = 0.0
        twist.angular.x=0.0
        twist.angular.y=0.0
        twist.angular.z=0.0
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
            
    def callback(self, rgb_msg, depth_msg):
        rgb = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
        depth = self.bridge.imgmsg_to_cv2(depth_msg, '16UC1')
        print("test")
        # 트랙바에서 값 읽기
        l_min = cv2.getTrackbarPos("L Min", "Trackbars")
        l_max = cv2.getTrackbarPos("L Max", "Trackbars")
        a_min = cv2.getTrackbarPos("A Min", "Trackbars")
        a_max = cv2.getTrackbarPos("A Max", "Trackbars")
        b_min = cv2.getTrackbarPos("B Min", "Trackbars")
        b_max = cv2.getTrackbarPos("B Max", "Trackbars")

        lower = np.array([l_min, a_min, b_min])
        upper = np.array([l_max, a_max, b_max])

        lab = cv2.cvtColor(rgb, cv2.COLOR_BGR2LAB)
        mask = cv2.inRange(lab, lower, upper)
        result = cv2.bitwise_and(rgb, rgb, mask=mask)

        # 컨투어 검출
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            biggest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(biggest)
            if area > 500:
                x, y, w, h = cv2.boundingRect(biggest)
                cx, cy = x + w // 2, y + h // 2
                frame_cx = rgb.shape[1] // 2
                offset = cx - frame_cx
                if 0 <= cy < depth.shape[0] and 0 <= cx < depth.shape[1]:
                    depth_val = depth[cy, cx]
                    cv2.putText(result, f"{depth_val} mm", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    if depth_val > 0 and depth_val < 500:
                        self.stop()
                    else:
                        if offset < -40:
                            self.go_left(0.2)
                        elif offset > 40:
                            self.go_right(0.2)
                        else:
                            self.go_straight(0.2)
                
                cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                

        combined = np.hstack((rgb, result))
        cv2.imshow("Color Tracked + Depth", cv2.resize(combined, (1280, 480)))
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = ColorDepthTracker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
