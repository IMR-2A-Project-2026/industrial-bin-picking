#!/usr/bin/env python3
"""
grasp_detector.py — YOLO grasp detection on RealSense D435 ROS topics.

Pipeline:
  1. Subscribe to /camera/camera/color/image_raw       (RGB)
  2. Subscribe to /camera/camera/aligned_depth_to_color/image_raw (depth)
  3. Subscribe to /camera/camera/color/camera_info     (intrinsics)
  4. YOLO inference -> grasp pixel
  5. Read depth at grasp pixel -> back-project to 3D in camera frame
  6. TF lookup base_link <- camera_color_optical_frame (via handeye calib)
  7. Print and visualize the 3D grasp point in base_link.

Prerequisites (each in its own terminal):
  - UR driver:           ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur3e robot_ip:=192.168.1.102
  - RealSense w/ depth:  ros2 launch realsense2_camera rs_launch.py enable_color:=true enable_depth:=true align_depth.enable:=true rgb_camera.color_profile:=640x480x30 depth_module.depth_profile:=640x480x30
  - Handeye publish:     ros2 launch easy_handeye2 publish.launch.py name:=ur3e_d435
"""

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from ultralytics import YOLO
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import PointStamped
import tf2_geometry_msgs  # noqa: F401  (registers PointStamped transform)
import message_filters

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_PATH      = "/home/julien/ur3e_ws/best.pt"   # adjust if needed
TARGET_CLASS    = "hole"
CONF_THRESHOLD  = 0.5

COLOR_TOPIC     = "/camera/camera/color/image_raw"
DEPTH_TOPIC     = "/camera/camera/aligned_depth_to_color/image_raw"
CAMINFO_TOPIC   = "/camera/camera/color/camera_info"

CAMERA_FRAME    = "camera_color_optical_frame"
ROBOT_FRAME     = "base_link"


# ─────────────────────────────────────────────
# GRASP STRATEGY
# ─────────────────────────────────────────────
def grasp_upper_center(x1, y1, x2, y2, offset_ratio=0.20):
    cx = int((x1 + x2) / 2)
    h  = y2 - y1
    cy = int(y1 + h * (0.5 - offset_ratio))
    return cx, cy


# ─────────────────────────────────────────────
# NODE
# ─────────────────────────────────────────────
class GraspDetector(Node):
    def __init__(self):
        super().__init__("grasp_detector")

        self.bridge = CvBridge()
        self.get_logger().info(f"Loading YOLO model: {MODEL_PATH}")
        self.model = YOLO(MODEL_PATH)
        self.get_logger().info("Model loaded.")

        # camera intrinsics (filled by camera_info callback)
        self.fx = self.fy = self.cx = self.cy = None

        # TF
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # camera_info subscriber (one-shot effect)
        self.create_subscription(
            CameraInfo, CAMINFO_TOPIC, self.camera_info_cb, 10
        )

        # synchronized color + depth
        color_sub = message_filters.Subscriber(self, Image, COLOR_TOPIC)
        depth_sub = message_filters.Subscriber(self, Image, DEPTH_TOPIC)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub], queue_size=10, slop=0.1
        )
        self.sync.registerCallback(self.image_cb)

        self.get_logger().info("GraspDetector ready. Waiting for frames...")

    # ───── callbacks ────────────────────────────────────────────────
    def camera_info_cb(self, msg: CameraInfo):
        if self.fx is None:
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]
            self.get_logger().info(
                f"Intrinsics: fx={self.fx:.1f} fy={self.fy:.1f} "
                f"cx={self.cx:.1f} cy={self.cy:.1f}"
            )

    def image_cb(self, color_msg: Image, depth_msg: Image):
        if self.fx is None:
            return  # still waiting for intrinsics

        color = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")

        # ── YOLO inference ──────────────────────────────────────────
        results = self.model(color, conf=CONF_THRESHOLD, verbose=False)[0]

        best_box, best_conf = None, 0.0
        for box in results.boxes:
            cls_name = self.model.names[int(box.cls)]
            conf     = float(box.conf)
            if cls_name == TARGET_CLASS and conf > best_conf:
                best_conf = conf
                best_box  = box.xyxy[0].cpu().numpy()

        vis = color.copy()

        if best_box is None:
            cv2.putText(vis, f"No '{TARGET_CLASS}' detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Grasp Detector", vis)
            cv2.waitKey(1)
            return

        x1, y1, x2, y2 = map(int, best_box)
        gx, gy = grasp_upper_center(x1, y1, x2, y2)

        # ── Depth lookup at grasp pixel ─────────────────────────────
        h_d, w_d = depth.shape[:2]
        if not (0 <= gx < w_d and 0 <= gy < h_d):
            self.get_logger().warn(f"Grasp pixel ({gx},{gy}) outside depth image")
            return

        depth_raw = depth[gy, gx]
        depth_m   = float(depth_raw) / 1000.0   # RealSense: mm → m

        if depth_m <= 0.0 or np.isnan(depth_m):
            self.get_logger().warn(
                f"Invalid depth at ({gx},{gy}) = {depth_raw}"
            )
            return

        # ── Back-project pixel + depth -> camera frame ──────────────
        X = (gx - self.cx) * depth_m / self.fx
        Y = (gy - self.cy) * depth_m / self.fy
        Z = depth_m

        # ── TF transform to base_link ───────────────────────────────
        pt = PointStamped()
        pt.header.frame_id = CAMERA_FRAME
        pt.header.stamp    = color_msg.header.stamp
        pt.point.x = X
        pt.point.y = Y
        pt.point.z = Z

        try:
            pt_base = self.tf_buffer.transform(
                pt, ROBOT_FRAME, timeout=Duration(seconds=1.0)
            )
        except Exception as exc:
            self.get_logger().error(f"TF transform failed: {exc}")
            return

        bx = pt_base.point.x
        by = pt_base.point.y
        bz = pt_base.point.z

        self.get_logger().info(
            f"[{TARGET_CLASS} @ {best_conf:.2f}] "
            f"px=({gx},{gy}) cam=({X:+.3f},{Y:+.3f},{Z:+.3f}) "
            f"base=({bx:+.3f},{by:+.3f},{bz:+.3f}) m"
        )

        # ── Visualize ───────────────────────────────────────────────
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 100, 0), 2)
        cv2.circle(vis,    (gx, gy), 10, (0, 0, 255), -1)
        cv2.circle(vis,    (gx, gy), 14, (255, 255, 255), 2)
        txt = f"base=({bx:+.3f},{by:+.3f},{bz:+.3f}) m"
        cv2.putText(vis, txt, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(vis, f"depth={depth_m:.3f} m", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("Grasp Detector", vis)
        cv2.waitKey(1)


# ─────────────────────────────────────────────
def main():
    rclpy.init()
    node = GraspDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
