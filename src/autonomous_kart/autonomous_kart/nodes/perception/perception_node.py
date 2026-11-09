"""Perception node: karts and obstacles from the camera and the lidar.

https://github.com/EVC-Purdue/A27/issues/30

Each camera frame and each scan becomes one ObjectMeasurementArray in the
body frame, stamped at that sensor's capture time. Filtering over time is
Localization's tracker (#29).

Camera: ROI crop (drops our own nose) -> ONNX detector -> monocular ranging
(ground contact fused with box height). Lidar: scan clusters of kart or
obstacle size. With no model file or no onnxruntime, camera frames still
publish empty arrays and the lidar keeps working.
"""

import math
import os
from pathlib import Path

import numpy as np
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image, LaserScan

from autonomous_kart.common import QOS_IMAGE, QOS_LOW, run_node
from autonomous_kart.nodes.perception.camera import camera_objects
from autonomous_kart.nodes.perception.lidar import LidarParams, scan_objects
from autonomous_kart.nodes.perception.ranging import CameraModel, RangingParams
from kart_interfaces.msg import ObjectMeasurement, ObjectMeasurementArray

PARAMS = {
    "model_path": Parameter.Type.STRING,
    "confidence": Parameter.Type.DOUBLE,
    "nms_iou": Parameter.Type.DOUBLE,
    "cpu_threads": Parameter.Type.INTEGER,
    "roi_frac": Parameter.Type.DOUBLE_ARRAY,
    "kart_footprint_m": Parameter.Type.DOUBLE,
    "camera.width_px": Parameter.Type.INTEGER,
    "camera.height_px": Parameter.Type.INTEGER,
    "camera.fx_px": Parameter.Type.DOUBLE,
    "camera.fy_px": Parameter.Type.DOUBLE,
    "camera.cx_px": Parameter.Type.DOUBLE,
    "camera.cy_px": Parameter.Type.DOUBLE,
    "camera.height_m": Parameter.Type.DOUBLE,
    "camera.pitch_deg": Parameter.Type.DOUBLE,
    "camera.yaw_deg": Parameter.Type.DOUBLE,
    "camera.mount_x_m": Parameter.Type.DOUBLE,
    "camera.mount_y_m": Parameter.Type.DOUBLE,
    "camera.max_range_m": Parameter.Type.DOUBLE,
    "ranging.kart_height_m": Parameter.Type.DOUBLE,
    "ranging.kart_height_sigma_m": Parameter.Type.DOUBLE,
    "ranging.pitch_sigma_deg": Parameter.Type.DOUBLE,
    "ranging.box_sigma_px": Parameter.Type.DOUBLE,
    "lidar.mount_x_m": Parameter.Type.DOUBLE,
    "lidar.mount_y_m": Parameter.Type.DOUBLE,
    "lidar.mount_yaw_deg": Parameter.Type.DOUBLE,
    "lidar.max_range_m": Parameter.Type.DOUBLE,
    "lidar.cluster_gap_m": Parameter.Type.DOUBLE,
    "lidar.min_extent_m": Parameter.Type.DOUBLE,
    "lidar.max_extent_m": Parameter.Type.DOUBLE,
}
FALLBACK_WARN_PERIOD_S = 30.0


def workspace_path(path: str) -> str:
    """Relative paths are under the colcon workspace (install/autonomous_kart/share/autonomous_kart up 4)."""
    if os.path.isabs(path):
        return path
    return str(Path(get_package_share_directory("autonomous_kart")).parents[3] / path)


def image_to_bgr(msg: Image) -> np.ndarray | None:
    if msg.encoding not in ("rgb8", "bgr8"):
        return None
    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)[:, : 3 * msg.width]
    img = img.reshape(msg.height, msg.width, 3)
    return img[:, :, ::-1] if msg.encoding == "rgb8" else img


def to_measurements(header, xy_m, cov_m2, object_class: int, footprints_m) -> ObjectMeasurementArray:
    out = ObjectMeasurementArray()
    out.header.stamp = header.stamp  # capture time, so the tracker places it with our pose then
    out.header.frame_id = "base_link"
    for (x, y), cov, footprint in zip(xy_m, cov_m2, footprints_m, strict=True):
        m = ObjectMeasurement()
        m.object_class = object_class
        m.footprint_size = float(footprint)
        m.position.x, m.position.y = float(x), float(y)
        m.position_covariance = [float(v) for v in np.asarray(cov).reshape(4)]
        out.objects.append(m)
    return out


class PerceptionNode(Node):
    def __init__(self):
        super().__init__("perception_node")
        for name, kind in PARAMS.items():
            self.declare_parameter(name, kind)
        p = {name: self.get_parameter(name).value for name in PARAMS}

        self.camera = CameraModel(
            fx=p["camera.fx_px"],
            fy=p["camera.fy_px"],
            cx=p["camera.cx_px"],
            cy=p["camera.cy_px"],
            width_px=p["camera.width_px"],
            height_px=p["camera.height_px"],
            height_m=p["camera.height_m"],
            pitch_rad=math.radians(p["camera.pitch_deg"]),
            yaw_rad=math.radians(p["camera.yaw_deg"]),
            mount_x_m=p["camera.mount_x_m"],
            mount_y_m=p["camera.mount_y_m"],
        )
        self.ranging = RangingParams(
            kart_height_m=p["ranging.kart_height_m"],
            kart_height_sigma_m=p["ranging.kart_height_sigma_m"],
            center_offset_m=0.5 * p["kart_footprint_m"],
            pitch_sigma_rad=math.radians(p["ranging.pitch_sigma_deg"]),
            box_sigma_px=p["ranging.box_sigma_px"],
        )
        self.lidar = LidarParams(
            mount_x_m=p["lidar.mount_x_m"],
            mount_y_m=p["lidar.mount_y_m"],
            mount_yaw_rad=math.radians(p["lidar.mount_yaw_deg"]),
            max_range_m=p["lidar.max_range_m"],
            cluster_gap_m=p["lidar.cluster_gap_m"],
            min_extent_m=p["lidar.min_extent_m"],
            max_extent_m=p["lidar.max_extent_m"],
        )
        self.roi_frac = list(p["roi_frac"])
        self.footprint_m = p["kart_footprint_m"]
        self.camera_max_range_m = p["camera.max_range_m"]
        self.detector, self.why_no_detector = self.load_detector(p)

        self.create_subscription(Image, "camera/image_raw", self.on_image, QOS_IMAGE)
        self.create_subscription(LaserScan, "scan", self.on_scan, QOS_LOW)
        self.measurements_pub = self.create_publisher(ObjectMeasurementArray, "perception/measurements", QOS_LOW)

    def load_detector(self, p):
        path = workspace_path(p["model_path"])
        if not os.path.exists(path):
            return None, f"no model at {path}"
        try:
            from autonomous_kart.nodes.perception.detector import KartDetector

            detector = KartDetector(path, conf=p["confidence"], iou=p["nms_iou"], threads=p["cpu_threads"])
        except ImportError as e:
            return None, f"onnxruntime not installed ({e})"
        self.get_logger().info(f"kart detector {path} on {detector.providers[0]}")
        return detector, ""

    def on_image(self, msg: Image):
        xy, cov = np.zeros((0, 2)), np.zeros((0, 2, 2))
        img = image_to_bgr(msg)
        if self.detector is None or img is None:
            reason = self.why_no_detector or f"unsupported encoding {msg.encoding}"
            self.get_logger().warn(f"camera detection off: {reason}", throttle_duration_sec=FALLBACK_WARN_PERIOD_S)
        else:
            xy, cov, _ = camera_objects(
                self.detector, self.camera, self.ranging, self.roi_frac, img, self.camera_max_range_m
            )
        out = to_measurements(msg.header, xy, cov, ObjectMeasurement.CLASS_KART, [self.footprint_m] * len(xy))
        self.measurements_pub.publish(out)

    def on_scan(self, msg: LaserScan):
        objs = scan_objects(msg.ranges, msg.angle_min, msg.angle_increment, self.lidar)
        out = to_measurements(msg.header, objs.xy_m, objs.cov_m2, ObjectMeasurement.CLASS_UNKNOWN, objs.extent_m)
        self.measurements_pub.publish(out)


def main(args=None):
    run_node(PerceptionNode, args)


if __name__ == "__main__":
    main()
