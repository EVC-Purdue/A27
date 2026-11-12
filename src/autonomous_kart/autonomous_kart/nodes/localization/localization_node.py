"""Localization node: EKF over [px, py, yaw, v] from IMU, GPS and wheel speed, publishing odom and TF.

https://github.com/EVC-Purdue/A27/issues/28
https://github.com/EVC-Purdue/A27/issues/29
"""

import glob
import math
import multiprocessing
import os
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Imu
from std_msgs.msg import String
from tf2_ros import TransformBroadcaster

from autonomous_kart.common import QOS_HIGH, QOS_LOW, QOS_STATE, run_node
from autonomous_kart.nodes.localization.ekf import LocalizationEKF
from autonomous_kart.nodes.localization.gru import GruCorrection
from kart_interfaces.msg import GpsFix, KartTelemetry, ObjectMeasurementArray, TrackedObjectArray


def stamp_s(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


class LocalizationNode(Node):
    def __init__(self):
        super().__init__("localization_node", automatically_declare_parameters_from_overrides=True)
        p = self.param
        self.mode = p("mode")
        self.ekf = LocalizationEKF(p("pos_noise"), p("yaw_drift_noise"), p("speed_drift_noise"), p("history_s"))
        self.gru = GruCorrection(p("gru_gap_s"), p("gru_clamp_mps"), p("gru_min_speed_mps"), p("gru_max_gyro_rad_s"))
        self.sigma_xy_floor = {
            GpsFix.FIX_RTK_FIXED: p("sigma_xy_floor_rtk_fixed_m"),
            GpsFix.FIX_RTK_FLOAT: p("sigma_xy_floor_rtk_float_m"),
        }

        # Latest (stamp, gyro z, accel x, gyro var, accel var) per IMU topic
        self.imus = {}
        self.omega_z = 0.0
        self.dv = (0.0, 0.0)
        self.last_imu_t = None
        self.last_gps_t = None
        self.rejects = 0
        self.first_reject_t = None

        self.create_subscription(GpsFix, "gps", self.on_gps, QOS_LOW)
        for topic in ("imu/front", "imu/back"):
            self.create_subscription(Imu, topic, lambda msg, topic=topic: self.on_imu(topic, msg), QOS_HIGH)
        self.create_subscription(KartTelemetry, "e_comms/telemetry", self.on_telemetry, QOS_HIGH)
        self.create_subscription(PoseWithCovarianceStamped, "slam/pose", self.on_unused, QOS_LOW)
        self.create_subscription(ObjectMeasurementArray, "perception/measurements", self.on_unused, QOS_LOW)
        self.odom_pub = self.create_publisher(Odometry, "odom", QOS_HIGH)
        self.tracks_pub = self.create_publisher(TrackedObjectArray, "localization/tracks", QOS_HIGH)
        self.transform_broadcaster = TransformBroadcaster(self)
        self.create_timer(1.0 / p("publish_rate_hz"), self.tick)
        self.create_timer(1.0, self.watchdog)
        if self.mode == "gru":
            self.start_training()
        self.get_logger().info(f"Localization node started in {self.mode} mode")

    def start_training(self):
        # torch is only needed in gru mode
        from autonomous_kart.nodes.localization.gru_training import train

        self.train = train
        self.weights = None
        self.weights_dir = os.path.expanduser(self.param("train_weights_dir"))
        self.buffers = {"imu": deque(), "telemetry": deque(), "gps": deque()}
        self.state = None
        self.training = None
        # A separate process keeps training off the GIL the filter callbacks need
        self.trainer = ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn"))
        self.create_subscription(String, "system_state", self.on_state, QOS_STATE)
        self.create_timer(self.param("train_cycle_s"), self.train_cycle)
        self.create_timer(1.0, self.collect_training)
        saved = sorted(glob.glob(os.path.join(self.weights_dir, "gru_*.npz")))
        if saved:
            with open(saved[-1], "rb") as f:
                self.load_weights(f.read())
            self.get_logger().info(f"GRU weights loaded from {saved[-1]}")

    def param(self, name):
        return self.get_parameter(name).value

    def now_s(self) -> float:
        return stamp_s(self.get_clock().now().to_msg())

    def on_unused(self, msg):
        pass

    def on_state(self, msg):
        self.state = msg.data

    def load_weights(self, npz: bytes):
        self.gru.load(npz)
        self.weights = npz

    def train_cycle(self):
        oldest = self.now_s() - self.param("train_buffer_s")
        for buf in self.buffers.values():
            while buf and buf[0][0] < oldest:
                buf.popleft()
        if self.state not in self.param("train_allowed_states") or self.training is not None:
            return
        params = {k: v.value for k, v in self.get_parameters_by_prefix("").items() if k.startswith(("train_", "gru_"))}
        snapshot = [list(buf) for buf in self.buffers.values()]
        self.training = self.trainer.submit(self.train, *snapshot, self.weights, params)

    def collect_training(self):
        if self.training is None or not self.training.done():
            return
        try:
            npz, log = self.training.result()
        except Exception as e:
            npz, log = None, f"GRU training failed: {type(e).__name__}: {e}"
        self.training = None
        self.get_logger().info(log)
        if npz is None:
            return
        self.load_weights(npz)
        os.makedirs(self.weights_dir, exist_ok=True)
        path = os.path.join(self.weights_dir, f"gru_{int(time.time())}.npz")
        with open(path + ".tmp", "wb") as f:
            f.write(npz)
        os.replace(path + ".tmp", path)

    def on_imu(self, topic, msg):
        stamp = stamp_s(msg.header.stamp)
        self.last_imu_t = self.now_s()
        self.imus[topic] = (
            stamp,
            msg.angular_velocity.z,
            msg.linear_acceleration.x,
            msg.angular_velocity_covariance[8],
            msg.linear_acceleration_covariance[0],
        )
        fresh = [s[1:] for s in self.imus.values() if stamp - s[0] <= self.param("imu_stale_s")]
        omega_z, accel_x, omega_var, accel_var = np.mean(fresh, axis=0)
        self.omega_z = float(omega_z)
        if self.mode == "gru":
            self.buffers["imu"].append((stamp, msg.angular_velocity.z))
        if not self.ekf.initialized:
            return

        dt = stamp - self.ekf.stamp
        if dt <= 0.0:
            return
        if dt > self.param("imu_max_dt_s"):
            # Skip the gap instead of integrating across it
            self.get_logger().warning(f"IMU dt out of range: {dt:.3f}s", throttle_duration_sec=2.0)
            self.ekf.stamp = stamp
            return
        self.ekf.advance(stamp, self.omega_z, float(accel_x), float(omega_var), float(accel_var), self.dv)

    def on_telemetry(self, msg):
        if not math.isfinite(msg.wheel_speed):
            return
        stamp = stamp_s(msg.header.stamp)
        if self.mode == "gru" and math.isfinite(msg.steering_angle):
            gain = min(max(self.param("gru_correction_gain"), 0.0), 1.0)
            dv = gain * self.gru.step(stamp, msg.wheel_speed, self.omega_z, msg.steering_angle)
            self.dv = (float(dv[0]), float(dv[1]))
            self.buffers["telemetry"].append((stamp, msg.wheel_speed, msg.steering_angle))
        if not self.ekf.initialized:
            return
        wheel_var, gate = self.param("wheel_speed_var"), self.param("gate_speed")
        if not self.ekf.update_at(stamp, self.ekf.update_speed, msg.wheel_speed, wheel_var, gate):
            self.get_logger().warning("Wheel speed rejected by gate", throttle_duration_sec=2.0)

    def on_gps(self, msg):
        if msg.fix_quality == GpsFix.FIX_NONE:
            return
        cov = msg.position_covariance
        values = (msg.x, msg.y, msg.heading, msg.speed, *cov, msg.heading_variance, msg.speed_variance)
        if not all(map(math.isfinite, values)) or min(cov[0], cov[3], msg.heading_variance, msg.speed_variance) < 0:
            return
        stamp = stamp_s(msg.header.stamp)
        self.last_gps_t = self.now_s()
        if self.mode == "gru":
            self.buffers["gps"].append((stamp, msg.x, msg.y, msg.heading, msg.fix_quality, msg.heading_valid))

        floor = self.sigma_xy_floor.get(msg.fix_quality, self.param("sigma_xy_floor_single_m")) ** 2
        R_xy = np.array(cov, dtype=float).reshape(2, 2)
        R_xy[0, 0] = max(R_xy[0, 0], floor)
        R_xy[1, 1] = max(R_xy[1, 1], floor)
        have_speed = msg.speed > self.param("gps_speed_min_mps")

        if not self.ekf.initialized:
            self.init_from_fix(msg, stamp, R_xy, have_speed)
            return
        if self.ekf.history and stamp < self.ekf.history[0].stamp:
            self.get_logger().warning(f"GPS fix {self.ekf.stamp - stamp:.2f}s old, dropped", throttle_duration_sec=2.0)
            return

        if self.ekf.update_at(stamp, self.ekf.update_gps_xy, msg.x, msg.y, R_xy, self.param("gate_xy")):
            self.rejects = 0
        elif msg.fix_quality in (GpsFix.FIX_RTK_FIXED, GpsFix.FIX_RTK_FLOAT):
            # Repeated confident fixes that disagree mean the filter has diverged, not the GPS
            self.get_logger().warning("GPS fix rejected by gate", throttle_duration_sec=2.0)
            if self.rejects == 0:
                self.first_reject_t = stamp
            self.rejects += 1
            if self.rejects >= self.param("reinit_rejects") and stamp - self.first_reject_t >= self.param("reinit_s"):
                self.init_from_fix(msg, stamp, R_xy, have_speed)
                return

        if msg.heading_valid:
            self.ekf.update_at(
                stamp, self.ekf.update_heading, msg.heading, msg.heading_variance, self.param("gate_heading")
            )
        if have_speed:
            self.ekf.update_at(stamp, self.ekf.update_speed, msg.speed, msg.speed_variance, self.param("gate_speed"))

    def init_from_fix(self, msg, stamp, R_xy, have_speed):
        if self.last_imu_t is None:
            return
        if not msg.heading_valid:
            # Seeding yaw at 0 points the kart east and nothing observes yaw at rest
            self.get_logger().warning("EKF waiting to init: no heading solution yet", throttle_duration_sec=5.0)
            return
        v0 = msg.speed if have_speed else 0.0
        var_v0 = msg.speed_variance if have_speed else self.param("init_speed_var")
        P0 = np.diag([R_xy[0, 0], R_xy[1, 1], msg.heading_variance, var_v0])
        self.ekf.reset(msg.x, msg.y, msg.heading, v0, P0, stamp)
        self.rejects = 0
        self.get_logger().info(
            f"EKF init: pos=({msg.x:.2f},{msg.y:.2f}) yaw={math.degrees(msg.heading):.1f} deg v={v0:.2f} m/s"
        )

    def watchdog(self):
        now = self.now_s()
        if self.last_imu_t is not None and now - self.last_imu_t > self.param("imu_dropout_warn_s"):
            self.get_logger().warning(f"IMU dropout: {now - self.last_imu_t:.1f}s since last sample")
        if self.last_gps_t is not None and now - self.last_gps_t > self.param("gps_dropout_warn_s"):
            self.get_logger().warning(f"GPS dropout: {now - self.last_gps_t:.1f}s since last fix")

    def tick(self):
        now = self.get_clock().now().to_msg()
        tracks = TrackedObjectArray()
        tracks.header.stamp = now
        tracks.header.frame_id = "odom"
        self.tracks_pub.publish(tracks)
        if not self.ekf.initialized:
            return

        px, py, yaw, v = self.ekf.x
        P = self.ekf.P
        odom = Odometry()
        # Stamp with the state's time so consumers can account for its age
        odom.header.stamp = Time(seconds=self.ekf.stamp).to_msg()
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"
        odom.pose.pose.position.x = float(px)
        odom.pose.pose.position.y = float(py)
        odom.pose.pose.orientation.z = math.sin(yaw / 2.0)
        odom.pose.pose.orientation.w = math.cos(yaw / 2.0)
        # Row-major 6x6 over [x, y, z, roll, pitch, yaw]; z, roll and pitch are unobserved
        cov = [0.0] * 36
        cov[0], cov[1], cov[5] = P[0, 0], P[0, 1], P[0, 2]
        cov[6], cov[7], cov[11] = P[1, 0], P[1, 1], P[1, 2]
        cov[30], cov[31], cov[35] = P[2, 0], P[2, 1], P[2, 2]
        odom.pose.covariance = [float(c) for c in cov]
        odom.twist.twist.linear.x = float(v) + self.dv[0]
        odom.twist.twist.linear.y = self.dv[1]
        odom.twist.twist.angular.z = self.omega_z
        twist_cov = [0.0] * 36
        twist_cov[0] = float(P[3, 3])
        odom.twist.covariance = twist_cov
        self.odom_pub.publish(odom)

        transform = TransformStamped()
        transform.header = odom.header
        transform.child_frame_id = odom.child_frame_id
        transform.transform.translation.x = odom.pose.pose.position.x
        transform.transform.translation.y = odom.pose.pose.position.y
        transform.transform.rotation = odom.pose.pose.orientation
        self.transform_broadcaster.sendTransform(transform)


def main(args=None):
    run_node(LocalizationNode, args)


if __name__ == "__main__":
    main()
