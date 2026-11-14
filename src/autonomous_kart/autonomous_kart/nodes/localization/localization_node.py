"""Localization node: EKF over [px, py, yaw, v] from IMU, GPS and wheel speed, publishing odom and TF.

https://github.com/EVC-Purdue/A27/issues/28
https://github.com/EVC-Purdue/A27/issues/29
"""

import math

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
        self.mode = self.param("mode")
        self.ekf = LocalizationEKF(
            self.param("pos_noise"),
            self.param("yaw_drift_noise"),
            self.param("speed_drift_noise"),
            self.param("history_s"),
        )
        self.gru = GruCorrection(
            self.param("gru_gap_s"),
            self.param("gru_clamp_mps"),
            self.param("gru_min_speed_mps"),
            self.param("gru_max_gyro_rad_s"),
        )
        self.trainer = None
        self.system_state = None

        self.imus = {}  # latest Imu message per topic
        self.omega_z = 0.0  # mean gyro z of the fresh IMUs, rad/s
        self.dv = (0.0, 0.0)  # GRU body-frame velocity correction, m/s
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
        self.create_timer(1.0 / self.param("publish_rate_hz"), self.tick)
        self.create_timer(1.0, self.watchdog)

        if self.mode == "gru":
            self.start_trainer()
        self.get_logger().info(f"Localization node started in {self.mode} mode")

    def param(self, name):
        return self.get_parameter(name).value

    def now_s(self) -> float:
        return stamp_s(self.get_clock().now().to_msg())

    def on_unused(self, msg):
        pass

    # IMU and wheel speed

    def on_imu(self, topic, msg):
        stamp = stamp_s(msg.header.stamp)
        self.last_imu_t = self.now_s()
        self.imus[topic] = msg
        fresh = [m for m in self.imus.values() if stamp - stamp_s(m.header.stamp) <= self.param("imu_stale_s")]
        self.omega_z = float(np.mean([m.angular_velocity.z for m in fresh]))
        accel_x = float(np.mean([m.linear_acceleration.x for m in fresh]))
        omega_var = float(np.mean([m.angular_velocity_covariance[8] for m in fresh]))
        accel_var = float(np.mean([m.linear_acceleration_covariance[0] for m in fresh]))
        if self.trainer:
            self.trainer.imu.append((stamp, msg.angular_velocity.z))
        if not self.ekf.initialized:
            return

        dt = stamp - self.ekf.stamp
        if dt <= 0.0:
            return
        if dt > self.param("imu_max_dt_s"):
            self.get_logger().warning(f"IMU gap of {dt:.3f}s skipped", throttle_duration_sec=2.0)
            self.ekf.skip_to(stamp)
            return
        self.ekf.advance(stamp, self.omega_z, accel_x, omega_var, accel_var, self.dv)

    def on_telemetry(self, msg):
        if not math.isfinite(msg.wheel_speed):
            return
        stamp = stamp_s(msg.header.stamp)
        if self.trainer and math.isfinite(msg.steering_angle):
            gain = min(max(self.param("gru_correction_gain"), 0.0), 1.0)
            dvx, dvy = gain * self.gru.step(stamp, msg.wheel_speed, self.omega_z, msg.steering_angle)
            self.dv = (float(dvx), float(dvy))
            self.trainer.telemetry.append((stamp, msg.wheel_speed, msg.steering_angle))
        if not self.ekf.initialized:
            return
        if not self.ekf.update_at(
            stamp, self.ekf.update_speed, msg.wheel_speed, self.param("wheel_speed_var"), self.param("gate_speed")
        ):
            self.get_logger().warning("Wheel speed rejected", throttle_duration_sec=2.0)

    # GPS

    def on_gps(self, msg):
        if not self.usable(msg):
            return
        stamp = stamp_s(msg.header.stamp)
        self.last_gps_t = self.now_s()
        if self.trainer:
            self.trainer.gps.append((stamp, msg.x, msg.y, msg.heading, msg.fix_quality, msg.heading_valid))

        if not self.ekf.initialized:
            self.init_from_fix(msg, stamp)
            return
        if self.ekf.too_old(stamp):
            self.get_logger().warning(f"GPS fix {self.ekf.stamp - stamp:.2f}s old, dropped", throttle_duration_sec=2.0)
            return

        R = self.position_covariance(msg)
        if self.ekf.update_at(stamp, self.ekf.update_gps_xy, msg.x, msg.y, R, self.param("gate_xy")):
            self.rejects = 0
        elif msg.fix_quality in (GpsFix.FIX_RTK_FIXED, GpsFix.FIX_RTK_FLOAT) and self.diverged(stamp):
            self.init_from_fix(msg, stamp)
            return
        if msg.heading_valid:
            self.ekf.update_at(
                stamp, self.ekf.update_heading, msg.heading, msg.heading_variance, self.param("gate_heading")
            )
        if msg.speed > self.param("gps_speed_min_mps"):
            self.ekf.update_at(stamp, self.ekf.update_speed, msg.speed, msg.speed_variance, self.param("gate_speed"))

    @staticmethod
    def usable(msg) -> bool:
        cov = msg.position_covariance
        values = (msg.x, msg.y, msg.heading, msg.speed, *cov, msg.heading_variance, msg.speed_variance)
        return (
            msg.fix_quality != GpsFix.FIX_NONE
            and all(map(math.isfinite, values))
            and min(cov[0], cov[3], msg.heading_variance, msg.speed_variance) >= 0
        )

    def position_covariance(self, msg) -> np.ndarray:
        """The fix's 2x2 covariance with each variance raised to the floor for its fix type."""
        floor_m = {
            GpsFix.FIX_RTK_FIXED: self.param("sigma_xy_floor_rtk_fixed_m"),
            GpsFix.FIX_RTK_FLOAT: self.param("sigma_xy_floor_rtk_float_m"),
        }.get(msg.fix_quality, self.param("sigma_xy_floor_single_m"))
        R = np.array(msg.position_covariance, dtype=float).reshape(2, 2)
        R[0, 0] = max(R[0, 0], floor_m**2)
        R[1, 1] = max(R[1, 1], floor_m**2)
        return R

    def diverged(self, stamp) -> bool:
        """Count a rejected RTK fix; True once enough of them span reinit_s."""
        # Repeated confident fixes that disagree mean the filter has diverged, not the GPS
        self.get_logger().warning("GPS fix rejected by gate", throttle_duration_sec=2.0)
        if self.rejects == 0:
            self.first_reject_t = stamp
        self.rejects += 1
        return self.rejects >= self.param("reinit_rejects") and stamp - self.first_reject_t >= self.param("reinit_s")

    def init_from_fix(self, msg, stamp):
        if self.last_imu_t is None:
            return
        if not msg.heading_valid:
            # Seeding yaw at 0 points the kart east and nothing observes yaw at rest
            self.get_logger().warning("EKF waiting to init: no heading solution yet", throttle_duration_sec=5.0)
            return
        R = self.position_covariance(msg)
        have_speed = msg.speed > self.param("gps_speed_min_mps")
        v0 = msg.speed if have_speed else 0.0
        var_v0 = msg.speed_variance if have_speed else self.param("init_speed_var")
        self.ekf.reset(msg.x, msg.y, msg.heading, v0, np.diag([R[0, 0], R[1, 1], msg.heading_variance, var_v0]), stamp)
        self.rejects = 0
        self.get_logger().info(
            f"EKF init: pos=({msg.x:.2f},{msg.y:.2f}) yaw={math.degrees(msg.heading):.1f} deg v={v0:.2f} m/s"
        )

    # GRU training

    def start_trainer(self):
        # torch is only needed in gru mode
        from autonomous_kart.nodes.localization.gru_training import GruTrainer

        self.trainer = GruTrainer(self.param("train_weights_dir"), self.param("train_buffer_s"))
        saved = self.trainer.load_latest()
        if saved:
            self.gru.load(saved[0])
            self.get_logger().info(f"GRU weights loaded from {saved[1]}")
        self.create_subscription(String, "system_state", self.on_system_state, QOS_STATE)
        self.create_timer(self.param("train_cycle_s"), self.train_cycle)
        self.create_timer(1.0, self.collect_training)

    def on_system_state(self, msg):
        self.system_state = msg.data

    def train_cycle(self):
        if self.system_state not in self.param("train_allowed_states"):
            return
        params = {k: v.value for k, v in self.get_parameters_by_prefix("").items() if k.startswith(("train_", "gru_"))}
        self.trainer.start(self.now_s(), params)

    def collect_training(self):
        result = self.trainer.finished()
        if result is None:
            return
        npz, log = result
        self.get_logger().info(log)
        if npz is not None:
            self.gru.load(npz)

    # Output

    def watchdog(self):
        now = self.now_s()
        if self.last_imu_t is not None and now - self.last_imu_t > self.param("imu_dropout_warn_s"):
            self.get_logger().warning(f"IMU dropout: {now - self.last_imu_t:.1f}s since last sample")
        if self.last_gps_t is not None and now - self.last_gps_t > self.param("gps_dropout_warn_s"):
            self.get_logger().warning(f"GPS dropout: {now - self.last_gps_t:.1f}s since last fix")

    def tick(self):
        tracks = TrackedObjectArray()
        tracks.header.stamp = self.get_clock().now().to_msg()
        tracks.header.frame_id = "odom"
        self.tracks_pub.publish(tracks)
        if not self.ekf.initialized:
            return
        odom = self.odometry()
        self.odom_pub.publish(odom)

        transform = TransformStamped()
        transform.header = odom.header
        transform.child_frame_id = odom.child_frame_id
        transform.transform.translation.x = odom.pose.pose.position.x
        transform.transform.translation.y = odom.pose.pose.position.y
        transform.transform.rotation = odom.pose.pose.orientation
        self.transform_broadcaster.sendTransform(transform)

    def odometry(self) -> Odometry:
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
        pose_cov = np.zeros((6, 6))
        pose_cov[np.ix_([0, 1, 5], [0, 1, 5])] = P[:3, :3]
        odom.pose.covariance = pose_cov.ravel().tolist()
        odom.twist.twist.linear.x = float(v) + self.dv[0]
        odom.twist.twist.linear.y = self.dv[1]
        odom.twist.twist.angular.z = self.omega_z
        twist_cov = np.zeros(36)
        twist_cov[0] = P[3, 3]
        odom.twist.covariance = twist_cov.tolist()
        return odom


def main(args=None):
    run_node(LocalizationNode, args)


if __name__ == "__main__":
    main()
