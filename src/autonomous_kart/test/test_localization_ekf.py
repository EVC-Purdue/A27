"""Checks the localization EKF predict, gated updates and late-measurement replay.

https://github.com/EVC-Purdue/A27/issues/28
"""

import math

import numpy as np
import pytest

from autonomous_kart.nodes.localization.ekf import LocalizationEKF

GATE = 13.8


def make_ekf(yaw=0.0, v=0.0):
    ekf = LocalizationEKF(pos_noise=0.01, yaw_drift_noise=0.01, speed_drift_noise=0.5, history_s=0.5)
    ekf.reset(0.0, 0.0, yaw, v, np.diag([0.01, 0.01, 0.01, 0.1]), stamp=0.0)
    return ekf


def test_predict_straight_line():
    ekf = make_ekf(yaw=math.pi / 2, v=2.0)
    for _ in range(100):
        ekf.predict(0.01, 0.0, 0.0, 1e-4, 1e-2)
    assert ekf.x == pytest.approx([0.0, 2.0, math.pi / 2, 2.0], abs=1e-9)


def test_predict_closes_a_circle():
    v, omega, dt = 5.0, 0.5, 0.001
    ekf = make_ekf(v=v)
    for _ in range(round(2 * math.pi / omega / dt)):
        ekf.predict(dt, omega, 0.0, 1e-4, 1e-2)
    assert ekf.x[:2] == pytest.approx([0.0, 0.0], abs=0.05)
    assert abs(ekf.x[2]) < 1e-3


def test_predict_applies_body_frame_correction():
    ekf = make_ekf(yaw=math.pi / 2, v=1.0)
    ekf.predict(1.0, 0.0, 0.0, 1e-4, 1e-2, dv=(0.5, 0.2))
    assert ekf.x[:2] == pytest.approx([-0.2, 1.5])


def test_predict_grows_covariance():
    ekf = make_ekf(v=3.0)
    before = ekf.P.copy()
    ekf.predict(0.1, 0.0, 0.0, 1e-4, 1e-2)
    assert np.all(np.diag(ekf.P) > np.diag(before))


def test_updates_move_state_toward_measurement():
    ekf = make_ekf()
    assert ekf.update_gps_xy(0.2, -0.1, np.eye(2) * 0.01, GATE)
    assert 0.0 < ekf.x[0] < 0.2 and -0.1 < ekf.x[1] < 0.0
    assert ekf.update_heading(0.1, 0.01, GATE)
    assert 0.0 < ekf.x[2] < 0.1
    assert ekf.update_speed(0.5, 0.1, GATE)
    assert 0.0 < ekf.x[3] < 0.5
    assert np.trace(ekf.P) < 0.13


def test_heading_innovation_wraps():
    ekf = make_ekf(yaw=math.pi - 0.05)
    assert ekf.update_heading(-math.pi + 0.05, 0.01, GATE)
    assert abs(ekf.x[2]) > math.pi - 0.05


def test_gate_rejects_outlier_and_leaves_state():
    ekf = make_ekf()
    x, P = ekf.x.copy(), ekf.P.copy()
    assert not ekf.update_gps_xy(5.0, 5.0, np.eye(2) * 0.01, GATE)
    assert np.array_equal(ekf.x, x) and np.array_equal(ekf.P, P)


def drive(ekf, stamps):
    for t in stamps:
        ekf.advance(t, 0.3, 0.5, 1e-4, 1e-2)


def test_late_gps_matches_in_order_gps():
    stamps = [round(0.01 * k, 2) for k in range(1, 41)]
    t_gps, t_wheel = stamps[14], stamps[24]
    R = np.eye(2) * 1e-3

    in_order = make_ekf(v=2.0)
    for t in stamps:
        in_order.advance(t, 0.3, 0.5, 1e-4, 1e-2)
        if t == t_gps:
            in_order.update_at(t, in_order.update_gps_xy, 0.35, 0.02, R, GATE)
        if t == t_wheel:
            in_order.update_at(t, in_order.update_speed, 2.3, 1e-2, GATE)

    late = make_ekf(v=2.0)
    for t in stamps:
        late.advance(t, 0.3, 0.5, 1e-4, 1e-2)
        if t == t_wheel:
            late.update_at(t, late.update_speed, 2.3, 1e-2, GATE)
    assert late.update_at(t_gps, late.update_gps_xy, 0.35, 0.02, R, GATE)

    assert late.x == pytest.approx(in_order.x, abs=1e-12)
    assert late.P == pytest.approx(in_order.P, abs=1e-12)


def test_gps_older_than_history_is_dropped():
    ekf = make_ekf(v=2.0)
    drive(ekf, [round(0.01 * k, 2) for k in range(1, 101)])
    assert ekf.history[0].stamp > 0.4
    x = ekf.x.copy()
    assert not ekf.update_at(0.1, ekf.update_gps_xy, x[0] + 0.01, x[1], np.eye(2) * 1e-3, GATE)
    assert ekf.x == pytest.approx(x)
