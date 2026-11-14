"""Checks the numpy GRU against torch.nn.GRU and its output guards.

https://github.com/EVC-Purdue/A27/issues/28
"""

import io
import math

import numpy as np
import pytest

from autonomous_kart.nodes.localization.gru import GruCorrection


def random_npz(rng, hidden=8):
    w = {
        "w_ih": rng.normal(size=(3 * hidden, 4)),
        "w_hh": rng.normal(size=(3 * hidden, hidden)),
        "b_ih": rng.normal(size=3 * hidden),
        "b_hh": rng.normal(size=3 * hidden),
        "w_head": rng.normal(size=(2, hidden)),
        "b_head": rng.normal(size=2),
        "mu": rng.normal(size=4),
        "sd": rng.uniform(0.5, 2.0, size=4),
    }
    buf = io.BytesIO()
    np.savez(buf, **w)
    return w, buf.getvalue()


def unguarded():
    return GruCorrection(gap_s=0.1, clamp_mps=math.inf, min_speed_mps=0.0, max_gyro_rad_s=math.inf)


def test_matches_torch_gru():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    w, npz = random_npz(rng)
    gru = torch.nn.GRU(4, 8, batch_first=True).double()
    head = torch.nn.Linear(8, 2).double()
    gru.load_state_dict(
        {k: torch.from_numpy(w[n]) for k, n in [("weight_ih_l0", "w_ih"), ("weight_hh_l0", "w_hh")]}
        | {k: torch.from_numpy(w[n]) for k, n in [("bias_ih_l0", "b_ih"), ("bias_hh_l0", "b_hh")]}
    )
    head.load_state_dict({"weight": torch.from_numpy(w["w_head"]), "bias": torch.from_numpy(w["b_head"])})

    t = 0.02 * np.arange(50)
    wheel, gyro, steering = rng.normal(size=(3, 50))
    accel = np.concatenate([[0.0], np.diff(wheel) / 0.02])
    x = (np.stack([wheel, gyro, accel, steering], axis=1) - w["mu"]) / w["sd"]
    expected = head(gru(torch.from_numpy(x)[None])[0])[0].detach().numpy()

    model = unguarded()
    model.load(npz)
    got = np.array([model.step(t[i], wheel[i], gyro[i], steering[i]) for i in range(50)])
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)


def test_zero_without_weights():
    assert not unguarded().step(0.0, 5.0, 0.1, 0.0).any()


def test_guards_zero_the_output():
    _, npz = random_npz(np.random.default_rng(1))
    gru = GruCorrection(gap_s=0.1, clamp_mps=0.5, min_speed_mps=1.0, max_gyro_rad_s=2.0)
    gru.load(npz)
    assert not gru.step(0.00, 0.5, 0.1, 0.0).any()
    assert not gru.step(0.02, 5.0, 3.0, 0.0).any()
    dv = gru.step(0.04, 5.0, 0.1, 0.0)
    assert np.all(np.abs(dv) <= 0.5)


def test_gap_resets_hidden_state():
    _, npz = random_npz(np.random.default_rng(2))
    a, b = unguarded(), unguarded()
    a.load(npz)
    b.load(npz)
    for i in range(10):
        a.step(0.02 * i, 3.0 + i, 0.2, 0.1)
    np.testing.assert_array_equal(a.step(5.0, 4.0, 0.1, 0.0), b.step(5.0, 4.0, 0.1, 0.0))
