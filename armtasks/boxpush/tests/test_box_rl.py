"""! @file test_box_rl.py
@brief 上位方策の口（BoxActionSpec / 方策 / 適用側）と学習環境 BoxPushEnv を固定する（短いエピソードだけ回す）。"""

from __future__ import annotations

import json

import numpy as np
import pytest

import boxpush  # noqa: F401
from boxpush import PKG_ROOT
from boxpush.cli.run_box import load_box_experiment
from boxpush.env import BOX_OBS_SIZE, BoxDomain, BoxPushEnv
from boxpush.policy import ACTION_NAMES, BoxActionSpec, BoxPushApplier, MlpBoxPolicy, ScheduleBoxPolicy, ZeroBoxPolicy

CFG = PKG_ROOT / "configs" / "experiments" / "box_p1_stand_push.yaml"


def test_action_spec_roundtrip_and_fixed():
    spec = BoxActionSpec(actions=("f_push", "lean_x"), f_push_max=80.0)
    vals = spec.to_values(np.array([0.0, 0.5]))
    assert vals["f_push"] == pytest.approx(40.0) and vals["lean_x"] == pytest.approx(0.10)   # scale 0.20
    assert vals["com_dz"] == spec.fixed["com_dz"] and vals["step_dx"] == 0.0      # 使わない行動は固定値
    a = spec.from_values({"f_push": 60.0, "lean_x": -0.02})
    assert a == pytest.approx([0.5, -0.1])
    with pytest.raises(ValueError):
        BoxActionSpec(actions=("f_pussh",))
    assert set(ACTION_NAMES) >= set(BoxActionSpec().scales)


def test_schedule_policy_ramps():
    spec = BoxActionSpec(actions=tuple(a for a in ACTION_NAMES if a != "cmp_dx"))
    pol = ScheduleBoxPolicy(spec, targets={"f_push": 40.0, "lean_x": 0.04, "com_dz": -0.1}, t_start=2.0, t_ramp=1.0, t_lean_lead=1.5)
    v0 = pol.values_at(0.0); v1 = pol.values_at(1.0); v2 = pol.values_at(2.5); v3 = pol.values_at(5.0)
    assert v0["f_push"] == 0.0 and v0["lean_x"] == 0.0
    assert v1["lean_x"] == pytest.approx(0.02) and v1["f_push"] == 0.0 and v1["com_dz"] == pytest.approx(-0.05)
    assert v2["f_push"] == pytest.approx(20.0) and v2["lean_x"] == pytest.approx(0.04)
    assert v3["f_push"] == pytest.approx(40.0)
    pol.set_time(5.0)
    assert spec.to_values(pol.act(np.zeros(1)))["f_push"] == pytest.approx(40.0, abs=1e-6)


def test_applier_rate_limit():
    spec = BoxActionSpec(actions=("f_push", "lean_x"))
    ap = BoxPushApplier(spec=spec, policy=ZeroBoxPolicy(spec), dt=0.01, ki=0.0)

    class _P:
        z_c = 0.6; z_c_nominal = 0.6; initialized = True
        def set_lean(self, dx, dy=0.0): self.lean = dx
        def set_com_height(self, z): self.z = z
        def rot2(self): return np.eye(2)

    class _S:
        box_pose = np.array([0.71, 0, 0.5, 1, 0, 0, 0, 0, 0, 0]); wrench_hands = np.zeros((2, 6)); c_W = np.array([0, 0, 0.6])

    pl, st = _P(), _S()
    ap.apply_values(pl, st, {**spec.fixed, "f_push": 40.0, "lean_x": 0.10})
    assert pl.lean == pytest.approx(0.2 * 0.01)                   # 0.2 m/s × dt
    assert ap.hand_task.f_ff[0, 0] == pytest.approx(0.5 * 100.0 * 0.01)   # 100 N/s × dt の半分（片手）
    for _ in range(200):
        ap.apply_values(pl, st, {**spec.fixed, "f_push": 40.0, "lean_x": 0.10})
    assert pl.lean == pytest.approx(0.10) and ap.hand_task.f_ff[0, 0] == pytest.approx(20.0)
    assert tuple(ap.hand_task.kp_axis[0]) == (0.0, 1.0, 1.0)      # 押す向きは力制御


def test_mlp_policy_reads_npz(tmp_path):
    rng = np.random.default_rng(0)
    W0, b0 = rng.standard_normal((BOX_OBS_SIZE, 8)) * 0.1, np.zeros(8)
    W1, b1 = rng.standard_normal((8, 2)) * 0.1, np.zeros(2)
    np.savez(tmp_path / "params.npz", W0=W0, b0=b0, W1=W1, b1=b1)
    (tmp_path / "meta.json").write_text(json.dumps({"actions": ["f_push", "lean_x"], "obs_mean": [0.0] * BOX_OBS_SIZE,
                                                     "obs_std": [1.0] * BOX_OBS_SIZE}), encoding="utf-8")
    pol = MlpBoxPolicy(tmp_path)
    a = pol.act(np.zeros(BOX_OBS_SIZE))
    assert a.shape == (2,) and np.all(np.abs(a) <= 1.0)
    with pytest.raises(ValueError):
        MlpBoxPolicy(tmp_path, BoxActionSpec(actions=("f_push",)))


@pytest.mark.slow
def test_env_reset_step_shapes_and_randomization():
    cfg, push = load_box_experiment(CFG)
    cfg.output.verbose = False
    spec = BoxActionSpec(actions=("f_push", "lean_x"))
    env = BoxPushEnv(cfg, spec=spec, domain=BoxDomain(mass=(10.0, 40.0), friction=(0.15, 0.35)), max_time=1.0, seed=1)
    obs = env.reset()
    assert obs.shape == (BOX_OBS_SIZE,) and np.all(np.isfinite(obs))
    info0 = dict(env.episode_info)
    assert 10.0 <= info0["box_mass"] <= 40.0 and 0.15 <= info0["box_friction"] <= 0.35
    total = 0.0
    for _ in range(3):
        obs, r, done, info = env.step(np.array([0.0, 0.0]))
        total += r
        assert obs.shape == (BOX_OBS_SIZE,) and np.isfinite(r)
    assert not info.get("fell") and info["t"] == pytest.approx(0.3 + env.dt_ctrl, abs=0.01)
    env.reset()
    assert env.episode_info["box_mass"] != info0["box_mass"]       # 乱択が効く


@pytest.mark.slow
def test_env_schedule_reproduces_stand_push():
    """環境経由でも手書きスケジュールは箱を動かす（box_p1_stand_push と同じ条件、4 s）。"""
    cfg, push = load_box_experiment(CFG)
    cfg.output.verbose = False
    spec = BoxActionSpec(actions=("f_push", "lean_x"))
    spec.fixed.update({"f_push": push.f_push, "lean_x": push.lean_x})
    env = BoxPushEnv(cfg, spec=spec, domain=BoxDomain(randomize=False), max_time=5.0, seed=0, vx=0.0)
    pol = ScheduleBoxPolicy(spec, t_start=push.t_start, t_ramp=push.t_ramp, t_lean_lead=push.t_lean_lead)
    obs = env.reset(); done = False
    while not done:
        pol.set_time(env.k * env.dt_ctrl)
        obs, r, done, info = env.step(pol.act(obs))
    assert not info.get("fell") and info["f_meas"] > 30.0 and info["box_x"] > 0.71 + 0.01


def test_schedule_lean_pre_max_splits_lead_and_push():
    """前傾は lean_pre_max までを先行し、残りは押付け力と一緒にランプする。"""
    spec = BoxActionSpec(actions=tuple(a for a in ACTION_NAMES if a != "cmp_dx"))
    pol = ScheduleBoxPolicy(spec, targets={"f_push": 60.0, "lean_x": 0.13}, t_start=2.0, t_ramp=1.0,
                            t_lean_lead=1.5, lean_pre_max=0.05)
    assert pol.values_at(1.5)["lean_x"] == pytest.approx(0.05)          # 先行分だけ
    assert pol.values_at(2.5)["lean_x"] == pytest.approx(0.05 + 0.5 * 0.08)
    assert pol.values_at(3.5)["lean_x"] == pytest.approx(0.13)


def test_applier_reach_limit_cuts_push_force():
    """手先が骨盤から reach_max を超えると押付け力を絞り、引き戻す。"""
    spec = BoxActionSpec(actions=("f_push", "lean_x"))
    ap = BoxPushApplier(spec=spec, policy=ZeroBoxPolicy(spec), dt=0.01, ki=0.0, rate_limits={})

    class _P:
        z_c = 0.6; z_c_nominal = 0.6; initialized = True
        def set_lean(self, dx, dy=0.0): pass
        def set_com_height(self, z): pass
        def rot2(self): return np.eye(2)

    def state(hand_x):
        class _S:
            box_pose = np.array([1.0, 0, 0.5, 1, 0, 0, 0, 0, 0, 0]); wrench_hands = np.zeros((2, 6))
            c_W = np.array([0, 0, 0.6]); p_WB = np.array([0.0, 0.0, 0.76])
            p_hands = np.array([[hand_x, 0.19, 0.77], [hand_x, -0.19, 0.77]])
        return _S()

    vals = {**spec.fixed, "f_push": 40.0}
    ap.apply_values(_P(), state(0.30), vals)
    assert ap.hand_task.f_ff[0, 0] == pytest.approx(20.0)              # リーチ内: 指令どおり
    ap.apply_values(_P(), state(0.45), vals)
    assert ap.hand_task.f_ff[0, 0] == pytest.approx(10.0)              # 帯の中間: 半分
    ap.apply_values(_P(), state(0.49), vals)
    assert ap.hand_task.f_ff[0, 0] == pytest.approx(-600.0 * 0.02)     # 超えた分は引き戻す


def test_eval_and_record_entrypoints_import():
    """評価・動画のスクリプトが import でき、方策を作れる（描画はしない）。"""
    from boxpush.cli import eval_policy, record  # noqa: F401
    from boxpush.cli.run_box import load_box_experiment

    cfg, push = load_box_experiment(PKG_ROOT / "configs" / "experiments" / "box_p2_walk_push.yaml")
    for name in ("upright", "schedule", "reactive"):
        pol, spec = eval_policy._make_policy(name, None, push, 30.0, 0.2)
        vals = pol.values_at(5.0)
        assert vals["f_push"] == pytest.approx(1.15 * 0.2 * 30.0 * 9.81, rel=1e-6)
        if name == "upright":
            assert vals["lean_x"] == 0.0
        elif name == "schedule":
            assert vals["lean_x"] == pytest.approx(vals["f_push"] / (33.34 * 3.90 ** 2), rel=1e-6)


# ----------------------------------------------------------------------
# 学習の条件 A / B / C（腕の上位方策 + 安定化層の残差 + 全身制御の残差）
# ----------------------------------------------------------------------
def test_presets_actions_obs_and_rate():
    from boxpush.policy import PRESETS, box_obs_size

    a, b, bm, c = (BoxActionSpec.preset(n) for n in ("A", "B", "B_mom", "C"))
    assert a.actions == ("f_push", "hand_dz") and a.decimation == 50 and not a.uses_lean_action
    assert b.decimation == 5 and c.decimation == 1 and b.residual_actions == ("cop_dx", "cop_dy")
    assert c.n_act == 2 + 2 + 12 and c.uses_tau_leg and list(c.residual_index) == list(range(2, 16))
    assert [box_obs_size(s) for s in (a, b, bm, c)] == [40, 40 + 6 + 2, 40 + 6 + 4, 40 + 6 + 14 + 24]
    vals = c.to_values(np.r_[0.0, 0.0, 1.0, -1.0, np.full(12, 0.5)])
    assert vals["cop_dx"] == pytest.approx(0.03) and vals["cop_dy"] == pytest.approx(-0.03)
    assert vals["tau_leg0"] == pytest.approx(5.0) and vals["lean_x"] == a.fixed["lean_x"]
    with pytest.raises(ValueError):
        BoxActionSpec.preset("D")
    assert set(PRESETS) >= {"A", "B", "C"}


class _Planner:
    z_c = 0.6; z_c_nominal = 0.6; initialized = True; omega = 3.9
    def set_lean(self, dx, dy=0.0): self.lean = dx
    def set_com_height(self, z): self.z = z
    def rot2(self):
        c, s = np.cos(0.5), np.sin(0.5)
        return np.array([[c, -s], [s, c]])


class _State:
    box_pose = np.array([0.71, 0, 0.5, 1, 0, 0, 0, 0, 0, 0]); wrench_hands = np.zeros((2, 6)); c_W = np.array([0, 0, 0.6])


def test_analytic_lean_leads_force():
    """条件 A: 前傾は押付け力の指令から f / (m omega^2)。押す前は lean_pre を先に入れ、力は前傾で頭打ち。"""
    spec = BoxActionSpec.preset("A")
    ap = BoxPushApplier(spec=spec, policy=ZeroBoxPolicy(spec), dt=0.01, ki=0.0, t_walk=1.0, t_lean_lead=0.5)
    pl, st = _Planner(), _State()
    for _ in range(40):                                   # 0.4 s: まだ前傾の開始前
        ap.apply_values(pl, st, {**spec.fixed, "f_push": 0.0})
    assert pl.lean == 0.0
    for _ in range(60):                                   # 1.0 s: lean_pre まで
        ap.apply_values(pl, st, {**spec.fixed, "f_push": 0.0})
    assert pl.lean == pytest.approx(ap.lean_pre)
    f = 60.0
    ap.apply_values(pl, st, {**spec.fixed, "f_push": f})
    k = ap.m_robot * pl.omega ** 2
    assert ap.last_vals["f_push"] <= pl.lean * k + ap.f_pre + 1e-9        # 力は前傾で支えられる分 + f_pre まで
    for _ in range(300):
        ap.apply_values(pl, st, {**spec.fixed, "f_push": f})
    assert pl.lean == pytest.approx(f / k) and ap.last_vals["f_push"] == pytest.approx(f)


def test_residual_written_to_controller_in_world_frame():
    class _Ctrl:
        def set_external_residual(self, dcop=None, dmom=None, tau_leg=None):
            self.r = (dcop, dmom, tau_leg)

    spec = BoxActionSpec.preset("C")
    ctrl = _Ctrl()
    ap = BoxPushApplier(spec=spec, policy=ZeroBoxPolicy(spec), dt=0.01, ki=0.0, controller=ctrl)
    pl = _Planner()
    vals = spec.to_values(np.r_[0.0, 0.0, 1.0, 0.0, np.linspace(-1, 1, 12)])
    ap.apply_values(pl, _State(), vals)
    dcop, dmom, tau = ctrl.r
    assert np.allclose(dcop, pl.rot2() @ [0.03, 0.0]) and np.allclose(dmom, 0.0)
    assert np.allclose(tau, 10.0 * np.linspace(-1, 1, 12))


def test_mlp_policy_reads_preset_and_rate(tmp_path):
    from boxpush.policy import box_obs_size

    spec = BoxActionSpec.preset("B")
    n_obs = box_obs_size(spec)
    np.savez(tmp_path / "params.npz", W0=np.zeros((n_obs, 4)), b0=np.zeros(4))
    (tmp_path / "meta.json").write_text(json.dumps({"actions": list(spec.actions), "decimation": 10, "preset": "B",
                                                     "obs_mean": [0.0] * n_obs, "obs_std": [1.0] * n_obs}), encoding="utf-8")
    pol = MlpBoxPolicy(tmp_path)
    assert pol.spec.decimation == 10 and pol.spec.preset_name == "B" and pol.spec.residual_actions == ("cop_dx", "cop_dy")


@pytest.mark.slow
def test_env_zero_residual_matches_and_obs_shapes():
    """B と C は残差 0 なら同じ軌道（判断周期は 100 Hz と 500 Hz で違うが、腕の行動は 100 ms 保持で同じ）。観測の次元は条件どおり。"""
    from boxpush.policy import box_obs_size

    cfg, _ = load_box_experiment(PKG_ROOT / "configs" / "experiments" / "box_p2_walk_push.yaml")
    cfg.output.verbose = False
    out = {}
    for name in ("B", "C"):
        spec = BoxActionSpec.preset(name)
        env = BoxPushEnv(cfg, spec=spec, domain=BoxDomain(randomize=False), max_time=0.6, seed=0)
        obs = env.reset()
        assert obs.shape == (box_obs_size(spec),) == (env.obs_size,)
        a = np.zeros(spec.n_act); a[0] = -0.5
        done = False
        while not done:
            obs, r, done, info = env.step(a)
            assert obs.shape == (env.obs_size,) and np.isfinite(r)
        out[name] = (np.array(env.state.q), env.ctrl.ext_residual)
    assert np.array_equal(out["B"][0], out["C"][0]) and out["C"][1] is None


# ----------------------------------------------------------------------
# 計画層の残差（P / P_S / P_S_T）。姿勢の指示なし
# ----------------------------------------------------------------------
def test_preset_p_alias_and_reward():
    from boxpush.policy import PLAN_NAMES, PRESET_REWARD, box_obs_size

    p = BoxActionSpec.preset("P")
    assert p.actions == ("f_push", "hand_dz") + PLAN_NAMES and p.decimation == 50 and p.uses_lean_action
    assert box_obs_size(p) == 40 and box_obs_size(BoxActionSpec.preset("P_S")) == 40 + 6 + 2
    vals = p.to_values(np.array([0.0, 0.0, 0.5, -0.5, -1.0, 0.2, 0.0, 0.0]))
    assert vals["lean_x"] == pytest.approx(0.10) and vals["cmp_dy"] == pytest.approx(-0.05)   # cmp_dx は lean_x に書く
    assert vals["com_dz"] == pytest.approx(-0.15) and vals["step_dx"] == pytest.approx(0.02)
    a = p.from_values({"f_push": 60.0, "lean_x": 0.10})          # 手書き（lean_x）を cmp_dx に戻す
    assert a[2] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        BoxActionSpec(actions=("lean_x", "cmp_dx"))
    assert PRESET_REWARD["P"]["w_post"] == 0.0


def test_applier_writes_cmp_y_and_keeps_zero_default():
    spec = BoxActionSpec.preset("P")
    ap = BoxPushApplier(spec=spec, policy=ZeroBoxPolicy(spec), dt=0.01, ki=0.0, rate_limits={})

    class _P:
        z_c = 0.6; z_c_nominal = 0.6; initialized = True
        def set_lean(self, dx, dy=0.0): self.lean = (dx, dy)
        def set_com_height(self, z): self.z = z
        def rot2(self): return np.eye(2)

    class _S:
        box_pose = np.array([0.71, 0, 0.5, 1, 0, 0, 0, 0, 0, 0]); wrench_hands = np.zeros((2, 6)); c_W = np.array([0, 0, 0.6])

    pl = _P()
    ap.apply_values(pl, _S(), spec.to_values(np.zeros(spec.n_act)))
    assert pl.lean == (0.0, 0.0) and pl.z == pytest.approx(0.6)          # 初期値 0: 前傾もしゃがみも指示しない
    ap.apply_values(pl, _S(), spec.to_values(np.array([0, 0, 0.25, 0.5, 0.0, 0, 0, 0])))
    assert pl.lean == (pytest.approx(0.05), pytest.approx(0.05))


@pytest.mark.slow
def test_env_preset_p_with_schedule_equals_legacy():
    """P の方策が legacy と同じ値（cmp_dx = 手書き前傾、ほかの残差 0）を出せば軌道は同一。報酬だけ w_post の分が違う。"""
    cfg, push = load_box_experiment(PKG_ROOT / "configs" / "experiments" / "box_p2_walk_push.yaml")
    cfg.output.verbose = False
    out = {}
    for name in ("legacy", "P"):
        spec = BoxActionSpec.preset(name)
        env = BoxPushEnv(cfg, spec=spec, domain=BoxDomain(randomize=False), max_time=1.0, seed=0)
        pol = ScheduleBoxPolicy(spec, targets={"f_push": 40.0, "lean_x": 0.08}, t_start=0.2, t_ramp=0.3, t_lean_lead=0.2)
        obs = env.reset(); done = False
        while not done:
            pol.set_time(env.k * env.dt_ctrl)
            obs, r, done, info = env.step(pol.act(obs))
        out[name] = (np.array(env.state.q), env.w.w_post, info["lean_x"])
    assert np.array_equal(out["legacy"][0], out["P"][0]) and out["P"][2] == pytest.approx(0.08, abs=1e-6)
    assert out["legacy"][1] == 1.0 and out["P"][1] == 0.0


def test_hold_slow_updates_arm_only_every_100ms():
    spec = BoxActionSpec.preset("P_S_T")        # 500 Hz、腕・計画層は 100 ms 保持
    prev = np.zeros(spec.n_act)
    a = np.ones(spec.n_act)
    held = spec.hold_slow(a, prev, k=7)
    assert np.all(held[spec.slow_index] == 0.0) and np.all(held[spec.residual_index] == 1.0)
    assert np.array_equal(spec.hold_slow(a, prev, k=50), a)
    b = BoxActionSpec.preset("B")              # 100 Hz: 判断が k = 1, 6, ... でも 100 ms の区切りで更新される
    assert np.array_equal(b.hold_slow(a, prev, k=51), a) and np.all(b.hold_slow(a, prev, k=56)[b.slow_index] == 0.0)
    assert np.array_equal(BoxActionSpec.preset("P").hold_slow(a, prev, k=3), a)   # 10 Hz は保持なし


def test_merge_base_residual_or_passthrough():
    """基底方策（legacy: f_push, lean_x）の出力は、RL に同じ口があれば残差として足し、なければそのまま使う。"""
    base_actions = ("f_push", "lean_x")
    base_vals = {"f_push": 60.0, "lean_x": 0.10}
    st = BoxActionSpec.preset("S_T")                                  # 残差だけ → 基底の値をそのまま
    v = st.merge_base(st.to_values(np.zeros(st.n_act)), base_vals, base_actions)
    assert v["f_push"] == 60.0 and v["lean_x"] == 0.10 and v["cop_dx"] == 0.0
    pst = BoxActionSpec.preset("P_S_T")                               # 上位も残差学習 → 基底 + RL
    a = np.zeros(pst.n_act); a[0] = 0.25; a[2] = -0.25                 # f_push +15 N、cmp_dx −5 cm
    v = pst.merge_base(pst.to_values(a), base_vals, base_actions)
    assert v["f_push"] == pytest.approx(75.0) and v["lean_x"] == pytest.approx(0.05) and v["cmp_dx"] == pytest.approx(0.05)
    a[0] = 1.0
    assert pst.merge_base(pst.to_values(a), base_vals, base_actions)["f_push"] == pytest.approx(120.0)   # clip
