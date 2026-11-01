"""! @file test_box_p1.py
@brief 箱押しの力制御: 手先タスク・外力の重心レンチ反映・前傾の口を固定する（歩行はしない軽いテスト）。"""

from __future__ import annotations

import numpy as np
import pytest

import boxpush  # noqa: F401
from boxpush import PKG_ROOT
from boxpush.cli.run_box import load_box_experiment
from boxpush.push import BoxPushHook, PushParams
from lococtrl.common.config import load_robot_config
from lococtrl.control.hand_task import HandTask
from lococtrl.model_dynamics.robot_model import create_robot_model
from lococtrl.sim.mujoco_env import MujocoEnv

ROBOT_YAML = PKG_ROOT / "configs" / "robot_g1_box.yaml"


@pytest.fixture(scope="module")
def cfg():
    return load_robot_config(str(ROBOT_YAML))


def test_hand_task_force_axis_mask_and_limit():
    """kp_axis で押す向きを力制御にでき、上限 f_max が効く。"""
    ht = HandTask(n_arm=2, kp=300.0, kd=30.0, f_max=50.0)
    ht.set_target(0, [1.0, 0.0, 0.0], f_ff=[20.0, 0.0, 0.0], kp_axis=(0.0, 1.0, 1.0))
    F = ht.force(0, np.array([0.9, 0.01, -0.02]), np.zeros(3))
    assert F[0] == pytest.approx(20.0)              # x は f_ff だけ（位置誤差 0.1 m は無視）
    assert F[1] == pytest.approx(-3.0) and F[2] == pytest.approx(6.0)
    ht.set_target(1, [0.0, 0.0, 0.0], f_ff=[80.0, 0.0, 0.0])
    assert np.linalg.norm(ht.force(1, np.zeros(3), np.zeros(3))) == pytest.approx(50.0)


def test_hand_task_latches_orientation_and_limits_moment():
    ht = HandTask(n_arm=1, kp_rot=8.0, kd_rot=0.0, m_rot_max=4.0)
    R0 = np.eye(3)
    assert np.allclose(ht.moment(0, R0, np.zeros(3)), 0.0)          # 最初の姿勢にラッチ
    th = 0.3
    Ry = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
    M = ht.moment(0, Ry, np.zeros(3))
    assert abs(M[1]) == pytest.approx(8.0 * th, rel=1e-6) and abs(M[0]) < 1e-9 and abs(M[2]) < 1e-9
    assert np.linalg.norm(ht.moment(0, Ry @ Ry @ Ry, np.zeros(3))) == pytest.approx(4.0)


def test_com_wrench_of_env_moves_moment_to_com():
    wh = np.array([[[-10.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]).reshape(1, 6)
    W = HandTask.com_wrench_of_env(wh, np.array([[0.4, 0.0, 0.9]]), np.array([0.0, 0.0, 0.65]))
    assert np.allclose(W[0:3], [-10.0, 0.0, 0.0])
    assert W[4] == pytest.approx(-0.25 * 10.0)      # r_z f_x = (0.9-0.65)(-10)


def test_hand_jacobian_matches_finite_difference(cfg):
    model = create_robot_model(cfg)
    rng = np.random.default_rng(0)
    q = model.make_q(np.array([0.0, 0.0, cfg.h_nominal]), np.array([1.0, 0.0, 0.0, 0.0]),
                     cfg.q_nominal + 0.05 * rng.standard_normal(cfg.n_joint))
    J = model.jacobian_hand6(q, 0)
    assert J.shape == (6, model.nv)
    assert np.allclose(J[:, 6:6 + cfg.n_leg_joint], 0.0)   # 脚の列は 0
    p0 = model.hand_position(q, 0)
    h = 1.0e-6
    for k in (6 + cfg.n_leg_joint, 6 + cfg.n_leg_joint + 3):   # 左肩 pitch、左肘
        dq = np.zeros(model.nv); dq[k] = h
        p1 = model.hand_position(model.integrate(q, dq), 0)
        assert np.allclose((p1 - p0) / h, J[0:3, k], atol=1e-4)


def test_com_wrench_from_dcm_shifts_cop_with_external_force(cfg):
    """外力があると足裏 CoP 指令が f_ext/(m omega^2) だけずれる。重心への要求レンチは変わらない。"""
    from lococtrl.control.biped_stack import StabilizerParams, com_wrench_from_dcm
    from lococtrl.control.contact_wrench import FootContactSpec
    from lococtrl.gait.biped_schedule import gait_params_from_config
    from lococtrl.planner.biped_online import OnlineBipedPlanner, OnlinePlannerParams

    model = create_robot_model(cfg)
    env = MujocoEnv(cfg)
    st = env.reset(q_joint=cfg.q_nominal, p_WB=np.array([0.0, 0.0, cfg.h_nominal]))
    par = StabilizerParams.from_config()
    spec = FootContactSpec.from_config(cfg)
    pl = OnlineBipedPlanner(model, cfg, gait_params_from_config(None),
                            OnlinePlannerParams(source="velocity", endless=True, vx=0.0), 0.002)
    pl.set_command(0.0, 0.0, 0.0)
    ref = pl.update(st, 0)
    W0, cop0, _, _ = com_wrench_from_dcm(model, st, ref, par, pl.omega, spec)
    f = np.array([-15.0, 0.0, 0.0])
    W1, cop1, _, _ = com_wrench_from_dcm(model, st, ref, par, pl.omega, spec, f_ext=f)
    shift = f[0] / (model.total_mass * pl.omega ** 2)
    assert cop1[0] - cop0[0] == pytest.approx(shift, abs=1e-6)
    assert np.allclose(W1, W0, atol=1e-6)


def test_planner_lean_moves_stand_target(cfg):
    from lococtrl.gait.biped_schedule import gait_params_from_config
    from lococtrl.planner.biped_online import OnlineBipedPlanner, OnlinePlannerParams

    model = create_robot_model(cfg)
    env = MujocoEnv(cfg)
    st = env.reset(q_joint=cfg.q_nominal, p_WB=np.array([0.0, 0.0, cfg.h_nominal]))
    pl = OnlineBipedPlanner(model, cfg, gait_params_from_config(None),
                            OnlinePlannerParams(source="velocity", endless=True, vx=0.0), 0.002)
    pl.set_command(0.0, 0.0, 0.0)
    r0 = pl.update(st, 0)
    pl.set_lean(0.05, 0.0)
    r1 = pl.update(st, 1)
    assert r1.xi_ref[0] - r0.xi_ref[0] == pytest.approx(0.05, abs=2e-3)
    assert abs(r1.xi_ref[1] - r0.xi_ref[1]) < 1e-6


def test_push_hook_ramps_force_and_lean():
    """手書きスケジュール: 前傾が先にランプし、押付け力は t_start から。接触前は面の手前に構える。"""
    class _P:  # 計画器の代わり
        z_c = 0.65; z_c_nominal = 0.65; initialized = True
        def __init__(self): self.lean = None
        def set_lean(self, dx, dy=0.0): self.lean = (dx, dy)
        def set_com_height(self, z): self.z = z
        def rot2(self): return np.eye(2)

    class _S:
        box_pose = np.array([0.71, 0.0, 0.5, 1, 0, 0, 0, 0, 0, 0]); wrench_hands = np.zeros((2, 6)); q = np.zeros(26)
        c_W = np.array([0.0, 0.0, 0.65]); p_hands = np.zeros((2, 3)); tau_meas = np.zeros(26)

    hook = BoxPushHook(PushParams(f_push=40.0, t_start=2.0, t_ramp=1.0, lean_x=0.04, t_lean_lead=1.5, box_half_x=0.25,
                                  ki=0.0), dt=0.002)
    hook.rate_limits = {}   # ここではスケジュールそのものを見る
    pl, st = _P(), _S()
    hook(pl, st, 0)
    assert pl.lean == (0.0, 0.0) and not hook.hand_task.f_ff.any()
    assert hook.hand_task.p_des[0, 0] == pytest.approx(0.46 - 0.10)      # 面の手前に構える
    hook(pl, st, int(1.2 / 0.002))                                        # 判断周期（50）の倍数の k で
    assert pl.lean[0] == pytest.approx(0.04 * 0.7, abs=1e-3)            # 前傾が先に入る（t0 = 0.5 s から 1 s でランプ）
    hook(pl, st, int(2.5 / 0.002))
    assert hook.hand_task.f_ff[0, 0] == pytest.approx(10.0, abs=0.5)     # 片手 = 半分、ランプ途中
    assert tuple(hook.hand_task.kp_axis[0]) == (0.0, 1.0, 1.0)


def test_p1_experiments_parse():
    for name in ("box_p1_stand_push", "box_p1_force_track", "box_p1_stand_push_upright"):
        cfg, push = load_box_experiment(PKG_ROOT / "configs" / "experiments" / f"{name}.yaml")
        assert push is not None and push.box_half_x == pytest.approx(cfg.scenario.box.half_size[0])
        assert cfg.planner.online.endless



def test_push_section_rejects_unknown_key(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("robot: ../robot_g1_box.yaml\nmapper: {kind: dlr}\nplanner: {kind: online}\npush: {f_pussh: 1.0}\n", encoding="utf-8")
    with pytest.raises(KeyError):
        load_box_experiment(p)
