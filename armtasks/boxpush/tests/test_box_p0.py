"""! @file test_box_p0.py
@brief 箱押しの土台: 腕付き G1（configs/robot_g1_box.yaml）・箱・手首 F/T の土台を固定する（歩行はしない軽いテスト）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import boxpush  # noqa: F401  lococtrl を sys.path に足す
from boxpush import PKG_ROOT
from boxpush.model import make_g1_box
from lococtrl.cli.biped_experiment import experiment_config_from_dict, load_experiment_config
from lococtrl.common.config import load_robot_config
from lococtrl.model_dynamics.robot_model import create_robot_model
from lococtrl.sim.mujoco_env import BoxObject, MujocoEnv

ROBOT_YAML = PKG_ROOT / "configs" / "robot_g1_box.yaml"


@pytest.fixture(scope="module")
def cfg():
    return load_robot_config(str(ROBOT_YAML))


def test_generated_model_is_up_to_date(tmp_path):
    """assets/g1_box.xml は boxpush.model の生成物と一致する（原典を更新したら再生成する約束）。"""
    out = make_g1_box(dst=tmp_path / "g1_box.xml")
    a = (PKG_ROOT / "assets" / "g1_box.xml").read_text(encoding="utf-8")
    b = out.read_text(encoding="utf-8")
    # meshdir は出力先で変わるので除いて比べる
    strip = lambda t: "\n".join(l for l in t.splitlines() if "meshdir" not in l)
    assert strip(a) == strip(b)


def test_arm_joints_follow_legs(cfg):
    """自前順は 脚 12 → 腕 14。腕の関節・手先 site が名前引きで解決される。"""
    env = MujocoEnv(cfg)
    assert cfg.n_leg_joint == 12 and cfg.n_arm_joint == 14 and cfg.n_joint == 26
    assert env.idx.arm_joints.tolist() == [list(range(12, 19)), list(range(19, 26))]
    assert env.idx.hand_site.shape == (2,)
    assert env.has_joint_torque_sensors and env.has_hand_ft_sensors
    assert len(cfg.q_nominal) == len(cfg.tau_max) == 26


def test_legs_only_robot_has_no_arms():
    """腕の無い robot_g1.yaml は従来どおり（腕の索引は空、手首 F/T も無い）。"""
    c = load_robot_config("robot_g1.yaml")
    env = MujocoEnv(c)
    st = env.reset(q_joint=c.q_nominal, p_WB=np.array([0.0, 0.0, c.h_nominal]))
    assert c.n_joint == 12 and env.idx.arm_joints.size == 0
    assert st.wrench_hands.shape == (0, 6) and st.box_pose.shape == (0,)


def test_box_is_excluded_from_com_and_mass(cfg):
    """箱を置いても重心・総質量はロボットだけ（動力学モデルと一致）。"""
    model = create_robot_model(cfg)
    env = MujocoEnv(cfg, objects=[BoxObject(mass=30.0, pos=(1.5, 0.0, None))])
    st = env.reset(q_joint=cfg.q_nominal, p_WB=np.array([0.0, 0.0, cfg.h_nominal]))
    q = model.make_q(st.p_WB, st.quat_WB, st.q)
    assert env.total_mass == pytest.approx(model.total_mass, rel=1e-9)
    assert np.allclose(st.c_W, model.com(q), atol=1e-9)
    assert st.box_pose.shape == (10,)
    assert st.box_pose[2] == pytest.approx(0.25, abs=1e-3)  # 床に置く（半高さ）


def test_hand_touches_only_box_and_ft_sees_it(cfg):
    """手は箱とだけ当たり、箱が手を押し返す力（-x）を手首 F/T が拾う。"""
    import mujoco

    env = MujocoEnv(cfg, objects=[BoxObject(half_size=(0.2, 0.3, 0.15), mass=20.0, pos=(3.0, 0.0, None))])
    st = env.reset(q_joint=cfg.q_nominal, p_WB=np.array([0.0, 0.0, cfg.h_nominal]))
    m, d = env.model, env.data
    p_hand = d.site_xpos[env.idx.hand_site[0]].copy()
    a, _ = env._box_free[0]
    d.qpos[a : a + 3] = [p_hand[0] + 0.2 - 0.01, 0.0, p_hand[2]]  # 手に 1 cm 食い込ませる
    mujoco.mj_forward(m, d)
    names = {(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1),
              mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)) for c in d.contact[: d.ncon]}
    assert ("box_geom", "left_hand") in names or ("left_hand", "box_geom") in names
    assert not any("hand" in (g1 or "") and "hand" in (g2 or "") for g1, g2 in names)  # 手どうしは当たらない
    st = env.observe()
    assert st.wrench_hands[0, 0] < -1.0


def test_box_section_parses_and_rejects_unknown_key():
    cfg = load_experiment_config(PKG_ROOT / "configs" / "experiments" / "box_p0.yaml")
    assert Path(cfg.robot).resolve() == ROBOT_YAML.resolve() and cfg.scenario.box.enable
    ob = cfg.scenario.box.to_object()
    assert ob is not None and ob.pos[0] == pytest.approx(2.5)
    with pytest.raises(KeyError):
        experiment_config_from_dict({"scenario": {"box": {"massss": 1.0}}})
    assert experiment_config_from_dict({}).scenario.box.to_object() is None
