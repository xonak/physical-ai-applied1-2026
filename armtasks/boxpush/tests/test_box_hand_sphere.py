"""! @file test_box_hand_sphere.py
@brief 手の衝突を球にした別モデル（configs/robot_g1_box_sphere.yaml、assets/g1_box_sphere.xml）を固定する。

球は手先 site を中心とする半径 HAND_SPHERE_RADIUS。前端が g1_box.xml の指先（ゴム手メッシュの最遠点）と
ほぼ同じ位置に来ること、手は箱とだけ当たること、手首 F/T が拾うことを確かめる。
"""

from __future__ import annotations

import numpy as np
import pytest

import boxpush  # noqa: F401  lococtrl を sys.path に足す
from boxpush import PKG_ROOT
from boxpush.model import HAND_SPHERE_RADIUS, make_g1_box_sphere
from lococtrl.common.config import load_robot_config
from lococtrl.sim.mujoco_env import BoxObject, MujocoEnv

ROBOT_YAML = PKG_ROOT / "configs" / "robot_g1_box_sphere.yaml"
ROBOT_MESH_YAML = PKG_ROOT / "configs" / "robot_g1_box.yaml"


@pytest.fixture(scope="module")
def cfg():
    return load_robot_config(str(ROBOT_YAML))


def test_generated_sphere_model_is_up_to_date(tmp_path):
    """assets/g1_box_sphere.xml は boxpush.model の生成物と一致する。"""
    out = make_g1_box_sphere(dst=tmp_path / "g1_box_sphere.xml")
    a = (PKG_ROOT / "assets" / "g1_box_sphere.xml").read_text(encoding="utf-8")
    b = out.read_text(encoding="utf-8")
    strip = lambda t: "\n".join(l for l in t.splitlines() if "meshdir" not in l)
    assert strip(a) == strip(b)


def test_hand_geom_is_sphere_at_site_and_front_matches_fingertip(cfg):
    """衝突 geom `*_hand` は手先 site 中心の球で、前端 x がメッシュ版の指先と 5 mm 以内。"""
    import mujoco

    env = MujocoEnv(cfg)
    m = env.model
    env_mesh = MujocoEnv(load_robot_config(str(ROBOT_MESH_YAML)))
    mm = env_mesh.model
    for k, hand in enumerate(("left", "right")):
        g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{hand}_hand")
        assert m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE
        assert m.geom_size[g, 0] == pytest.approx(HAND_SPHERE_RADIUS)
        s = env.idx.hand_site[k]
        assert m.geom_bodyid[g] == m.site_bodyid[s]
        assert np.allclose(m.geom_pos[g], m.site_pos[s], atol=1e-9)
        # 手ボディ系での前端 x（球）と、メッシュ版の指先 x を比べる
        front_sphere = m.geom_pos[g, 0] + m.geom_size[g, 0]
        gm = mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_GEOM, f"{hand}_hand")
        mid = mm.geom_dataid[gm]
        v0, nv = mm.mesh_vertadr[mid], mm.mesh_vertnum[mid]
        verts = mm.mesh_vert[v0 : v0 + nv]
        # MuJoCo はメッシュを重心・主軸系に直すので、geom_quat で手ボディ系に戻してから最遠点をとる
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, mm.geom_quat[gm])
        verts_body = mm.geom_pos[gm] + verts @ R.reshape(3, 3).T
        front_mesh = float(verts_body[:, 0].max())
        assert abs(front_sphere - front_mesh) < 0.005
        # 衝突ビット・優先度・摩擦はメッシュ版と同じ
        assert m.geom_contype[g] == mm.geom_contype[gm] and m.geom_conaffinity[g] == mm.geom_conaffinity[gm]
        assert m.geom_priority[g] == mm.geom_priority[gm]
        assert np.allclose(m.geom_friction[g], mm.geom_friction[gm])
        # 見た目: メッシュは透明、球の visual がある
        vis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{hand}_hand_vis_sphere")
        assert vis >= 0 and m.geom_contype[vis] == 0 and m.geom_conaffinity[vis] == 0


def test_sphere_hand_touches_only_box_and_ft_sees_it(cfg):
    """球の手は箱とだけ当たり、箱が手を押し返す力（-x）を手首 F/T が拾う。"""
    import mujoco

    env = MujocoEnv(cfg, objects=[BoxObject(half_size=(0.2, 0.3, 0.15), mass=20.0, pos=(3.0, 0.0, None))])
    st = env.reset(q_joint=cfg.q_nominal, p_WB=np.array([0.0, 0.0, cfg.h_nominal]))
    m, d = env.model, env.data
    p_hand = d.site_xpos[env.idx.hand_site[0]].copy()
    a, _ = env._box_free[0]
    # 球の前端（site + r）に 1 cm 食い込ませる
    d.qpos[a : a + 3] = [p_hand[0] + HAND_SPHERE_RADIUS + 0.2 - 0.01, 0.0, p_hand[2]]
    mujoco.mj_forward(m, d)
    names = {(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1),
              mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)) for c in d.contact[: d.ncon]}
    assert ("box_geom", "left_hand") in names or ("left_hand", "box_geom") in names
    assert not any("hand" in (g1 or "") and "hand" in (g2 or "") for g1, g2 in names)
    assert sum(1 for c in d.contact[: d.ncon] if "left_hand" in {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1),
                                                                   mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)}) == 1  # 接触点は 1 点
    st = env.observe()
    assert st.wrench_hands[0, 0] < -1.0
