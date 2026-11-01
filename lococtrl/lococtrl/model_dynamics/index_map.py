"""! @file index_map.py
@brief MuJoCo の内部順序と、自前の脚・関節の順序を結ぶ索引表。

**索引の対応付けは必ず名前引きで構築する。** MJCF の記述順に依存した決め打ちは、
モデルを差し替えた瞬間に静かに壊れる。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["IndexMap", "build_index_map"]


@dataclass
class IndexMap:
    """! @brief 名前引きで構築した索引表。

    @param joint_qpos  関節 j の qpos 添字 shape (n,) [-]
    @param joint_qvel  関節 j の qvel 添字 shape (n,) [-]
    @param joint_act   関節 j のアクチュエータ添字 shape (n,) [-]
    @param leg_joints  脚 i の関節番号（自前順） shape (nf,k) [-]。
                       k は 1 脚あたりの関節数（G1 の脚 6）
    @param leg_qpos    脚 i の qpos 添字 shape (nf,3) [-]
    @param leg_qvel    脚 i の qvel 添字 shape (nf,3) [-]
    @param foot_site   脚 i の site id shape (nf,) [-]
    @param foot_geom   脚 i の geom id shape (nf,) [-]
    @param base_body   ベースボディ id [-]
    @param arm_joints  腕 a の関節番号（自前順）shape (n_arm,k_arm) [-]（腕が無ければ (0,0)）
    @param hand_site   腕 a の手先 site id shape (n_arm,) [-]
    @param n_joint     駆動関節数 n [-]
    @param n_leg       脚数 nf [-]
    @param nq          一般化座標次元 [-]
    @param nv          一般化速度次元 [-]
    @note 添字は「自前順（ロボット設定の leg_names / joint_names の順）-> MuJoCo 添字」の写像。
    """

    joint_qpos: np.ndarray
    joint_qvel: np.ndarray
    joint_act: np.ndarray
    leg_joints: np.ndarray
    leg_qpos: np.ndarray
    leg_qvel: np.ndarray
    foot_site: np.ndarray
    foot_geom: np.ndarray
    base_body: int
    n_joint: int
    n_leg: int
    nq: int
    nv: int
    arm_joints: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=int))
    hand_site: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))


def build_index_map(model, cfg) -> IndexMap:
    """! @brief MJCF モデルと設定から索引表を構築する。
    @param model mjModel（mujoco.MjModel）[-]
    @param cfg   RobotConfig [-]
    @return IndexMap
    @note MuJoCo に依存する唯一の索引構築点。制御層はこの結果のみを見る。
    """
    import mujoco

    n_leg = cfg.n_leg
    n_joint = cfg.n_joint

    joint_qpos = np.zeros(n_joint, dtype=int)
    joint_qvel = np.zeros(n_joint, dtype=int)
    joint_act = np.zeros(n_joint, dtype=int)
    # 1 脚あたりの関節数は機体で変わる（G1 の脚は hip x3 + knee + ankle x2 = 6）。
    # 全脚で同数であることだけ要求する。
    per_leg = {len(names) for names in cfg.joint_names}
    if len(per_leg) != 1:
        raise ValueError(f"脚ごとの関節数が揃っていない: {[len(n) for n in cfg.joint_names]}")
    k_leg = per_leg.pop()
    leg_joints = np.zeros((n_leg, k_leg), dtype=int)

    def _resolve(jname: str, k: int) -> None:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid < 0:
            raise KeyError(f"関節 '{jname}' が MJCF に存在しない")
        joint_qpos[k] = int(model.jnt_qposadr[jid])
        joint_qvel[k] = int(model.jnt_dofadr[jid])
        aid = -1
        for a in range(model.nu):
            if int(model.actuator_trnid[a, 0]) == jid:
                aid = a
                break
        if aid < 0:
            raise KeyError(f"関節 '{jname}' を駆動するアクチュエータが無い")
        joint_act[k] = aid

    k = 0
    for i, names in enumerate(cfg.joint_names):
        for j, jname in enumerate(names):
            _resolve(jname, k)
            leg_joints[i, j] = k
            k += 1

    # 腕（箱押し用 G1）。自前順で脚の後ろに並べる
    arm_names = list(getattr(cfg, "arm_joint_names", []) or [])
    per_arm = {len(a) for a in arm_names}
    if len(per_arm) > 1:
        raise ValueError(f"腕ごとの関節数が揃っていない: {[len(a) for a in arm_names]}")
    arm_joints = np.zeros((len(arm_names), per_arm.pop() if per_arm else 0), dtype=int)
    for a, names in enumerate(arm_names):
        for j, jname in enumerate(names):
            _resolve(jname, k)
            arm_joints[a, j] = k
            k += 1
    hand_names = list(getattr(cfg, "hand_sites", []) or [])
    hand_site = np.zeros(len(hand_names), dtype=int)
    for a, sname in enumerate(hand_names):
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, sname)
        if sid < 0:
            raise KeyError(f"site '{sname}' が MJCF に存在しない")
        hand_site[a] = sid

    foot_site = np.zeros(n_leg, dtype=int)
    foot_geom = np.zeros(n_leg, dtype=int)
    for i in range(n_leg):
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, cfg.foot_sites[i])
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, cfg.foot_geoms[i])
        if sid < 0:
            raise KeyError(f"site '{cfg.foot_sites[i]}' が MJCF に存在しない")
        if gid < 0:
            raise KeyError(f"geom '{cfg.foot_geoms[i]}' が MJCF に存在しない")
        foot_site[i] = sid
        foot_geom[i] = gid

    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cfg.base_body)
    if bid < 0:
        raise KeyError(f"body '{cfg.base_body}' が MJCF に存在しない")

    return IndexMap(
        joint_qpos=joint_qpos,
        joint_qvel=joint_qvel,
        joint_act=joint_act,
        leg_joints=leg_joints,
        leg_qpos=joint_qpos[leg_joints],
        leg_qvel=joint_qvel[leg_joints],
        foot_site=foot_site,
        foot_geom=foot_geom,
        base_body=int(bid),
        n_joint=n_joint,
        n_leg=n_leg,
        nq=int(model.nq),
        nv=int(model.nv),
        arm_joints=arm_joints,
        hand_site=hand_site,
    )
