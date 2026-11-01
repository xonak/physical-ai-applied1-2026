"""! @file mujoco_backend.py
@brief MuJoCo を用いた動力学バックエンド（既定）。

**このファイルと sim/ 以下だけが mjModel / mjData を直接触ってよい。**
制御層は RobotModelBase のインタフェース越しにしかここへ触れない。

内部で保持する mjModel / mjData は、シミュレータ本体とは別インスタンスである。
モデル評価が物理状態を汚さないことを保証するためにこの分離を守る。
"""

from __future__ import annotations

import numpy as np

from ...common.config import RobotConfig
from ...common.frames import quat_to_rotmat, skew
from ..index_map import build_index_map
from ..robot_model import RobotModelBase

__all__ = ["MujocoRobotModel"]


class MujocoRobotModel(RobotModelBase):
    """! @brief MuJoCo の剛体動力学を RobotModelBase として提供する。"""

    def __init__(self, cfg: RobotConfig) -> None:
        """! @brief MJCF を読み込み、名前引きで索引表を構築する。
        @param cfg RobotConfig [-]
        @return なし
        """
        super().__init__(cfg)
        import mujoco

        self._mj = mujoco
        path = cfg.mjcf_abspath()
        if not path.is_file():
            raise FileNotFoundError(f"MJCF が見つからない: {path}")
        self._m = mujoco.MjModel.from_xml_path(str(path))
        self._d = mujoco.MjData(self._m)
        self.idx = build_index_map(self._m, cfg)

        # 浮遊ベース（free joint）の添字
        free = [j for j in range(self._m.njnt) if self._m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        if len(free) != 1:
            raise ValueError("MujocoRobotModel: free joint はちょうど 1 個であること")
        self._free_qadr = int(self._m.jnt_qposadr[free[0]])
        self._free_vadr = int(self._m.jnt_dofadr[free[0]])

        # 自前 dof 順 -> MuJoCo dof 順 の置換
        self._perm = np.concatenate(
            [np.arange(self._free_vadr, self._free_vadr + 6, dtype=int), self.idx.joint_qvel]
        )
        self._total_mass = float(np.sum(self._m.body_mass[1:]))

    # ---- 次元 -------------------------------------------------------------
    @property
    def nq(self) -> int:
        """! @brief 一般化座標次元 [-]"""
        return 7 + self.n_joint

    @property
    def nv(self) -> int:
        """! @brief 一般化速度次元 [-]"""
        return 6 + self.n_joint

    @property
    def total_mass(self) -> float:
        """! @brief 全質量 [kg]"""
        return self._total_mass

    # ---- 内部ユーティリティ ----------------------------------------------
    def _scatter_q(self, q: np.ndarray) -> None:
        """! @brief 自前順の q を mjData.qpos へ書く。
        @param q 一般化座標 shape (nq,) [m, -, rad]
        @return なし
        """
        q = np.asarray(q, dtype=np.float64).reshape(self.nq)
        a = self._free_qadr
        self._d.qpos[a : a + 7] = q[0:7]
        self._d.qpos[self.idx.joint_qpos] = q[7:]

    def _scatter_v(self, v: np.ndarray) -> None:
        """! @brief 自前順の v を mjData.qvel へ書く。
        @param v 一般化速度 shape (nv,) [m/s, rad/s]
        @return なし
        """
        v = np.asarray(v, dtype=np.float64).reshape(self.nv)
        a = self._free_vadr
        self._d.qvel[a : a + 6] = v[0:6]
        self._d.qvel[self.idx.joint_qvel] = v[6:]

    def _forward_pos(self, q: np.ndarray) -> None:
        """! @brief 位置依存量（運動学・重心・慣性）を更新する。
        @param q 一般化座標 shape (nq,) [m, -, rad]
        @return なし
        """
        self._scatter_q(q)
        self._mj.mj_kinematics(self._m, self._d)
        self._mj.mj_comPos(self._m, self._d)

    def _forward_vel(self, q: np.ndarray, v: np.ndarray) -> None:
        """! @brief 位置・速度依存量を更新する。
        @param q 一般化座標 shape (nq,) [m, -, rad]
        @param v 一般化速度 shape (nv,) [m/s, rad/s]
        @return なし
        """
        self._scatter_q(q)
        self._scatter_v(v)
        self._mj.mj_kinematics(self._m, self._d)
        self._mj.mj_comPos(self._m, self._d)
        self._mj.mj_comVel(self._m, self._d)

    # ---- 座標の演算 -------------------------------------------------------
    def integrate(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        """! @brief 一般化座標を dq（速度 x 時間）ぶん進める。
        @param q  一般化座標 shape (nq,) [m, -, rad]
        @param dq 変位 shape (nv,) [m, rad, rad]
        @return q_next shape (nq,) [m, -, rad]
        @note MuJoCo の mj_integratePos に委譲するため、クォータニオンの
              指数写像による更新が保証される。
        """
        self._scatter_q(q)
        qpos = np.array(self._d.qpos, dtype=np.float64)
        qvel = np.zeros(self._m.nv, dtype=np.float64)
        dq = np.asarray(dq, dtype=np.float64).reshape(self.nv)
        a = self._free_vadr
        qvel[a : a + 6] = dq[0:6]
        qvel[self.idx.joint_qvel] = dq[6:]
        self._mj.mj_integratePos(self._m, qpos, qvel, 1.0)
        out = np.empty(self.nq, dtype=np.float64)
        b = self._free_qadr
        out[0:7] = qpos[b : b + 7]
        out[7:] = qpos[self.idx.joint_qpos]
        return out

    # ---- 運動学 -----------------------------------------------------------
    def fk(self, q: np.ndarray) -> np.ndarray:
        """! @brief 全脚の足先位置（{W}）を返す。
        @param q 一般化座標 shape (nq,) [m, -, rad]
        @return p_WF shape (nf,3) [m]
        """
        self._forward_pos(q)
        return np.array(self._d.site_xpos[self.idx.foot_site], dtype=np.float64)

    def foot_position_base(self, q: np.ndarray, leg: int) -> np.ndarray:
        """! @brief 足先位置をベース系 {B} で返す。
        @param q   一般化座標 shape (nq,) [m, -, rad]
        @param leg 脚インデックス [-]
        @return p_BF shape (3,) [m]
        """
        p_WF = self.fk(q)[leg]
        p_WB = np.asarray(q[0:3], dtype=np.float64)
        R_WB = quat_to_rotmat(q[3:7])
        return R_WB.T @ (p_WF - p_WB)

    def jacobian_foot(self, q: np.ndarray, leg: int) -> np.ndarray:
        """! @brief 足先の並進ヤコビアン（{W} 表現、自前 dof 順）。
        @param q   一般化座標 shape (nq,) [m, -, rad]
        @param leg 脚インデックス [-]
        @return J shape (3,nv) [-]、v_WF = J @ v
        """
        self._forward_pos(q)
        jacp = np.zeros((3, self._m.nv), dtype=np.float64)
        self._mj.mj_jacSite(self._m, self._d, jacp, None, int(self.idx.foot_site[leg]))
        return jacp[:, self._perm]

    def jacobian_foot_rot(self, q: np.ndarray, leg: int) -> np.ndarray:
        """! @brief 足先の回転ヤコビアン（{W} 角速度）。
        @param q   一般化座標 shape (nq,) [m, -, rad]
        @param leg 脚インデックス [-]
        @return J_rot shape (3, nv) [-]、w_WF = J_rot @ v
        """
        self._forward_pos(q)
        jacr = np.zeros((3, self.nv), dtype=np.float64)
        self._mj.mj_jacSite(self._m, self._d, None, jacr, int(self.idx.foot_site[leg]))
        return jacr

    def foot_rotation(self, q: np.ndarray, leg: int) -> np.ndarray:
        """! @brief 足先 site の姿勢行列 {W}。
        @param q   一般化座標 shape (nq,) [m, -, rad]
        @param leg 脚インデックス [-]
        @return R_WF shape (3,3) [-]
        """
        self._forward_pos(q)
        return np.array(
            self._d.site_xmat[int(self.idx.foot_site[leg])], dtype=np.float64
        ).reshape(3, 3)

    def hand_position(self, q: np.ndarray, arm: int) -> np.ndarray:
        """! @brief 手先 site の位置 {W}。
        @param q   一般化座標 shape (nq,) [m, -, rad]
        @param arm 腕インデックス [-]
        @return p_WH shape (3,) [m]
        """
        self._forward_pos(q)
        return np.array(self._d.site_xpos[int(self.idx.hand_site[arm])], dtype=np.float64)

    def hand_rotation(self, q: np.ndarray, arm: int) -> np.ndarray:
        """! @brief 手先 site の姿勢行列 {W}。
        @param q   一般化座標 shape (nq,) [m, -, rad]
        @param arm 腕インデックス [-]
        @return R_WH shape (3,3) [-]
        """
        self._forward_pos(q)
        return np.array(self._d.site_xmat[int(self.idx.hand_site[arm])], dtype=np.float64).reshape(3, 3)

    def jacobian_hand6(self, q: np.ndarray, arm: int) -> np.ndarray:
        """! @brief 手先 site の 6 次元ヤコビアン（{W}、自前 dof 順）。
        @param q   一般化座標 shape (nq,) [m, -, rad]
        @param arm 腕インデックス [-]
        @return J shape (6,nv) [-]
        """
        self._forward_pos(q)
        jacp = np.zeros((3, self._m.nv), dtype=np.float64)
        jacr = np.zeros((3, self._m.nv), dtype=np.float64)
        self._mj.mj_jacSite(self._m, self._d, jacp, jacr, int(self.idx.hand_site[arm]))
        return np.vstack([jacp[:, self._perm], jacr[:, self._perm]])

    def jacobian_foot_leg(self, q: np.ndarray, leg: int) -> np.ndarray:
        """! @brief 脚 3 関節のみのヤコビアン（ベース固定、{B} 表現）。
        @param q   一般化座標 shape (nq,) [m, -, rad]
        @param leg 脚インデックス [-]
        @return J_leg shape (3,3) [-]、v_BF = J_leg @ dq_leg
        """
        J = self.jacobian_foot(q, leg)  # (3, nv)
        cols = 6 + self.idx.leg_joints[leg]  # 自前 dof 順での脚関節列
        R_WB = quat_to_rotmat(q[3:7])
        return R_WB.T @ J[:, cols]

    def jacobian_foot_dot_v(self, q: np.ndarray, v: np.ndarray, h: float = 1.0e-6) -> np.ndarray:
        """! @brief Jdot @ v を中心差分で評価する。
        @param q 一般化座標 shape (nq,) [m, -, rad]
        @param v 一般化速度 shape (nv,) [m/s, rad/s]
        @param h 差分幅 [s]
        @return acc_bias shape (nf,3) [m/s^2]
        @note 解析形（mj_rnePostConstraint 経由）に置き換える余地がある。
        """
        v = np.asarray(v, dtype=np.float64).reshape(self.nv)
        q_p = self.integrate(q, v * h)
        q_m = self.integrate(q, -v * h)
        out = np.zeros((self.n_leg, 3), dtype=np.float64)
        for i in range(self.n_leg):
            jp = self.jacobian_foot(q_p, i) @ v
            jm = self.jacobian_foot(q_m, i) @ v
            out[i] = (jp - jm) / (2.0 * h)
        return out

    # ---- 動力学 -----------------------------------------------------------
    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        """! @brief 慣性行列（自前 dof 順）。
        @param q 一般化座標 shape (nq,) [m, -, rad]
        @return M shape (nv,nv) [kg, kg*m^2]
        """
        self._forward_pos(q)
        self._mj.mj_crb(self._m, self._d)
        M = np.zeros((self._m.nv, self._m.nv), dtype=np.float64)
        self._full_M(M)
        return M[np.ix_(self._perm, self._perm)]

    def _full_M(self, dst: np.ndarray) -> None:
        """! @brief 疎形式の慣性行列を密行列に展開する（MuJoCo の版差を吸収）。
        @param dst 書き込み先 shape (nv,nv) [kg, kg*m^2]
        @return なし
        @note mj_fullM の引数順は MuJoCo 3.3 前後で
              (m, dst, qM_sparse) -> (m, d, dst) に変わった。
        """
        try:
            self._mj.mj_fullM(self._m, self._d, dst)  # 新しい版
        except TypeError:
            qM = getattr(self._d, "qM", None)
            if qM is None:
                qM = self._d.M
            self._mj.mj_fullM(self._m, dst, qM)  # 旧版

    def nle(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """! @brief 非線形項 h = C(q,v)v + g(q)（自前 dof 順）。
        @param q 一般化座標 shape (nq,) [m, -, rad]
        @param v 一般化速度 shape (nv,) [m/s, rad/s]
        @return h shape (nv,) [N, N*m]
        """
        self._forward_vel(q, v)
        out = np.zeros(self._m.nv, dtype=np.float64)
        self._mj.mj_rne(self._m, self._d, 0, out)  # flg_acc=0 -> qacc を無視
        return out[self._perm]

    def com(self, q: np.ndarray) -> np.ndarray:
        """! @brief 全身重心位置 {W}。
        @param q 一般化座標 shape (nq,) [m, -, rad]
        @return c_W shape (3,) [m]
        """
        self._forward_pos(q)
        return np.array(self._d.subtree_com[0], dtype=np.float64)

    def Ag(self, q: np.ndarray) -> np.ndarray:
        """! @brief 重心運動量行列 A_G（自前 dof 順）。
        @param q 一般化座標 shape (nq,) [m, -, rad]
        @return A_G shape (6,nv)、h_G = A_G @ v
                上 3 行 [kg*m/s]、下 3 行 [kg*m^2/s]（重心まわり）
        @note 各剛体 i について
              A_lin += m_i * Jp_i,
              A_ang += I_i^W * Jr_i + m_i * skew(r_i - c) * Jp_i
              I_i^W = R_i diag(I_i^B) R_i^T（R_i は慣性主軸系の姿勢）。
        """
        self._forward_pos(q)
        nv_mj = self._m.nv
        c = np.array(self._d.subtree_com[0], dtype=np.float64)
        A = np.zeros((6, nv_mj), dtype=np.float64)
        jacp = np.zeros((3, nv_mj), dtype=np.float64)
        jacr = np.zeros((3, nv_mj), dtype=np.float64)
        for b in range(1, self._m.nbody):
            m_b = float(self._m.body_mass[b])
            if m_b <= 0.0:
                continue
            jacp[:] = 0.0
            jacr[:] = 0.0
            self._mj.mj_jacBodyCom(self._m, self._d, jacp, jacr, b)
            R_i = np.array(self._d.ximat[b], dtype=np.float64).reshape(3, 3)
            I_b = np.diag(np.array(self._m.body_inertia[b], dtype=np.float64))
            I_W = R_i @ I_b @ R_i.T
            r_i = np.array(self._d.xipos[b], dtype=np.float64)
            A[0:3] += m_b * jacp
            A[3:6] += I_W @ jacr + m_b * skew(r_i - c) @ jacp
        return A[:, self._perm]
