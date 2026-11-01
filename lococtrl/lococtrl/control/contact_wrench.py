"""! @file contact_wrench.py
@brief 平面足の接触レンチ: 6 次元ヤコビアンと、摩擦錐 + CoP 矩形の線形制約。

点接触なら足先 1 点の力 f in R^3 で足りるが、二足の足裏は面で接するので
接触は **レンチ w = [f; tau] in R^6**（足先 site = 足裏中心まわり、{W} 表現）になる。
CoP が足裏の内側にあるという条件は、足裏系 {F} で書くと線形になる:

    |f_x| <= mu f_z,  |f_y| <= mu f_z            （摩擦錐の 4 面近似）
    |tau_y| <= L f_z   (x_cop = -tau_y / f_z)      （前後の CoP 制限）
    |tau_x| <= W f_z   (y_cop =  tau_x / f_z)      （左右の CoP 制限）
    |tau_z| <= mu_yaw f_z                          （ねじり摩擦）
    f_min <= f_z <= f_max

{W} 表現のレンチに対しては T = blkdiag(R_WF^T, R_WF^T) を左から掛けてから適用する。
足裏が床に平らでないと {F} の z は床法線と一致しないが、本実装は IK / 全身制御で
足裏を平らに保つタスクを入れているので、その前提で近似する。

記号: nf 脚数 [-]、nv 一般化速度次元 [-]、{W} ワールド系、{F} 足裏系。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..common.config import RobotConfig
from ..model_dynamics.robot_model import RobotModelBase

__all__ = [
    "FootContactSpec",
    "foot_jacobian6",
    "foot_jacobian6_dot_v",
    "wrench_cone_rows",
    "wrench_to_cop",
    "N_CONE_ROWS",
]

## 1 足あたりの線形制約の行数 [-]（摩擦 4 + CoP 4 + ねじり 2 + 法線 1）
N_CONE_ROWS = 11


@dataclass
class FootContactSpec:
    """! @brief 足裏の接触条件。

    @param half_length 足裏の前後半長 L [m]
    @param half_width  足裏の左右半長 W [m]
    @param mu          摩擦係数 [-]
    @param mu_yaw      ねじり摩擦の係数 [m]
    @param f_min       立脚の法線力下限 [N]
    @param f_max       立脚の法線力上限 [N]
    """

    half_length: float
    half_width: float
    mu: float
    mu_yaw: float
    f_min: float
    f_max: float

    @staticmethod
    def from_config(rcfg: RobotConfig) -> "FootContactSpec":
        """! @brief configs/robot_*.yaml から読む。
        @param rcfg RobotConfig [-]
        @return FootContactSpec
        @note `foot_half_size` と `mu_yaw` は `extra` に入る（二足だけが持つ項目）。
        """
        hs = rcfg.extra.get("foot_half_size")
        if hs is None:
            raise KeyError("robot config に foot_half_size が無い（平面足の二足で必要）")
        return FootContactSpec(
            half_length=float(hs[0]),
            half_width=float(hs[1]),
            mu=float(rcfg.mu),
            mu_yaw=float(rcfg.extra.get("mu_yaw", 0.02)),
            f_min=float(rcfg.f_min),
            f_max=float(rcfg.f_max),
        )


def foot_jacobian6(model: RobotModelBase, q: np.ndarray, leg: int) -> np.ndarray:
    """! @brief 足先 site の 6 次元ヤコビアン（{W}、[並進; 回転]）。
    @param model RobotModelBase [-]
    @param q     一般化座標 shape (nq,) [m, -, rad]
    @param leg   脚インデックス [-]
    @return J shape (6, nv) [-]、[v_WF; w_WF] = J @ v
    """
    return np.vstack([model.jacobian_foot(q, leg), model.jacobian_foot_rot(q, leg)])


def foot_jacobian6_dot_v(
    model: RobotModelBase, q: np.ndarray, v: np.ndarray, h: float = 1.0e-6
) -> np.ndarray:
    """! @brief 6 次元ヤコビアンの時間微分と速度の積 Jdot @ v（中心差分）。
    @param model RobotModelBase [-]
    @param q     一般化座標 shape (nq,) [m, -, rad]
    @param v     一般化速度 shape (nv,) [m/s, rad/s]
    @param h     差分幅 [s]
    @return bias shape (nf, 6) [m/s^2, rad/s^2]
    @note `MujocoRobotModel.jacobian_foot_dot_v` と同じ手法を回転にも広げたもの。
    """
    v = np.asarray(v, dtype=np.float64).reshape(model.nv)
    q_p = model.integrate(q, v * h)
    q_m = model.integrate(q, -v * h)
    out = np.zeros((model.n_leg, 6), dtype=np.float64)
    for i in range(model.n_leg):
        jp = foot_jacobian6(model, q_p, i) @ v
        jm = foot_jacobian6(model, q_m, i) @ v
        out[i] = (jp - jm) / (2.0 * h)
    return out


def wrench_cone_rows(
    spec: FootContactSpec, R_WF: np.ndarray, load: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """! @brief 1 足ぶんの線形制約 lo <= A w_W <= up を作る。
    @param spec FootContactSpec [-]
    @param R_WF 足裏姿勢 {W} shape (3,3) [-]
    @param load 荷重係数 [-] in [0,1]。法線力の上下限に掛ける（0 で遊脚 = レンチ 0）
    @return (A shape (11,6), lo shape (11,), up shape (11,))
    @note 荷重係数を両脚支持期で線形に動かすと、踏み替えで接触レンチが
          不連続に飛ばない（Mesesan 2019 がトルクの連続性を重視する理由と同じ）。
    """
    R = np.asarray(R_WF, dtype=np.float64).reshape(3, 3)
    T = np.zeros((6, 6))
    T[0:3, 0:3] = R.T
    T[3:6, 3:6] = R.T
    A_F = np.zeros((N_CONE_ROWS, 6))
    lo = np.full(N_CONE_ROWS, -np.inf)
    up = np.zeros(N_CONE_ROWS)
    r = 0
    # 摩擦錐: +-f_x - mu f_z <= 0, +-f_y - mu f_z <= 0
    for ax in (0, 1):
        for sgn in (1.0, -1.0):
            A_F[r, ax] = sgn
            A_F[r, 2] = -spec.mu
            r += 1
    # CoP: +-tau_x - W f_z <= 0, +-tau_y - L f_z <= 0
    for ax, lim in ((3, spec.half_width), (4, spec.half_length)):
        for sgn in (1.0, -1.0):
            A_F[r, ax] = sgn
            A_F[r, 2] = -lim
            r += 1
    # ねじり: +-tau_z - mu_yaw f_z <= 0
    for sgn in (1.0, -1.0):
        A_F[r, 5] = sgn
        A_F[r, 2] = -spec.mu_yaw
        r += 1
    # 法線力
    A_F[r, 2] = 1.0
    lam = float(np.clip(load, 0.0, 1.0))
    lo[r] = spec.f_min * lam
    up[r] = spec.f_max * lam
    return A_F @ T, lo, up


def wrench_to_cop(wrench_W: np.ndarray, p_site: np.ndarray, R_WF: np.ndarray) -> np.ndarray:
    """! @brief 足先 site まわりのレンチから足裏内の CoP（{W}）を求める。
    @param wrench_W レンチ shape (6,) [N, N*m]（{W}、site まわり）
    @param p_site   足先 site 位置 {W} shape (3,) [m]
    @param R_WF     足裏姿勢 {W} shape (3,3) [-]
    @return cop shape (3,) [m]（f_z <= 0 なら NaN）
    """
    w = np.asarray(wrench_W, dtype=np.float64).reshape(6)
    R = np.asarray(R_WF, dtype=np.float64).reshape(3, 3)
    f = R.T @ w[0:3]
    t = R.T @ w[3:6]
    if f[2] <= 1.0e-6:
        return np.full(3, np.nan)
    d_F = np.array([-t[1] / f[2], t[0] / f[2], 0.0])
    return np.asarray(p_site, dtype=np.float64) + R @ d_F
