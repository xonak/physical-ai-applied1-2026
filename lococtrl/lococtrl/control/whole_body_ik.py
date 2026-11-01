"""! @file whole_body_ik.py
@brief 全身逆運動学（速度レベルの減衰最小二乗、重み付き単一問題）。

重心・足先の参照軌道（`planner.dcm`、`gait.biped_schedule`）を関節角に落とす。

## 定式化

決定変数は一般化速度 `v` shape (nv,) = `[v_W(3); w_B(3); dq(n)]` である
（`robot_model` の規約。並進はワールド表現、回転はボディ表現）。
各タスク `i` のヤコビアン `J_i` と誤差 `e_i` に対して

    min_v  sum_i w_i || J_i v - e_i ||^2 + lambda || v ||^2

を解き、`q <- integrate(q, v)` で反復する。正規方程式は

    (sum_i w_i J_i^T J_i + lambda I) v = sum_i w_i J_i^T e_i

であり、`lambda > 0` が特異姿勢（膝が伸びきる、足先が届かない）での発散を防ぐ。

**階層 QP（null-space 射影）にはしていない。**
重み付き単一問題で解く。重みの比が優先度を表す。

## タスクの優先度

| 優先 | タスク | 次元 | 既定の重み |
|---|---|---|---|
| 1 | 足先位置 {W} | 3 x nf | 1e3 |
| 1 | 足先姿勢 {W} | 3 x nf | 1e2 |
| 2 | 重心の水平位置 | 2 | 1e2 |
| 3 | 胴体姿勢 | 3 | 1e1 |
| 4 | 関節姿勢の正則化 | n | 1e-1 |

## 足先と重心の重み比はトレードオフである（実測）

足を 5 cm 持ち上げる目標に対する IK の残差と、同じ重みで 0.2 m/s 歩行させた結果:

| foot_pos / com_xy | 足先残差 [m] | 歩行の CoP 内包率 [%] | 歩行の重心誤差 RMS [mm] |
|---|---|---|---|
| 1e3 / 1e2（既定） | 3.1e-4 | 64 | 117 |
| 1e5 / 1e1 | 4.7e-7 | 27 | 158 |

**足先を強く優先すると IK は精密になるが歩行は不安定になる。** 足先と重心は
12 自由度に対して 14 本の拘束であり同時には満たせないので、どちらかが犠牲になる。
歩行中は重心が LIPM 軌道から外れることが転倒に直結するため、既定では
重心側を落とさない比（1e3 / 1e2）を採る。足先残差 0.3 mm は着地位置の誤差として
許容できる大きさである。

足先姿勢を位置と同じ優先度に置くのは、二足では足裏を床に平らに保たないと
片側が浮いて CoP が足裏内を動けなくなるためである。

@note 本モジュールは `RobotModelBase` のみに依存し、MuJoCo には触れない。
      浮遊ベースの位置・姿勢も解の一部として動くが、実機では直接動かせないので、
      呼び出し側は関節角 `q[7:]` だけを指令に使う。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..common.frames import log_so3, quat_to_rotmat
from ..model_dynamics.robot_model import RobotModelBase

__all__ = ["WbikWeights", "WbikTarget", "WbikResult", "solve_whole_body_ik"]


@dataclass
class WbikWeights:
    """! @brief タスクの重みと反復の設定。

    @param foot_pos  足先位置の重み [-]
    @param foot_rot  足先姿勢の重み [-]
    @param com_xy    重心水平位置の重み [-]
    @param base_rot  胴体姿勢の重み [-]
    @param posture   関節姿勢正則化の重み [-]
    @param damping   減衰 lambda [-]
    @param max_iter  最大反復回数 [-]
    @param tol       収束判定（足先位置誤差の最大値）[m]
    @param step_limit 1 反復あたりの `|v|` の上限 [m, rad]
    @note 重みの桁差は 1e6 を超えないこと（超えると数値的に壊れる）。
          既定値は foot_pos 1e5 / posture 1e-1 で比が 1e6、上限ちょうどである。
    """

    foot_pos: float = 1.0e3
    foot_rot: float = 1.0e2
    com_xy: float = 1.0e2
    base_rot: float = 1.0e1
    posture: float = 1.0e-1
    damping: float = 1.0e-4
    max_iter: int = 50
    tol: float = 1.0e-6
    step_limit: float = 0.5


@dataclass
class WbikTarget:
    """! @brief 全身 IK の目標値。

    @param p_WF_des  足先位置 {W} shape (nf,3) [m]
    @param R_WF_des  足先姿勢 {W} shape (nf,3,3) [-]
    @param c_xy_des  重心の水平位置 {W} shape (2,) [m]
    @param quat_WB_des 胴体姿勢 (w,x,y,z) shape (4,) [-]
    @param q_posture 正則化の目標関節角 shape (n,) [rad]
    """

    p_WF_des: np.ndarray
    R_WF_des: np.ndarray
    c_xy_des: np.ndarray
    quat_WB_des: np.ndarray
    q_posture: np.ndarray


@dataclass
class WbikResult:
    """! @brief 全身 IK の結果。

    @param q          解いた一般化座標 shape (nq,) [m, -, rad]
    @param n_iter     反復回数 [-]
    @param res_foot_pos 足先位置の残差（最大）[m]
    @param res_foot_rot 足先姿勢の残差（最大）[rad]
    @param res_com_xy   重心水平の残差 [m]
    @param res_base_rot 胴体姿勢の残差 [rad]
    @param converged  収束したか [-]
    """

    q: np.ndarray
    n_iter: int
    res_foot_pos: float
    res_foot_rot: float
    res_com_xy: float
    res_base_rot: float
    converged: bool


def _residuals(
    model: RobotModelBase, q: np.ndarray, tgt: WbikTarget
) -> tuple[list[tuple[np.ndarray, np.ndarray, float]], dict[str, float]]:
    """! @brief 各タスクの (ヤコビアン, 誤差, 重み無し) を作る。
    @param model RobotModelBase [-]
    @param q     一般化座標 shape (nq,) [m, -, rad]
    @param tgt   目標値 [-]
    @return (タスク列 [(J, e, 種別重みのキー)], 残差の辞書)
    @note 種別は "foot_pos" / "foot_rot" / "com_xy" / "base_rot" / "posture"。
    """
    nv = model.nv
    nf = model.n_leg
    n = model.n_joint
    tasks: list[tuple[np.ndarray, np.ndarray, str]] = []

    # --- 足先位置・姿勢 ---
    res_pos = 0.0
    res_rot = 0.0
    p_WF = model.fk(q)
    for leg in range(nf):
        e = np.asarray(tgt.p_WF_des[leg], dtype=np.float64) - p_WF[leg]
        tasks.append((model.jacobian_foot(q, leg), e, "foot_pos"))
        res_pos = max(res_pos, float(np.abs(e).max()))

        R = model.foot_rotation(q, leg)
        R_des = np.asarray(tgt.R_WF_des[leg], dtype=np.float64)
        e_rot = log_so3(R_des @ R.T)
        tasks.append((model.jacobian_foot_rot(q, leg), e_rot, "foot_rot"))
        res_rot = max(res_rot, float(np.abs(e_rot).max()))

    # --- 重心の水平位置 ---
    # dc_W = (A_G[0:3] @ v) / m なので、水平成分のヤコビアンは A_G[0:2] / m。
    J_com = model.Ag(q)[0:2] / model.total_mass
    e_com = np.asarray(tgt.c_xy_des, dtype=np.float64) - model.com(q)[0:2]
    tasks.append((J_com, e_com, "com_xy"))

    # --- 胴体姿勢 ---
    # w_B（ボディ表現）が v[3:6] に入るので、選択行列で取り出す。
    # 誤差もボディ表現に合わせる: R_WB^T log(R_des R_WB^T) ではなく
    # log(R_WB^T R_des) がボディ系の回転ベクトルである。
    J_base = np.zeros((3, nv), dtype=np.float64)
    J_base[:, 3:6] = np.eye(3)
    R_WB = quat_to_rotmat(q[3:7])
    R_WB_des = quat_to_rotmat(np.asarray(tgt.quat_WB_des, dtype=np.float64))
    e_base = log_so3(R_WB.T @ R_WB_des)
    tasks.append((J_base, e_base, "base_rot"))

    # --- 関節姿勢の正則化 ---
    J_post = np.zeros((n, nv), dtype=np.float64)
    J_post[:, nv - n:] = np.eye(n)
    e_post = np.asarray(tgt.q_posture, dtype=np.float64) - q[7:]
    tasks.append((J_post, e_post, "posture"))

    res = {
        "foot_pos": res_pos,
        "foot_rot": res_rot,
        "com_xy": float(np.abs(e_com).max()),
        "base_rot": float(np.abs(e_base).max()),
    }
    return tasks, res


def solve_whole_body_ik(
    model: RobotModelBase,
    tgt: WbikTarget,
    q_init: np.ndarray,
    w: WbikWeights | None = None,
) -> WbikResult:
    """! @brief 全身 IK を解く。

    @param model  RobotModelBase [-]
    @param tgt    目標値 [-]
    @param q_init 初期値 shape (nq,) [m, -, rad]
    @param w      重みと反復設定 [-]。None なら既定
    @return WbikResult
    @note 収束判定は足先位置の残差だけで行う。他のタスクは優先度が低く、
          足先が合っていれば残差が残っていてよい（それが重み付けの意味である）。
    """
    w = WbikWeights() if w is None else w
    q = np.asarray(q_init, dtype=np.float64).copy()
    nv = model.nv
    weight_of = {
        "foot_pos": w.foot_pos,
        "foot_rot": w.foot_rot,
        "com_xy": w.com_xy,
        "base_rot": w.base_rot,
        "posture": w.posture,
    }

    res: dict[str, float] = {}
    it = 0
    for it in range(1, int(w.max_iter) + 1):
        tasks, res = _residuals(model, q, tgt)
        if res["foot_pos"] < w.tol and res["foot_rot"] < w.tol:
            break

        H = w.damping * np.eye(nv)
        g = np.zeros(nv, dtype=np.float64)
        for J, e, kind in tasks:
            wi = weight_of[kind]
            H += wi * (J.T @ J)
            g += wi * (J.T @ e)
        v = np.linalg.solve(H, g)

        norm = float(np.linalg.norm(v))
        if norm > w.step_limit:
            v *= w.step_limit / norm
        q = model.integrate(q, v)

    tasks, res = _residuals(model, q, tgt)
    return WbikResult(
        q=q,
        n_iter=it,
        res_foot_pos=res["foot_pos"],
        res_foot_rot=res["foot_rot"],
        res_com_xy=res["com_xy"],
        res_base_rot=res["base_rot"],
        converged=bool(res["foot_pos"] < w.tol and res["foot_rot"] < w.tol),
    )
