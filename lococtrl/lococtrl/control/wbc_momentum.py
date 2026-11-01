"""! @file wbc_momentum.py
@brief 運動量ベース全身 QP（IHMC 型。Koolen et al. 2016 の縮約版）。

二足・平面足の全身 QP。決定変数は

    z = [ddq(nv); w(6*nf); tau(na)]      w_i = [f_i; tau_i] 足先 site まわりの接触レンチ {W}

で、等式拘束に浮遊ベース動力学 `M ddq + h = S^T tau + sum_i J_i^T w_i` を置き、
接触レンチは `contact_wrench.wrench_cone_rows`（摩擦錐 + CoP 矩形 + ねじり + 法線力）で縛る。
コストは重み付き最小二乗の和（階層 QP にはしない）:

    w_contact  ||J_i ddq + Jdot_i v||^2                （立脚: 足の 6 次元加速度ゼロ）
    w_momentum ||sum_i G_i w_i - W_des||^2             （重心レンチ = 運動量変化率の目標）
    w_swing    ||J_sw ddq + Jdot_sw v - a_sw_des||^2    （遊脚の 6 次元加速度追従）
    w_base     ||R ddq_rot - alpha_des||^2              （胴体姿勢の角加速度）
    w_posture  ||ddq_j - ddq_j_des||^2                  （関節姿勢へ戻す加速度）
    w_wrench   ||w - w_nominal||^2                      （荷重係数で配った公称レンチへの正則化）

Koolen 2016 との対応: 彼らの QP は決定変数を [ddq; rho]（rho は接触レンチの基底係数）に取り、
運動量変化率タスク `A_G ddq + Adot_G v = hdot_des` と空間加速度タスクを重みで束ね、
逆動力学でトルクを出す。本実装は接触レンチを直接変数にし（Adot_G の
数値微分を避けるため運動量タスクを接触レンチで書く）、tau も変数に含めて動力学を
等式拘束で課す。数学的には同じ問題の別の書き方である。

記号: nv 一般化速度次元 [-]、na 駆動関節数 [-]、nf 脚数 [-]。
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..common.config import RobotConfig
from ..common.frames import log_so3, quat_to_rotmat, skew
from ..common.types import RobotState
from ..model_dynamics.robot_model import RobotModelBase
from .biped_stack import BipedReference, StabilizerParams
from .contact_wrench import (
    N_CONE_ROWS,
    FootContactSpec,
    foot_jacobian6,
    foot_jacobian6_dot_v,
    wrench_cone_rows,
)

__all__ = ["MomentumWbcWeights", "MomentumWbc"]


@dataclass
class MomentumWbcWeights:
    """! @brief タスク重み（比が優先度）。
    @param contact   立脚の加速度ゼロ [-]
    @param momentum  重心レンチ [-]
    @param swing     遊脚追従 [-]
    @param base      胴体姿勢 [-]
    @param posture   関節姿勢 [-]
    @param wrench    接触レンチの正則化 [-]
    @note 比は 接触 : 運動量 : 遊脚 = 1e4 : 1e2 : 1e2 とし、絶対値を
          1/100 にしてある。接触レンチ（数百 N）を含む目的関数の絶対値が 1e8 を超えると
          OSQP の相対停止条件が効かず max_iter（解時間 40 ms）に達するため。
          桁差は 1e6 以内。
    """

    contact: float = 1.0e2
    momentum: float = 1.0
    swing: float = 1.0
    base: float = 1.0e-1
    posture: float = 1.0e-4
    wrench: float = 1.0e-4

    @staticmethod
    def from_params(par: StabilizerParams) -> "MomentumWbcWeights":
        """! @brief biped.yaml の stabilizer.ihmc 節（無ければ既定）から読む。
        @param par StabilizerParams [-]
        @return MomentumWbcWeights
        """
        d = dict(par.extra.get("ihmc", {}) or {})
        w = MomentumWbcWeights(
            contact=float(d.get("w_contact", 1.0e2)),
            momentum=float(d.get("w_momentum", 1.0)),
            swing=float(d.get("w_swing", 1.0)),
            base=float(d.get("w_base", 1.0e-1)),
            posture=float(d.get("w_posture", 1.0e-4)),
            wrench=float(d.get("w_wrench", 1.0e-4)),
        )
        vals = np.array([w.contact, w.momentum, w.swing, w.base, w.posture, w.wrench])
        if vals.max() / vals.min() > 1.0e8 * (1.0 + 1.0e-9):
            raise ValueError("MomentumWbc のタスク重みの桁差が 1e8 を超えている")
        return w


class MomentumWbc:
    """! @brief 二足・平面足の運動量ベース全身 QP（IHMC 型）。"""

    def __init__(
        self,
        model: RobotModelBase,
        rcfg: RobotConfig,
        par: StabilizerParams,
        spec: FootContactSpec | None = None,
    ) -> None:
        """! @brief QP の構造を固定する。
        @param model RobotModelBase [-]
        @param rcfg  RobotConfig [-]
        @param par   StabilizerParams [-]
        @param spec  FootContactSpec（None なら rcfg から）[-]
        @return なし
        """
        self.model = model
        self.rcfg = rcfg
        self.par = par
        self.spec = FootContactSpec.from_config(rcfg) if spec is None else spec
        self.w = MomentumWbcWeights.from_params(par)
        self.reg = 1.0e-6
        self.nv, self.na, self.nf = model.nv, model.n_joint, model.n_leg
        self.n_var = self.nv + 6 * self.nf + self.na
        self.n_con = self.nv + N_CONE_ROWS * self.nf + self.na
        self._prob = None
        self._build_pattern()
        self.last_solve_time = 0.0
        self.last_status = "none"
        self.last_tau = np.zeros(self.na)
        self.last_wrench = np.zeros((self.nf, 6))
        self.last_ddq = np.zeros(self.nv)
        self.last_residual = np.zeros(6)
        self.n_solve = 0
        self.n_fail = 0
        self.n_cold = 0

    @property
    def name(self) -> str:
        """! @brief 名前 [-]"""
        return "ihmc"

    @property
    def i_ddq(self) -> slice:
        """! @brief ddq のブロック [-]"""
        return slice(0, self.nv)

    def i_w(self, leg: int) -> slice:
        """! @brief 足 leg のレンチのブロック [-]"""
        return slice(self.nv + 6 * leg, self.nv + 6 * leg + 6)

    @property
    def i_tau(self) -> slice:
        """! @brief tau のブロック [-]"""
        return slice(self.nv + 6 * self.nf, self.n_var)

    def _build_pattern(self) -> None:
        """! @brief 密ブロックのスパースパターンを固定する。
        @return なし
        """
        import scipy.sparse as sp

        n, m = self.n_var, self.n_con
        rp, cp = np.triu_indices(n)
        coo = sp.coo_matrix((np.arange(rp.size, dtype=float), (rp, cp)), shape=(n, n)).tocsc()
        self._P_rows, self._P_cols = rp, cp
        self._P_perm = coo.data.astype(np.int64)
        self._P_csc = sp.csc_matrix((np.zeros(rp.size), coo.indices, coo.indptr), shape=(n, n))
        ra = np.repeat(np.arange(m), n)
        ca = np.tile(np.arange(n), m)
        cooa = sp.coo_matrix((np.arange(ra.size, dtype=float), (ra, ca)), shape=(m, n)).tocsc()
        self._A_perm = cooa.data.astype(np.int64)
        self._A_csc = sp.csc_matrix((np.zeros(ra.size), cooa.indices, cooa.indptr), shape=(m, n))

    # ------------------------------------------------------------------ 組み立て
    def build_qp(
        self,
        state: RobotState,
        ref: BipedReference,
        wrench_des: np.ndarray,
        contact: np.ndarray,
        load: np.ndarray,
    ) -> tuple:
        """! @brief QP を組み立てる（解かない）。
        @param state      推定状態 [-]
        @param ref        参照 [-]
        @param wrench_des 重心に要求するレンチ shape (6,) [N, N*m]
        @param contact    立脚として扱う足 shape (nf,) [bool]
        @param load       荷重係数 shape (nf,) [-]
        @return (P, q, A, lo, up, info)
        """
        import scipy.sparse as sp

        model, par = self.model, self.par
        nv, na, nf, n = self.nv, self.na, self.nf, self.n_var
        R = quat_to_rotmat(state.quat_WB)
        q = model.make_q(state.p_WB, state.quat_WB, state.q)
        v = model.make_v(R @ state.v_B, state.w_B, state.dq)
        M = model.mass_matrix(q)
        h = model.nle(q, v)
        J = np.array([foot_jacobian6(model, q, i) for i in range(nf)])  # (nf,6,nv)
        Jdv = foot_jacobian6_dot_v(model, q, v)  # (nf,6)
        R_feet = np.array([model.foot_rotation(q, i) for i in range(nf)])
        p_feet = np.asarray(state.p_WF, dtype=np.float64)
        c = np.asarray(state.c_W, dtype=np.float64)
        con = np.asarray(contact, dtype=bool)

        P = np.zeros((n, n))
        g = np.zeros(n)

        def add_task(A: np.ndarray, b: np.ndarray, w: float) -> None:
            P[:, :] += w * (A.T @ A)
            g[:] += -w * (A.T @ b)

        # 1) 立脚: 足の 6 次元加速度。純粋なゼロではなく、ラッチした着地位置と
        #    平らな足裏姿勢へ戻す PD を加速度に直して入れる（stance foot stabilization）。
        a_st_cmd = np.zeros((nf, 6))
        for i in np.flatnonzero(con):
            vel = J[i] @ v
            e_R = log_so3(ref.R_foot[i] @ R_feet[i].T)
            a_st_cmd[i, 0:3] = par.kp_stance_foot * (ref.p_foot[i] - p_feet[i]) - par.kd_stance_foot * vel[0:3]
            a_st_cmd[i, 3:6] = par.kp_stance_rot * e_R - par.kd_stance_rot * vel[3:6]
            Ac = np.zeros((6, n))
            Ac[:, self.i_ddq] = J[i]
            add_task(Ac, a_st_cmd[i] - Jdv[i], self.w.contact)

        # 2) 重心レンチ: sum_i G_i w_i = W_des、G_i = [[I, 0], [skew(p_i - c), I]]
        Am = np.zeros((6, n))
        for i in range(nf):
            if not con[i]:
                continue
            s = self.i_w(i)
            Am[0:3, s.start : s.start + 3] = np.eye(3)
            Am[3:6, s.start : s.start + 3] = skew(p_feet[i] - c)
            Am[3:6, s.start + 3 : s.start + 6] = np.eye(3)
        add_task(Am, np.asarray(wrench_des, dtype=np.float64).reshape(6), self.w.momentum)

        # 3) 遊脚: 位置・姿勢の PD を加速度に直して追従
        w_W = R @ np.asarray(state.w_B, dtype=np.float64)
        a_sw_cmd = np.zeros((nf, 6))
        for i in np.flatnonzero(~con):
            vel = J[i] @ v
            e_R = log_so3(ref.R_foot[i] @ R_feet[i].T)
            a_sw_cmd[i, 0:3] = (
                ref.a_foot[i]
                + par.kp_foot * (ref.p_foot[i] - p_feet[i])
                + par.kd_foot * (ref.v_foot[i] - vel[0:3])
            )
            a_sw_cmd[i, 3:6] = par.kp_foot_rot * e_R - par.kd_foot_rot * vel[3:6]
            As = np.zeros((6, n))
            As[:, self.i_ddq] = J[i]
            add_task(As, a_sw_cmd[i] - Jdv[i], self.w.swing)

        # 4) 胴体姿勢: ベースの角加速度（{W}）。ddq[3:6] はボディ表現なので R を掛ける
        e_Rb = log_so3(ref.R_base @ R.T)
        alpha_des = par.kp_rot * e_Rb - par.kd_rot * w_W
        Ab = np.zeros((3, n))
        Ab[:, 3:6] = R
        add_task(Ab, alpha_des, self.w.base)

        # 5) 関節姿勢
        ddq_j_des = par.kp_posture * (ref.q_posture - np.asarray(state.q)) - par.kd_posture * np.asarray(state.dq)
        Ap = np.zeros((na, n))
        Ap[:, 6:nv] = np.eye(na)
        add_task(Ap, ddq_j_des, self.w.posture)

        # 6) 接触レンチの正則化: 荷重係数で配った公称レンチ（法線力のみ）
        lam = np.asarray(load, dtype=np.float64) * con
        tot = float(np.sum(lam))
        w_nom = np.zeros(6 * nf)
        if tot > 0.0:
            for i in range(nf):
                w_nom[6 * i + 2] = float(wrench_des[2]) * lam[i] / tot
        Aw = np.zeros((6 * nf, n))
        Aw[:, nv : nv + 6 * nf] = np.eye(6 * nf)
        add_task(Aw, w_nom, self.w.wrench)

        P[np.diag_indices(n)] += self.reg

        # ---- 制約 ----
        A = np.zeros((self.n_con, n))
        lo = np.zeros(self.n_con)
        up = np.zeros(self.n_con)
        A[0:nv, self.i_ddq] = M
        for i in range(nf):
            A[0:nv, self.i_w(i)] = -J[i].T
        A[6:nv, self.i_tau] = -np.eye(na)
        lo[0:nv] = -h
        up[0:nv] = -h
        r0 = nv
        lam_b = par.bound_factor(lam)
        for i in range(nf):
            Ai, li, ui = wrench_cone_rows(self.spec, R_feet[i], float(lam_b[i]))
            A[r0 : r0 + N_CONE_ROWS, self.i_w(i)] = Ai
            lo[r0 : r0 + N_CONE_ROWS] = li
            up[r0 : r0 + N_CONE_ROWS] = ui
            r0 += N_CONE_ROWS
        A[r0 : r0 + na, self.i_tau] = np.eye(na)
        lo[r0 : r0 + na] = -self.rcfg.tau_max
        up[r0 : r0 + na] = self.rcfg.tau_max

        P_csc = sp.csc_matrix(
            (P[self._P_rows, self._P_cols][self._P_perm], self._P_csc.indices, self._P_csc.indptr),
            shape=(n, n),
        )
        A_csc = sp.csc_matrix(
            (A.ravel()[self._A_perm], self._A_csc.indices, self._A_csc.indptr),
            shape=(self.n_con, n),
        )
        info = {"J": J, "Jdv": Jdv, "con": con, "Am": Am, "wrench_des": wrench_des,
                "a_sw": a_sw_cmd, "a_st": a_st_cmd, "alpha": alpha_des, "Ab": Ab, "ddq_j_des": ddq_j_des}
        return P_csc, g, A_csc, lo, up, info

    # ------------------------------------------------------------------ 求解
    def solve(
        self,
        state: RobotState,
        ref: BipedReference,
        wrench_des: np.ndarray,
        contact: np.ndarray,
        load: np.ndarray,
    ) -> np.ndarray:
        """! @brief QP を解いて関節トルクを返す。
        @param state      推定状態 [-]
        @param ref        参照 [-]
        @param wrench_des 重心に要求するレンチ shape (6,) [N, N*m]
        @param contact    立脚として扱う足 shape (nf,) [bool]
        @param load       荷重係数 shape (nf,) [-]
        @return tau shape (na,) [N*m]
        @note 失敗時は前回のトルクを返し、残差を NaN にする（黙って壊れない）。
        """
        import osqp

        P, g, A, lo, up, info = self.build_qp(state, ref, wrench_des, contact, load)
        t0 = time.perf_counter()
        if self._prob is None:
            self._prob = osqp.OSQP()
            self._prob.setup(
                P, g, A, lo, up, verbose=False, warm_starting=True, polishing=False,
                eps_abs=1.0e-5, eps_rel=1.0e-5, max_iter=4000,
            )
        else:
            self._prob.update(q=g, l=lo, u=up, Px=P.data, Ax=A.data)
        res = self._prob.solve()
        self.last_status = str(res.info.status)
        if self.last_status not in ("solved", "solved inaccurate"):
            # ウォームスタートの退化対策: 同じ周期内に一度だけコールドで解き直す
            self._prob = osqp.OSQP()
            self._prob.setup(
                P, g, A, lo, up, verbose=False, warm_starting=True, polishing=False,
                eps_abs=1.0e-5, eps_rel=1.0e-5, max_iter=4000,
            )
            res = self._prob.solve()
            self.last_status = str(res.info.status)
            self.n_cold += 1
        self.last_solve_time = time.perf_counter() - t0
        self.n_solve += 1
        if self.last_status not in ("solved", "solved inaccurate") or res.x is None or not np.all(np.isfinite(res.x)):
            self.n_fail += 1
            self.last_residual = np.full(6, np.nan)
            return self.last_tau.copy()
        z = np.asarray(res.x, dtype=np.float64)
        self.last_tau = z[self.i_tau].copy()
        self.last_wrench = z[self.nv : self.nv + 6 * self.nf].reshape(self.nf, 6).copy()
        self.last_ddq = z[self.i_ddq].copy()
        self.last_residual = self._residuals(z, info)
        return self.last_tau.copy()

    def _residuals(self, z: np.ndarray, info: dict) -> np.ndarray:
        """! @brief 各タスクの重み無し残差 [接触, 運動量, 遊脚, 胴体, 姿勢, レンチ]。
        @param z    解 [-]
        @param info build_qp の補助情報 [-]
        @return residual shape (6,)
        """
        ddq = z[self.i_ddq]
        J, Jdv, con = info["J"], info["Jdv"], info["con"]
        out = np.zeros(6)
        if np.any(con):
            out[0] = float(np.linalg.norm(
                [J[i] @ ddq + Jdv[i] - info["a_st"][i] for i in np.flatnonzero(con)]))
        out[1] = float(np.linalg.norm(info["Am"] @ z - info["wrench_des"]))
        sw = ~con
        if np.any(sw):
            out[2] = float(np.linalg.norm(
                [J[i] @ ddq + Jdv[i] - info["a_sw"][i] for i in np.flatnonzero(sw)]))
        out[3] = float(np.linalg.norm(info["Ab"] @ z - info["alpha"]))
        out[4] = float(np.linalg.norm(ddq[6:] - info["ddq_j_des"]))
        out[5] = float(np.linalg.norm(z[self.nv : self.nv + 6 * self.nf]))
        return out

    def reset(self) -> None:
        """! @brief 内部状態を初期化する。
        @return なし
        """
        self.last_tau = np.zeros(self.na)
        self.last_wrench = np.zeros((self.nf, 6))
        self.n_solve = 0
        self.n_fail = 0
