"""! @file wbc_passivity.py
@brief 受動性ベース全身制御（DLR 型。Henze 2016 / Mesesan 2019 の縮約版）。

IHMC 型（`wbc_momentum`）が加速度レベルの逆動力学 QP で全変数を同時に解くのに対し、
DLR 型は **加速度を解かない**。閉ループを PD+（Paden & Panja）型

    M dnu + C nu = tau_ext - J^T [w_c^imp; f_grf; f_imp]

に置くことを目標に、次の 3 段で関節トルクを作る（Mesesan 2019 式 (8)-(13)）。

1. 重心インピーダンスと DCM 追従則が要求する重心レンチ W_des（`biped_stack.com_wrench_from_dcm`）
2. 接触レンチ配分 QP:  min ||sum_i G_i w_i - W_des||^2_Qc + ||w - w_nom||^2_Qf
                       s.t. 摩擦錐・CoP 矩形・ねじり・法線力（荷重係数つき）
3. トルク写像:          tau = h(q, v)[6:] - sum_stance (J_i^T w_i)[6:] + sum_swing (J_i^T F_imp,i)[6:]
                             + N tau_posture

遊脚と胴体姿勢は **インピーダンス**（位置・速度の誤差に比例する力）で扱い、
関節姿勢は接触・遊脚タスクの零空間 N に射影して階層化する（Henze 2016 の零空間射影）。
下半身 12 自由度の G1 では両足の 6 次元タスクで零空間がほぼ空になるので、
姿勢タスクは実質的に効かない。これは理論どおりで、上半身を持つ TORO では効く。

**何が IHMC 型と違うか**: 速度の数値微分や Jdot v を使わないので速度雑音に強い。
接触レンチだけが QP で、関節トルクは J^T 写像の閉形式。受動性に基づく安定性の証明が
あるのはこの構造（トルクセンサ付きの関節トルク制御を前提）。代償として、立脚の
加速度拘束を明示的に課さないため、モデル誤差があると足が床をこする方向に力が出る。

記号: nv 一般化速度次元 [-]、na 駆動関節数 [-]、nf 脚数 [-]、{W} ワールド系。
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
from .contact_wrench import N_CONE_ROWS, FootContactSpec, foot_jacobian6, wrench_cone_rows

__all__ = ["PassivityWbcParams", "PassivityWbc"]


@dataclass
class PassivityWbcParams:
    """! @brief DLR 型の固有パラメータ（biped.yaml の stabilizer.dlr 節）。
    @param w_com         レンチ配分 QP の重心レンチ残差の重み [-]
    @param w_reg         公称配分への正則化の重み [-]
    @param m_foot_eq     遊脚インピーダンスの等価質量 [kg]（加速度ゲイン -> 剛性 [N/m] の換算）
    @param i_foot_eq     遊脚姿勢インピーダンスの等価慣性 [kg*m^2]
    @param posture_scale 関節姿勢インピーダンスを慣性行列の対角でスケールする係数 [-]
    @param null_damping  零空間射影の擬似逆の減衰 [-]
    @param i_base_eq     胴体姿勢インピーダンス（立脚の関節で胴体にモーメントを返す）の等価慣性 [kg*m^2]。
                         0 で無効。重心レンチのモーメント成分は接触のねじり摩擦で頭打ちになり、
                         胴体ヨーが ±13 deg 揺れた（実測）。立脚の股関節で直接押さえる（Henze 2016 の胴体タスク）。ヨーのみ
    @param arm_gain_scale 腕の関節姿勢インピーダンスに掛ける倍率 [-]（箱押し用 G1。腕が無い機体では効かない）。
                         腕の慣性行列の対角は脚より 1 桁以上小さく、脚と同じ kp_posture では腕が柔らかすぎる。
                         倍率 1 だと前押し 18 N*s で腕が 15 deg 振れて転倒し、10 で腕固定（robot_g1.yaml）と
                         同じ押し限界（18・19 は耐え 20 で転倒）になった（box_p0_push_forward、実測）
    """

    w_com: float = 1.0e2
    w_reg: float = 1.0e-2
    m_foot_eq: float = 2.0
    i_foot_eq: float = 0.05
    posture_scale: float = 1.0
    null_damping: float = 1.0e-3
    i_base_eq: float = 0.3
    arm_gain_scale: float = 10.0

    @staticmethod
    def from_params(par: StabilizerParams) -> "PassivityWbcParams":
        """! @brief stabilizer.dlr 節（無ければ既定）から読む。
        @param par StabilizerParams [-]
        @return PassivityWbcParams
        """
        d = dict(par.extra.get("dlr", {}) or {})
        p = PassivityWbcParams()
        for k in list(vars(p)):
            if k in d:
                setattr(p, k, float(d[k]))
        return p


class PassivityWbc:
    """! @brief 受動性ベース全身制御（接触レンチ配分 QP + J^T 写像 + 零空間階層）。"""

    def __init__(
        self,
        model: RobotModelBase,
        rcfg: RobotConfig,
        par: StabilizerParams,
        spec: FootContactSpec | None = None,
    ) -> None:
        """! @brief 構築する。
        @param model RobotModelBase [-]
        @param rcfg  RobotConfig [-]
        @param par   StabilizerParams [-]
        @param spec  FootContactSpec（None なら rcfg から）[-]
        @return なし
        """
        self.model = model
        self.rcfg = rcfg
        self.par = par
        self.p = PassivityWbcParams.from_params(par)
        self.spec = FootContactSpec.from_config(rcfg) if spec is None else spec
        self.nv, self.na, self.nf = model.nv, model.n_joint, model.n_leg
        self.n_var = 6 * self.nf
        self.n_con = N_CONE_ROWS * self.nf
        self._prob = None
        self._build_pattern()
        self.last_solve_time = 0.0
        self.last_status = "none"
        self.last_tau = np.zeros(self.na)
        self.last_wrench = np.zeros((self.nf, 6))
        self.last_residual = np.zeros(3)  # [重心レンチ, 遊脚インピーダンス力, 姿勢トルク]
        ## 手先タスク（腕のある機体で上位が与える。None なら従来どおり）`control.hand_task.HandTask`
        self.hand_task = None
        ## 直前周期に足へ配分した重心レンチ（手の分を引いた後）shape (6,)
        self.last_wrench_feet = np.zeros(6)
        self.n_solve = 0
        self.n_fail = 0
        self.n_cold = 0

    @property
    def name(self) -> str:
        """! @brief 名前 [-]"""
        return "dlr"

    def _build_pattern(self) -> None:
        """! @brief QP のスパースパターンを固定する（密）。
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

    def distribute_wrench(
        self,
        wrench_des: np.ndarray,
        p_feet: np.ndarray,
        c: np.ndarray,
        R_feet: np.ndarray,
        contact: np.ndarray,
        load: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """! @brief 重心レンチを接触レンチに配分する QP（Mesesan 2019 式 (13)）。
        @param wrench_des 重心に要求するレンチ shape (6,) [N, N*m]
        @param p_feet     足先 site 位置 {W} shape (nf,3) [m]
        @param c          重心位置 {W} shape (3,) [m]
        @param R_feet     足裏姿勢 shape (nf,3,3) [-]
        @param contact    立脚 shape (nf,) [bool]
        @param load       荷重係数 shape (nf,) [-]
        @return (w shape (nf,6) [N, N*m], 重心レンチの残差ノルム)
        """
        import osqp
        import scipy.sparse as sp

        nf, n = self.nf, self.n_var
        con = np.asarray(contact, dtype=bool)
        lam = np.asarray(load, dtype=np.float64) * con
        G = np.zeros((6, n))
        for i in range(nf):
            if not con[i]:
                continue
            G[0:3, 6 * i : 6 * i + 3] = np.eye(3)
            G[3:6, 6 * i : 6 * i + 3] = skew(np.asarray(p_feet)[i] - np.asarray(c))
            G[3:6, 6 * i + 3 : 6 * i + 6] = np.eye(3)
        W = np.asarray(wrench_des, dtype=np.float64).reshape(6)
        tot = float(np.sum(lam))
        w_nom = np.zeros(n)
        if tot > 0.0:
            for i in range(nf):
                w_nom[6 * i : 6 * i + 3] = W[0:3] * lam[i] / tot
        P = self.p.w_com * (G.T @ G) + self.p.w_reg * np.eye(n)
        g = -self.p.w_com * (G.T @ W) - self.p.w_reg * w_nom
        A = np.zeros((self.n_con, n))
        lo = np.zeros(self.n_con)
        up = np.zeros(self.n_con)
        lam_b = self.par.bound_factor(lam)
        for i in range(nf):
            Ai, li, ui = wrench_cone_rows(self.spec, R_feet[i], float(lam_b[i]))
            r = N_CONE_ROWS * i
            A[r : r + N_CONE_ROWS, 6 * i : 6 * i + 6] = Ai
            lo[r : r + N_CONE_ROWS] = li
            up[r : r + N_CONE_ROWS] = ui
        P_csc = sp.csc_matrix(
            (P[self._P_rows, self._P_cols][self._P_perm], self._P_csc.indices, self._P_csc.indptr),
            shape=(n, n),
        )
        A_csc = sp.csc_matrix(
            (A.ravel()[self._A_perm], self._A_csc.indices, self._A_csc.indptr),
            shape=(self.n_con, n),
        )
        t0 = time.perf_counter()
        if self._prob is None:
            self._prob = osqp.OSQP()
            self._prob.setup(
                P_csc, g, A_csc, lo, up, verbose=False, warm_starting=True, polishing=False,
                eps_abs=1.0e-5, eps_rel=1.0e-5, max_iter=10000,
            )
        else:
            self._prob.update(q=g, l=lo, u=up, Px=P_csc.data, Ax=A_csc.data)
        res = self._prob.solve()
        self.last_status = str(res.info.status)
        if self.last_status not in ("solved", "solved inaccurate"):
            # ウォームスタートの退化で max_iter に達することがある（コールドなら解ける）。
            # 同じ周期内に一度だけ作り直して解き直す。
            self._prob = osqp.OSQP()
            self._prob.setup(
                P_csc, g, A_csc, lo, up, verbose=False, warm_starting=True, polishing=False,
                eps_abs=1.0e-5, eps_rel=1.0e-5, max_iter=10000,
            )
            res = self._prob.solve()
            self.last_status = str(res.info.status)
            self.n_cold += 1
        self.last_solve_time = time.perf_counter() - t0
        self.n_solve += 1
        if self.last_status not in ("solved", "solved inaccurate") or res.x is None or not np.all(np.isfinite(res.x)):
            self.n_fail += 1
            return self.last_wrench.copy(), float("nan")
        w = np.asarray(res.x, dtype=np.float64)
        return w.reshape(nf, 6), float(np.linalg.norm(G @ w - W))

    def solve(
        self,
        state: RobotState,
        ref: BipedReference,
        wrench_des: np.ndarray,
        contact: np.ndarray,
        load: np.ndarray,
    ) -> np.ndarray:
        """! @brief 関節トルクを返す。
        @param state      推定状態 [-]
        @param ref        参照 [-]
        @param wrench_des 重心に要求するレンチ shape (6,) [N, N*m]
        @param contact    立脚として扱う足 shape (nf,) [bool]
        @param load       荷重係数 shape (nf,) [-]
        @return tau shape (na,) [N*m]
        """
        model, par, p = self.model, self.par, self.p
        na, nf = self.na, self.nf
        R = quat_to_rotmat(state.quat_WB)
        q = model.make_q(state.p_WB, state.quat_WB, state.q)
        v = model.make_v(R @ state.v_B, state.w_B, state.dq)
        h = model.nle(q, v)
        J = np.array([foot_jacobian6(model, q, i) for i in range(nf)])
        R_feet = np.array([model.foot_rotation(q, i) for i in range(nf)])
        p_feet = np.asarray(state.p_WF, dtype=np.float64)
        con = np.asarray(contact, dtype=bool)

        # 0) 手先タスク（腕のある機体）: 手が受ける実測レンチを重心レンチから差し引く
        ht = self.hand_task
        n_arm = self.rcfg.n_arm
        W_feet = np.asarray(wrench_des, dtype=np.float64).reshape(6).copy()
        J_hand = None
        p_hands = None
        if ht is not None and n_arm > 0:
            J_hand = [model.jacobian_hand6(q, a) for a in range(n_arm)]
            p_hands = np.array([model.hand_position(q, a) for a in range(n_arm)])
            wh = np.asarray(state.wrench_hands, dtype=np.float64)
            if ht.comp_f_env and wh.shape == (n_arm, 6):
                W_feet -= ht.com_wrench_of_env(wh, p_hands, np.asarray(state.c_W))
        self.last_wrench_feet = W_feet

        # 1-2) 重心レンチ -> 接触レンチ配分
        w, res_c = self.distribute_wrench(W_feet, p_feet, state.c_W, R_feet, con, load)
        self.last_wrench = w

        # 3) トルク写像: 重力・コリオリ補償 - 接触レンチ + 遊脚インピーダンス
        tau = h[6:].copy()
        for i in np.flatnonzero(con):
            tau -= (J[i].T @ w[i])[6:]
        f_imp_norm = 0.0
        for i in np.flatnonzero(~con):
            vel = J[i] @ v
            e_R = log_so3(ref.R_foot[i] @ R_feet[i].T)
            F = np.empty(6)
            F[0:3] = p.m_foot_eq * (
                ref.a_foot[i]
                + par.kp_foot * (ref.p_foot[i] - p_feet[i])
                + par.kd_foot * (ref.v_foot[i] - vel[0:3])
            )
            F[3:6] = p.i_foot_eq * (par.kp_foot_rot * e_R - par.kd_foot_rot * vel[3:6])
            tau += (J[i].T @ F)[6:]
            f_imp_norm += float(np.linalg.norm(F[0:3]))

        # 立脚の足裏安定化（Mesesan 2019 貢献 iii）: ラッチ位置・平らな姿勢へのインピーダンス
        for i in np.flatnonzero(con):
            vel = J[i] @ v
            e_R = log_so3(ref.R_foot[i] @ R_feet[i].T)
            F = np.empty(6)
            F[0:3] = p.m_foot_eq * (par.kp_stance_foot * (ref.p_foot[i] - p_feet[i]) - par.kd_stance_foot * vel[0:3])
            F[3:6] = p.i_foot_eq * (par.kp_stance_rot * e_R - par.kd_stance_rot * vel[3:6])
            tau += (J[i].T @ F)[6:]

        # 胴体姿勢インピーダンス: 立脚の関節を通して胴体にモーメントを返す。
        # 足が固定なら w_B = -J_rot,leg qdot なので、仮想仕事から tau = -J_rot,leg^T M。荷重で両脚に配分
        if p.i_base_eq > 0.0 and int(np.sum(con)) == 1:  # 単脚支持のみ（両脚だと閉ループ鎖で配分と干渉し、立位で転倒した）
            e_Rb = log_so3(ref.R_base @ R.T)
            w_W = R @ np.asarray(state.w_B, dtype=np.float64)
            M_base = p.i_base_eq * (par.kp_rot * e_Rb - par.kd_rot * w_W)
            M_base[0:2] = 0.0  # roll / pitch は重心レンチのモーメント成分に任せ、ここはヨーだけ（前押しで膝と干渉した）
            ld = np.asarray(load, dtype=np.float64) * con
            ld = ld / max(float(ld.sum()), 1.0e-6)
            for i in np.flatnonzero(con):
                tau -= ld[i] * (J[i][3:6, 6:].T @ M_base)

        # 手先インピーダンス（腕のある機体、上位が目標を与えたときだけ）
        if J_hand is not None:
            for a in range(n_arm):
                if not ht.active[a]:
                    continue
                Jp = J_hand[a][0:3]
                F = ht.force(a, p_hands[a], Jp @ v)
                tau += (Jp.T @ F)[6:]
                if ht.hold_rot:
                    # 姿勢モーメントは位置タスクの零空間（腕の関節、トルクレベル）に射影する。
                    # 射影しないと、腕を伸ばす向きの力と姿勢保持のモーメントが肩で打ち消し合い、
                    # 手が前に出なくなった（上限 4 N*m のモーメントが 30 N の押し力を止めた、実測）
                    Jr = J_hand[a][3:6]
                    M_h = ht.moment(a, model.hand_rotation(q, a), Jr @ v)
                    cols_a = 6 + np.asarray(model.idx.arm_joints[a])
                    Jpa = Jp[:, cols_a]
                    Npa = np.eye(Jpa.shape[1]) - Jpa.T @ np.linalg.solve(Jpa @ Jpa.T + ht.null_damping * np.eye(3), Jpa)
                    tau[cols_a - 6] += Npa @ (Jr[:, cols_a].T @ M_h)

        # 零空間に射影した関節姿勢インピーダンス（Henze 2016 の階層化）
        Jt = np.vstack([J[i][:, 6:] for i in range(nf)])  # (6 nf, na)
        JJt = Jt @ Jt.T + p.null_damping * np.eye(Jt.shape[0])
        N = np.eye(na) - Jt.T @ np.linalg.solve(JJt, Jt)
        M = model.mass_matrix(q)
        scale = p.posture_scale * np.diag(M)[6:]
        n_lj = self.rcfg.n_leg_joint
        if na > n_lj and p.arm_gain_scale != 1.0:
            scale[n_lj:] *= p.arm_gain_scale  # 腕だけ（脚のみの機体では何もしない）
        tau_post = scale * (par.kp_posture * (ref.q_posture - np.asarray(state.q)) - par.kd_posture * np.asarray(state.dq))
        tau_post = N @ tau_post
        # 手先タスクのある腕は、その腕の姿勢トルクを手先ヤコビアン（腕の列）の零空間に落とす
        if J_hand is not None:
            for a in range(n_arm):
                if not ht.active[a]:
                    continue
                cols = model.idx.arm_joints[a] if hasattr(model, "idx") else None
                if cols is None:
                    continue
                Ja = (J_hand[a] if ht.hold_rot else J_hand[a][0:3])[:, 6 + np.asarray(cols)]
                Na = np.eye(Ja.shape[1]) - Ja.T @ np.linalg.solve(Ja @ Ja.T + ht.null_damping * np.eye(Ja.shape[0]), Ja)
                tau_post[cols] = Na @ tau_post[cols]
        tau += tau_post

        tau = np.clip(tau, -self.rcfg.tau_max, self.rcfg.tau_max)
        self.last_tau = tau
        self.last_residual = np.array([res_c, f_imp_norm, float(np.linalg.norm(tau_post))])
        return tau.copy()

    def reset(self) -> None:
        """! @brief 内部状態を初期化する。
        @return なし
        """
        self.last_tau = np.zeros(self.na)
        self.last_wrench = np.zeros((self.nf, 6))
        self.n_solve = 0
        self.n_fail = 0
