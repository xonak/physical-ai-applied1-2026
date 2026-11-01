"""! @file biped_stack.py
@brief 二足のトルク制御スタックの共通層: 参照生成と DCM 安定化。

「IK + 関節 PD（位置制御）」の代わりに、トルク制御の全身制御で歩かせる。
全身制御の作り方は 3 通り用意する（`wbc_momentum` = IHMC 型、`wbc_passivity` = DLR 型、
`policy.biped_residual_policy` = 学習方策）が、**その上の 2 層はすべて共通**で、本モジュールが持つ:

    BipedPlannerLayer   スケジュール（DS/SS、足配置、区分一定 CoP）+ DCM 閉形式軌道
                        -> 時刻 t の参照 BipedReference（重心・DCM・足先・荷重係数）
    com_wrench_from_dcm DCM 追従則で CoP 指令を作り、重心に要求するレンチ W_des に直す

これは Koolen 2016（ICP 軌道 + ICP フィードバック -> 運動量変化率）と
Mesesan 2019（DCM 閉形式 + DCM 追従則 -> 重心インピーダンス）の上 2 層が同じ式である
ことを使っている。式は `dcm_feedback.dcm_feedback_cop` そのもので、Englsberger 2015 の
`v = v_ref + (1 + k)(xi - xi_ref)` と等価である。

## 荷重係数 lambda_i in [0, 1]

接触レンチの法線力の上下限に掛ける係数。両脚支持期（DS）で離陸側を 1 -> 0、
着地側を 0 -> 1 に線形に動かし、踏み替えで接触レンチ（したがってトルク）が
飛ばないようにする。Mesesan 2019 §II/IV の「トルクの不連続を避ける」に対応する。

記号: nf 脚数（= 2）[-]、{W} ワールド系、omega = sqrt(g / z_c) [rad/s]。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..common.config import RobotConfig, load_yaml
from ..common.frames import log_so3, quat_to_rotmat
from ..common.types import RobotState
from ..gait.biped_schedule import BipedGaitParams, BipedSchedule, SegKind, make_biped_schedule
from ..gait.swing import swing_trajectory
from ..model_dynamics.robot_model import RobotModelBase
from ..planner.dcm import DcmPlanner
from .contact_wrench import FootContactSpec
from .dcm_feedback import clamp_cop, dcm_feedback_cop

__all__ = [
    "StabilizerParams",
    "BipedReference",
    "BipedPlannerLayer",
    "com_wrench_from_dcm",
    "support_rectangle",
]

_G_W = np.array([0.0, 0.0, -9.81])


@dataclass
class StabilizerParams:
    """! @brief DCM 安定化と全身制御の共通ゲイン（configs/biped.yaml の stabilizer 節）。

    @param k_dcm        DCM フィードバックゲイン [-]（閉ループ極 -k omega。k > 0 で安定）
    @param kp_com_z     重心高さ P ゲイン [1/s^2]
    @param kd_com_z     重心高さ D ゲイン [1/s]
    @param kp_rot       胴体姿勢 P ゲイン [1/s^2]（IHMC 型: 胴体角加速度タスクで使う）
    @param kd_rot       胴体角速度 D ゲイン [1/s]
    @param kp_rot_wrench 重心レンチのモーメント成分に入れる胴体姿勢 P ゲイン [1/s^2]
    @param kd_rot_wrench 同 D ゲイン [1/s]
    @param tau_rot_max  重心レンチのモーメント成分の上限 [N*m]。重心まわりのモーメントは
                        CoP を動かすことでしか実現できないので、大きいと DCM 追従則の
                        CoP 指令と衝突して転倒する（実測: 130 N*m の要求で CoP が 0.4 m 飛んだ）
    @param kp_foot      遊脚足先位置 P ゲイン [1/s^2]（IHMC: 加速度、DLR: 剛性 [N/m] に m_foot_eq を掛ける）
    @param kd_foot      遊脚足先速度 D ゲイン [1/s]
    @param kp_foot_rot  足裏姿勢 P ゲイン [1/s^2]
    @param kd_foot_rot  足裏角速度 D ゲイン [1/s]
    @param kp_stance_foot 立脚の足先位置保持 P ゲイン [1/s^2]（着地位置にラッチした目標へ戻す）
    @param kd_stance_foot 同 D ゲイン [1/s]
    @param kp_stance_rot  立脚の足裏姿勢保持 P ゲイン [1/s^2]（足裏を床に平らに保つ）
    @param kd_stance_rot  同 D ゲイン [1/s]
    @note 立脚を「加速度ゼロ」だけで扱うと、着地衝撃で傾いた足裏がそのまま残り、
          エッジで転がって横滑りする（実測: 立脚が 0.25 s で 10 cm 外へ流れた）。
          Mesesan 2019 の「stance foot stabilization」（貢献 iii）と同じ処置が要る。
    @param kp_posture   関節姿勢 P ゲイン [1/s^2]
    @param kd_posture   関節姿勢 D ゲイン [1/s]
    @param early_contact_ratio 遊脚進捗がこれを超えて接地したら立脚扱いにする [-]
    @param early_contact_load  早期接地した足に与える荷重係数 [-]
    @param cop_margin   CoP の clamp を足裏の縁から内側に取る余裕 [m]
    @param load_bound_ramp 荷重係数 lambda がこの値に達するまでに法線力の上限を 0 -> f_max に
                        立ち上げる [-]。lambda そのものを上限に掛けると両脚支持期の大半で
                        重心レンチが実現不能になり（両足の上限の和が体重を下回る）、QP が
                        妥協した解で転倒する。lambda は公称配分（ソフト）にだけ使い、
                        上限（ハード）は踏み替えの前後 load_bound_ramp の区間だけ絞る
    """

    k_dcm: float = 2.0
    kp_com_z: float = 100.0
    kd_com_z: float = 20.0
    kp_rot: float = 200.0
    kd_rot: float = 30.0
    kp_rot_wrench: float = 20.0
    kd_rot_wrench: float = 4.0
    tau_rot_max: float = 20.0
    kp_foot: float = 400.0
    kd_foot: float = 40.0
    kp_foot_rot: float = 200.0
    kd_foot_rot: float = 20.0
    kp_stance_foot: float = 100.0
    kd_stance_foot: float = 20.0
    kp_stance_rot: float = 100.0
    kd_stance_rot: float = 10.0
    kp_posture: float = 50.0
    kd_posture: float = 5.0
    early_contact_ratio: float = 0.7
    early_contact_load: float = 0.3
    cop_margin: float = 0.01
    load_bound_ramp: float = 0.3
    extra: dict = field(default_factory=dict)

    def bound_factor(self, load: np.ndarray) -> np.ndarray:
        """! @brief 荷重係数から法線力上限の係数を作る。
        @param load 荷重係数 shape (nf,) [-]
        @return 係数 shape (nf,) [-] in [0,1]
        """
        r = max(float(self.load_bound_ramp), 1.0e-6)
        return np.clip(np.asarray(load, dtype=np.float64) / r, 0.0, 1.0)

    @staticmethod
    def from_config(name: str = "biped.yaml", section: str = "stabilizer") -> "StabilizerParams":
        """! @brief configs/biped.yaml から読む。
        @param name    ファイル名 [-]
        @param section 節名 [-]
        @return StabilizerParams（節が無ければ既定値）
        """
        d = dict(load_yaml(name).get(section, {}) or {})
        p = StabilizerParams()
        for k in list(vars(p)):
            if k in d and k != "extra":
                setattr(p, k, type(getattr(p, k))(d[k]))
        p.extra = {k: v for k, v in d.items() if not hasattr(p, k)}
        return p


@dataclass
class BipedReference:
    """! @brief 時刻 t における参照量（計画層の出力）。

    @param t            時刻 [s]
    @param kind         区間の種類 [-]
    @param c_ref        参照重心位置 {W} shape (3,) [m]（z は z_c）
    @param dc_ref       参照重心速度 shape (3,) [m/s]
    @param ddc_ref      参照重心加速度 shape (3,) [m/s^2]（LIPM: omega^2 (c - p)）
    @param xi_ref       参照 DCM 水平 shape (2,) [m]
    @param xi_dot_ref   参照 DCM 速度 shape (2,) [m/s]
    @param cop_sched    スケジュールの CoP shape (2,) [m]
    @param p_foot       足先目標位置 {W} shape (nf,3) [m]
    @param v_foot       足先目標速度 shape (nf,3) [m/s]
    @param a_foot       足先目標加速度 shape (nf,3) [m/s^2]
    @param R_foot       足裏目標姿勢 shape (nf,3,3) [-]
    @param contact      立脚予定 shape (nf,) [bool]
    @param load         荷重係数 shape (nf,) [-] in [0,1]
    @param swing_ratio  遊脚進捗 shape (nf,) [-]
    @param q_posture    関節姿勢の参照 shape (n,) [rad]
    @param R_base       胴体目標姿勢 shape (3,3) [-]
    """

    t: float
    kind: SegKind
    c_ref: np.ndarray
    dc_ref: np.ndarray
    ddc_ref: np.ndarray
    xi_ref: np.ndarray
    xi_dot_ref: np.ndarray
    cop_sched: np.ndarray
    p_foot: np.ndarray
    v_foot: np.ndarray
    a_foot: np.ndarray
    R_foot: np.ndarray
    contact: np.ndarray
    load: np.ndarray
    swing_ratio: np.ndarray
    q_posture: np.ndarray
    R_base: np.ndarray


class BipedPlannerLayer:
    """! @brief スケジュール + DCM 閉形式軌道から参照を出す層（`planner.dcm` と `gait.biped_schedule` を使う）。"""

    def __init__(
        self,
        model: RobotModelBase,
        rcfg: RobotConfig,
        params: BipedGaitParams,
        dt: float,
    ) -> None:
        """! @brief 全区間の計画をオフラインで作る。
        @param model  RobotModelBase [-]
        @param rcfg   RobotConfig [-]
        @param params BipedGaitParams [-]
        @param dt     参照の時間刻み（= 制御周期）[s]
        @return なし
        @note z_c は公称姿勢の実測重心高さ（h_nominal は pelvis 高さなので別物）。
        """
        self.model = model
        self.rcfg = rcfg
        self.params = params
        self.dt = float(dt)
        q_nom = model.make_q(
            np.array([0.0, 0.0, rcfg.h_nominal]), np.array([1.0, 0.0, 0.0, 0.0]), rcfg.q_nominal
        )
        self.q_nominal = np.asarray(rcfg.q_nominal, dtype=np.float64).copy()
        foot_init = model.fk(q_nom).copy()
        foot_init[:, 2] = params.foot_z
        self.schedule: BipedSchedule = make_biped_schedule(params, foot_init=foot_init)
        c_nom = model.com(q_nom)
        self.z_c = float(c_nom[2])
        self.planner = DcmPlanner(z_c=self.z_c)
        self.omega = float(self.planner.omega)
        sch = self.schedule
        self.plan_x = self.planner.plan(sch.seg_cop[:, 0], sch.seg_dur, self.dt, c0=float(c_nom[0]))
        self.plan_y = self.planner.plan(sch.seg_cop[:, 1], sch.seg_dur, self.dt, c0=float(c_nom[1]))
        self.n_t = int(min(self.plan_x.t.size, self.plan_y.t.size))
        self._ds_blocks = self._find_ds_blocks()

    @property
    def total_time(self) -> float:
        """! @brief 計画の全体時間 [s]"""
        return float(self.schedule.total_time)

    def _find_ds_blocks(self) -> list[tuple[int, int, int, int]]:
        """! @brief 連続する DS 区間をブロックにまとめ、着地側・離陸側の足を決める。
        @return [(開始区間, 終了区間+1, 着地側の足 or -1, 離陸側の足 or -1)]
        """
        sch = self.schedule
        kinds = sch.seg_kind
        blocks = []
        i = 0
        n = len(kinds)
        while i < n:
            if kinds[i] is not SegKind.DS:
                i += 1
                continue
            j = i
            while j < n and kinds[j] is SegKind.DS:
                j += 1
            arriving = int(sch.seg_swing[i - 1]) if i > 0 and kinds[i - 1] is SegKind.SS else -1
            departing = int(sch.seg_swing[j]) if j < n and kinds[j] is SegKind.SS else -1
            blocks.append((i, j, arriving, departing))
            i = j
        return blocks

    def load_factors(self, t: float) -> np.ndarray:
        """! @brief 荷重係数 lambda_i(t) を返す。
        @param t 時刻 [s]
        @return load shape (nf,) [-]
        """
        sch = self.schedule
        nf = sch.foot_pos.shape[1]
        i = sch.segment_at(t)
        kind = sch.seg_kind[i]
        lam = np.ones(nf, dtype=np.float64)
        if kind is SegKind.SS:
            lam[int(sch.seg_swing[i])] = 0.0
            return lam
        if kind is SegKind.STAND and i > 0 and sch.seg_kind[i - 1] is SegKind.SS:
            # 最後の 1 歩の直後は DS が無いので、着地した足を t_ds で 0 -> 1 に立ち上げる
            landed = int(sch.seg_swing[i - 1])
            u = float(np.clip((t - sch.seg_t0[i]) / max(self.params.t_ds, 1.0e-9), 0.0, 1.0))
            lam[landed] = u
            return lam
        if kind is SegKind.DS:
            for (a, b, arriving, departing) in self._ds_blocks:
                if a <= i < b:
                    t0 = float(sch.seg_t0[a])
                    T = float(sch.seg_t0[b - 1] + sch.seg_dur[b - 1] - t0)
                    u = float(np.clip((t - t0) / max(T, 1.0e-9), 0.0, 1.0))
                    if arriving >= 0:
                        lam[arriving] = u
                    if departing >= 0:
                        lam[departing] = 1.0 - u
                    break
        return lam

    def reference_at(self, k: int) -> BipedReference:
        """! @brief k 番目の制御周期（t = k dt）の参照を返す。
        @param k 周期番号 [-]（範囲外は端で止める）
        @return BipedReference
        """
        sch = self.schedule
        k = int(np.clip(k, 0, self.n_t - 1))
        t = float(self.plan_x.t[k])
        i = sch.segment_at(t)
        kind = sch.seg_kind[i]
        nf = sch.foot_pos.shape[1]

        p_foot = sch.foot_pos[i].copy()
        v_foot = np.zeros((nf, 3))
        a_foot = np.zeros((nf, 3))
        contact = np.ones(nf, dtype=bool)
        swing_ratio = np.zeros(nf)
        if kind is SegKind.SS:
            sw = int(sch.seg_swing[i])
            u = float(np.clip((t - sch.seg_t0[i]) / sch.seg_dur[i], 0.0, 1.0))
            p, v, a = swing_trajectory(
                u, sch.swing_from[i], sch.swing_to[i],
                self.params.swing_height, float(sch.seg_dur[i]),
            )
            p_foot[sw], v_foot[sw], a_foot[sw] = p, v, a
            contact[sw] = False
            swing_ratio[sw] = u

        cop = sch.seg_cop[i]
        c_xy = np.array([self.plan_x.c[k], self.plan_y.c[k]])
        dc_xy = np.array([self.plan_x.c_dot[k], self.plan_y.c_dot[k]])
        ddc_xy = (self.omega ** 2) * (c_xy - np.array([self.plan_x.p[k], self.plan_y.p[k]]))
        return BipedReference(
            t=t,
            kind=kind,
            c_ref=np.array([c_xy[0], c_xy[1], self.z_c]),
            dc_ref=np.array([dc_xy[0], dc_xy[1], 0.0]),
            ddc_ref=np.array([ddc_xy[0], ddc_xy[1], 0.0]),
            xi_ref=np.array([self.plan_x.xi[k], self.plan_y.xi[k]]),
            xi_dot_ref=np.array([self.plan_x.xi_dot[k], self.plan_y.xi_dot[k]]),
            cop_sched=np.asarray(cop, dtype=np.float64).copy(),
            p_foot=p_foot,
            v_foot=v_foot,
            a_foot=a_foot,
            R_foot=np.array([np.eye(3)] * nf),
            contact=contact,
            load=self.load_factors(t),
            swing_ratio=swing_ratio,
            q_posture=self.q_nominal,
            R_base=np.eye(3),
        )


def support_rectangle(
    p_feet: np.ndarray, load: np.ndarray, spec: FootContactSpec, margin: float
) -> tuple[np.ndarray, np.ndarray]:
    """! @brief 荷重のある足の足裏矩形の和集合の外接矩形（CoP の clamp 用）。
    @param p_feet 足先 site 位置 {W} shape (nf,3) [m]
    @param load   荷重係数 shape (nf,) [-]
    @param spec   FootContactSpec [-]
    @param margin 縁からの余裕 [m]
    @return (中心 shape (2,) [m], 半幅 shape (2,) [m])
    @note 足裏のヨーは 0 と仮定する（本実装の歩行は直進のみ）。
          外接矩形は両脚支持で足の間の空隙も含むが、両足に荷重が乗っている間は
          全体の CoP がそこにあっても個別の足の CoP は足裏内に置けるので問題ない。
    """
    act = np.flatnonzero(np.asarray(load) > 1.0e-3)
    if act.size == 0:
        act = np.arange(p_feet.shape[0])
    half = np.array([spec.half_length, spec.half_width]) - margin
    lo = np.asarray(p_feet)[act, 0:2].min(axis=0) - half
    hi = np.asarray(p_feet)[act, 0:2].max(axis=0) + half
    return 0.5 * (lo + hi), 0.5 * (hi - lo)


def com_wrench_from_dcm(
    model: RobotModelBase,
    state: RobotState,
    ref: BipedReference,
    par: StabilizerParams,
    omega: float,
    spec: FootContactSpec,
    f_ext: np.ndarray | None = None,
    dcop: np.ndarray | None = None,
    dmom: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """! @brief DCM 追従則から、重心に要求するレンチ W_des を作る（安定化層）。
    @param model RobotModelBase [-]
    @param state 推定状態 [-]
    @param ref   参照 [-]
    @param par   StabilizerParams [-]
    @param omega LIPM 固有振動数 [rad/s]
    @param spec  FootContactSpec [-]
    @param f_ext 手などが環境から受ける外力の合計 {W} shape (3,) [N]（None で 0。腕のある機体で手先タスクを使うとき）
    @param dcop  CoP 指令 p_cmd に足す残差 {W} shape (2,) [m]（None で 0。学習方策の安定化層残差）。支持矩形の clamp の前に足す
    @param dmom  重心まわりモーメントに足す残差 {W} shape (3,) [N*m]（None で 0）。tau_rot_max の上限の後に足す
    @return (W_des shape (6,) [N, N*m], cop_cmd shape (2,) [m], ddc_des shape (3,) [m/s^2],
             xi_hat shape (2,) [m])
    @note 外力 f_ext があると、足裏の CoP は DCM 追従則の p_cmd から `f_ext,xy / (m omega^2)` だけずれる
          （水平の釣り合い: m ddc_xy = F_feet,xy + f_ext,xy）。ずれた先の足裏 CoP を支持矩形に clamp し、
          clamp 後から p_cmd を逆算する。押し負けの限界はここで決まる（CoP が縁に張り付くと DCM 追従を
          あきらめる）。返す cop_cmd は足裏 CoP（F/T の実測と比べる量）。
    @note 水平: `p_cmd = xi_hat + k (xi_hat - xi_ref) - xi_dot_ref / omega` を支持矩形に
          clamp し、LIPM で `ddc_xy = omega^2 (c - p_cmd)`。
          鉛直: 高さの PD。姿勢: SO(3) 対数写像の PD を重心まわり慣性で掛ける。
          力は `F = m (ddc_des - g)`、すなわち重力補償込み。
    """
    c = np.asarray(state.c_W, dtype=np.float64)
    dc = np.asarray(state.dc_W, dtype=np.float64)
    xi_hat = c[0:2] + dc[0:2] / omega
    p_cmd = dcm_feedback_cop(xi_hat, ref.xi_ref, ref.xi_dot_ref, omega, par.k_dcm)
    center, half = support_rectangle(state.p_WF, ref.load, spec, par.cop_margin)
    shift = np.zeros(2)
    if f_ext is not None:
        shift = np.asarray(f_ext, dtype=np.float64).reshape(3)[0:2] / (model.total_mass * omega ** 2)
    if dcop is not None:
        # 残差は CoP 指令（= 重心加速度を決める CMP）に足す。足裏 CoP もその分ずれ、同じ clamp に掛かる
        p_cmd = p_cmd + np.asarray(dcop, dtype=np.float64).reshape(2)
    p_feet, _ = clamp_cop(p_cmd + shift, center, half)   # 足裏 CoP を clamp
    p_cmd = p_feet - shift

    ddc = np.empty(3)
    ddc[0:2] = (omega ** 2) * (c[0:2] - p_cmd)
    ddc[2] = ref.ddc_ref[2] + par.kp_com_z * (ref.c_ref[2] - c[2]) + par.kd_com_z * (ref.dc_ref[2] - dc[2])
    F = model.total_mass * (ddc - _G_W)

    R = quat_to_rotmat(state.quat_WB)
    w_W = R @ np.asarray(state.w_B, dtype=np.float64)
    e_R = log_so3(ref.R_base @ R.T)
    alpha = par.kp_rot_wrench * e_R - par.kd_rot_wrench * w_W
    q = model.make_q(state.p_WB, state.quat_WB, state.q)
    I_W = model.Ag(q)[3:6, 3:6] @ R.T
    tau = I_W @ alpha
    # LIPM の仮定は「重心まわりの角運動量変化率ゼロ」。姿勢の立て直しに使うモーメントは
    # 上限で縛り、CoP 指令（= 並進力の向き）を壊さない範囲に留める。
    n = float(np.linalg.norm(tau))
    if n > par.tau_rot_max:
        tau *= par.tau_rot_max / n
    if dmom is not None:
        tau = tau + np.asarray(dmom, dtype=np.float64).reshape(3)
    return np.concatenate([F, tau]), p_feet, ddc, xi_hat
