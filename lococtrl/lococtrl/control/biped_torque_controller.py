"""! @file biped_torque_controller.py
@brief 二足のトルク制御スタックの統合。

1 制御周期の流れ:

    BipedPlannerLayer.reference_at(k) ----> 参照（重心・DCM・足先・荷重係数）
    接触の判定（予定 + 早期接地）
    com_wrench_from_dcm ------------------> 重心に要求するレンチ W_des と CoP 指令
    torque mapper（ihmc / dlr）-----------> 関節トルク tau
    residual policy（任意、rl）-----------> tau += delta_tau
    ControlOutput(tau, kp = 0, kd = 0) ---> シミュレータの関節 PD は使わない（純トルク制御）

位置制御の `run_biped.run_biped_walk` との違いは、IK と関節 PD の代わりに全身制御器が
トルクを直接決めること、CoP 指令が接触レンチの制約として実現されること、の 2 点である。

記号: nf 脚数 [-]、n 駆動関節数 [-]。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from ..common.config import RobotConfig
from ..common.types import ControlOutput, RobotState
from ..model_dynamics.robot_model import RobotModelBase
from .biped_stack import BipedPlannerLayer, BipedReference, StabilizerParams, com_wrench_from_dcm
from .contact_wrench import FootContactSpec

__all__ = ["IBipedTorqueMapper", "IResidualPolicy", "CycleInfo", "ExternalResidual", "BipedTorqueController"]


class IBipedTorqueMapper(Protocol):
    """! @brief 重心レンチ・参照から関節トルクを作る層（ihmc / dlr）。"""

    name: str
    last_solve_time: float
    last_status: str
    last_wrench: np.ndarray

    def solve(
        self,
        state: RobotState,
        ref: BipedReference,
        wrench_des: np.ndarray,
        contact: np.ndarray,
        load: np.ndarray,
    ) -> np.ndarray:  # pragma: no cover - Protocol
        """! @brief 関節トルク shape (n,) [N*m] を返す。"""
        ...

    def reset(self) -> None:  # pragma: no cover - Protocol
        """! @brief 内部状態を初期化する。"""
        ...


class IResidualPolicy(Protocol):
    """! @brief 学習 LL 方策（残差トルク）の口。"""

    def act(self, obs: np.ndarray) -> np.ndarray:  # pragma: no cover - Protocol
        """! @brief 観測 -> 残差トルク shape (n,) [N*m]"""
        ...


@dataclass
class CycleInfo:
    """! @brief 1 周期の内部量（ログ・観測・報酬に使う）。
    @param ref        参照 [-]
    @param contact    立脚として扱った足 shape (nf,) [bool]
    @param load       荷重係数 shape (nf,) [-]
    @param wrench_des 重心に要求したレンチ shape (6,) [N, N*m]
    @param cop_cmd    CoP 指令 shape (2,) [m]
    @param xi_hat     推定 DCM shape (2,) [m]
    @param ddc_des    目標重心加速度 shape (3,) [m/s^2]
    @param tau_base   全身制御器のトルク shape (n,) [N*m]
    @param tau_res    残差トルク shape (n,) [N*m]
    @param early      早期接地を検出した足 shape (nf,) [bool]
    """

    ref: BipedReference
    contact: np.ndarray
    load: np.ndarray
    wrench_des: np.ndarray
    cop_cmd: np.ndarray
    xi_hat: np.ndarray
    ddc_des: np.ndarray
    tau_base: np.ndarray
    tau_res: np.ndarray
    early: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=bool))
    dcop: np.ndarray = field(default_factory=lambda: np.zeros(2))
    dmom: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class ExternalResidual:
    """! @brief 外（上位の学習方策など）から書き込む残差。各層の解析則の出力に足す。
    @param dcop    安定化層: CoP 指令の残差 {W} shape (2,) [m]（支持矩形の clamp の前）
    @param dmom    安定化層: 重心まわりモーメントの残差 {W} shape (3,) [N*m]
    @param tau_leg 全身制御: 脚関節トルクの残差 shape (n_leg_joint,) [N*m]（関節の並びは脚 → 腕の先頭）
    @note 全部 0 のときは解析則だけと同じ結果になる（足し算を省くので数値も一致）。
          保持は呼び出し側の責任（方策の判断周期のあいだ同じ値を置いておく）。
    """

    dcop: np.ndarray = field(default_factory=lambda: np.zeros(2))
    dmom: np.ndarray = field(default_factory=lambda: np.zeros(3))
    tau_leg: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def is_zero(self) -> bool:
        """! @brief 全成分が 0 か。"""
        return not (np.any(self.dcop) or np.any(self.dmom) or np.any(self.tau_leg))


class BipedTorqueController:
    """! @brief 計画層 + DCM 安定化 + 全身制御器（+ 残差方策）を束ねる。"""

    def __init__(
        self,
        model: RobotModelBase,
        rcfg: RobotConfig,
        planner: BipedPlannerLayer,
        par: StabilizerParams,
        mapper: IBipedTorqueMapper,
        residual: IResidualPolicy | None = None,
        residual_obs_fn=None,
    ) -> None:
        """! @brief 構築する。
        @param model          RobotModelBase [-]
        @param rcfg           RobotConfig [-]
        @param planner        BipedPlannerLayer [-]
        @param par            StabilizerParams [-]
        @param mapper         全身制御器（ihmc / dlr）[-]
        @param residual       残差方策（None で無し）[-]
        @param residual_obs_fn 観測を作る関数 (state, info, omega) -> obs（residual があるとき必須）
        @return なし
        """
        self.model = model
        self.rcfg = rcfg
        self.planner = planner
        self.par = par
        self.mapper = mapper
        self.residual = residual
        self.residual_obs_fn = residual_obs_fn
        self.spec = FootContactSpec.from_config(rcfg)
        self.n = model.n_joint
        self.nf = model.n_leg
        self.last_info: CycleInfo | None = None
        self.last_obs: np.ndarray | None = None
        ## 立脚の足先目標（着地時の実位置にラッチ）shape (nf,3) [m]
        self._p_latch = np.full((self.nf, 3), np.nan)
        self._prev_contact = np.zeros(self.nf, dtype=bool)
        ## 外から書き込む残差（None で無し）。`set_external_residual` で設定する
        self.ext_residual: ExternalResidual | None = None

    def set_external_residual(self, dcop=None, dmom=None, tau_leg=None) -> None:
        """! @brief 安定化層と全身制御に足す残差を設定する（次の compute から効く。値は保持される）。
        @param dcop    CoP 指令の残差 {W} shape (2,) [m]（None で 0）
        @param dmom    重心まわりモーメントの残差 {W} shape (3,) [N*m]（None で 0）
        @param tau_leg 脚関節トルクの残差 shape (n_leg_joint,) [N*m]（None で 0）
        @return なし
        """
        n_leg = int(getattr(self.rcfg, "n_leg_joint", self.n))
        r = ExternalResidual(
            dcop=np.zeros(2) if dcop is None else np.asarray(dcop, dtype=np.float64).reshape(2).copy(),
            dmom=np.zeros(3) if dmom is None else np.asarray(dmom, dtype=np.float64).reshape(3).copy(),
            tau_leg=np.zeros(n_leg) if tau_leg is None else np.asarray(tau_leg, dtype=np.float64).reshape(n_leg).copy(),
        )
        self.ext_residual = None if r.is_zero() else r

    @property
    def omega(self) -> float:
        """! @brief LIPM 固有振動数 [rad/s]"""
        return self.planner.omega

    def reset(self) -> None:
        """! @brief 内部状態を初期化する。
        @return なし
        """
        self.mapper.reset()
        if hasattr(self.planner, "reset"):
            self.planner.reset()
        self.last_info = None
        self.last_obs = None
        self._p_latch[:] = np.nan
        self._prev_contact[:] = False
        self.ext_residual = None

    def effective_contact(self, state: RobotState, ref: BipedReference) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """! @brief 予定接触と実測接触から、立脚扱いの足と荷重係数を決める。
        @param state 推定状態 [-]
        @param ref   参照 [-]
        @return (contact shape (nf,) [bool], load shape (nf,) [-], early shape (nf,) [bool])
        @note 遊脚の後半で接地した足はその周期から立脚扱いにする。
              荷重係数は early_contact_load を下限に与え、着地衝撃でレンチが飛ばないようにする。
        """
        meas = np.asarray(state.contact, dtype=bool)
        early = (~ref.contact) & meas & (ref.swing_ratio > self.par.early_contact_ratio)
        contact = ref.contact | early
        load = np.asarray(ref.load, dtype=np.float64).copy()
        load[early] = np.maximum(load[early], self.par.early_contact_load)
        load[~contact] = 0.0
        return contact, load, early

    def compute(self, state: RobotState, k: int) -> ControlOutput:
        """! @brief 1 制御周期ぶんの関節指令を計算する。
        @param state 推定状態 [-]
        @param k     制御周期番号 [-]（参照は t = k dt）
        @return ControlOutput（kp = kd = 0 の純トルク指令）
        """
        # オンライン計画器は状態が要る（update）。オフライン計画は reference_at
        ref = self.planner.update(state, k) if hasattr(self.planner, "update") else self.planner.reference_at(k)
        contact, load, early = self.effective_contact(state, ref)
        # 立脚の足先目標は「着地した瞬間の実位置（z は地面）」にラッチする。
        # 計画上の着地位置に引き戻すと、数 cm の着地誤差ぶん床を引きずって滑る。
        for i in range(self.nf):
            if contact[i] and (not self._prev_contact[i] or not np.all(np.isfinite(self._p_latch[i]))):
                self._p_latch[i] = np.asarray(state.p_WF[i], dtype=np.float64)
                self._p_latch[i, 2] = float(ref.p_foot[i, 2])
            if contact[i]:
                ref.p_foot[i] = self._p_latch[i]
                ref.v_foot[i] = 0.0
                ref.a_foot[i] = 0.0
        self._prev_contact = contact.copy()
        f_ext = None
        ht = getattr(self.mapper, "hand_task", None)
        wh = np.asarray(getattr(state, "wrench_hands", np.zeros((0, 6))), dtype=np.float64)
        if ht is not None and ht.comp_f_env and wh.ndim == 2 and wh.shape[0] > 0:
            f_ext = wh[:, 0:3].sum(axis=0)
        er = self.ext_residual
        wrench_des, cop_cmd, ddc_des, xi_hat = com_wrench_from_dcm(
            self.model, state, ref, self.par, self.omega, self.spec, f_ext=f_ext,
            dcop=None if er is None or not np.any(er.dcop) else er.dcop,
            dmom=None if er is None or not np.any(er.dmom) else er.dmom,
        )
        tau_base = self.mapper.solve(state, ref, wrench_des, contact, load)
        info = CycleInfo(
            ref=ref, contact=contact, load=load, wrench_des=wrench_des, cop_cmd=cop_cmd,
            xi_hat=xi_hat, ddc_des=ddc_des, tau_base=tau_base, tau_res=np.zeros(self.n), early=early,
        )
        tau = tau_base.copy()
        if er is not None:
            info.dcop = er.dcop.copy()
            info.dmom = er.dmom.copy()
            if np.any(er.tau_leg):
                nl = er.tau_leg.size
                info.tau_res[0:nl] += er.tau_leg
                tau[0:nl] += er.tau_leg
        if self.residual is not None:
            if self.residual_obs_fn is None:
                raise ValueError("residual を使うときは residual_obs_fn が必要")
            obs = self.residual_obs_fn(state, info, self.omega)
            self.last_obs = obs
            tau_res = np.asarray(self.residual.act(obs), dtype=np.float64).reshape(self.n)
            info.tau_res = info.tau_res + tau_res
            tau = tau + tau_res
        tau = np.clip(tau, -self.rcfg.tau_max, self.rcfg.tau_max)
        self.last_info = info
        u = ControlOutput.zeros(self.n)
        u.tau = tau
        u.q_des = np.asarray(state.q, dtype=np.float64).copy()
        u.dq_des = np.zeros(self.n)
        u.kp = np.zeros(self.n)
        u.kd = np.zeros(self.n)
        return u
