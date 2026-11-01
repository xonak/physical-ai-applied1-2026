"""! @file types.py
@brief 層をまたいで受け渡すデータ型の定義。

フィールド名と単位はここが唯一の正である。
配列はすべて np.ndarray / dtype=np.float64（bool 配列を除く）。

記号:
- {W} ワールド, {B} ベース, {H} 水平系（ヨーのみ）
- n  : 駆動関節数 [-]（G1 の脚 12、箱押し用は脚 12 + 腕 14）
- nf : 脚数 [-]（二足 2）
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "RobotState",
    "GaitState",
    "ControlOutput",
]


@dataclass
class RobotState:
    """! @brief 制御器が参照するロボット状態（推定器の出力）。

    @param p_WB   ベース位置 {W} shape (3,) [m]
    @param quat_WB ベース姿勢 (w,x,y,z) shape (4,) [-]
    @param v_B    ベース並進速度（ボディ表現） shape (3,) [m/s]
    @param w_B    ベース角速度（ボディ表現） shape (3,) [rad/s]
    @param q      関節角 shape (n,) [rad]
    @param dq     関節角速度 shape (n,) [rad/s]
    @param p_WF   足先位置 {W} shape (nf,3) [m]
    @param contact 接触フラグ shape (nf,) [bool]
    @param f_meas 足先接触力 {W} shape (nf,3) [N]
    @param c_W    重心位置 {W} shape (3,) [m]
    @param dc_W   重心速度 {W} shape (3,) [m/s]
    @param t      シミュレーション時刻 [s]
    @param tau_meas    関節トルクセンサの値 shape (n,) [N*m]（無いときは 0）
    @param wrench_feet 足裏 F/T センサから推定した床反力レンチ {W} shape (nf,6)
                       [N, N*m]。上 3 = 力、下 3 = 足先 site（足裏中心）まわりのモーメント
                       （無いときは 0）
    @param wrench_hands 手首 F/T から推定した手の接触レンチ {W} shape (n_arm,6) [N, N*m]。
                       環境が手に加える力。上 3 = 力、下 3 = 手先 site まわりのモーメント
                       （箱押し用モデルのみ。無いときは shape (0,6)）
    @param box_pose    箱の位置・姿勢・並進速度 shape (10,) [m, -, m/s]（p 3, quat 4, v 3。箱が無ければ shape (0,)）
    @param p_hands     手先 site の位置 {W} shape (n_arm,3) [m]（腕のある機体のみ。無ければ shape (0,3)）
    @note 速度のボディ表現は Pinocchio 既定に合わせる。{W} 表現が必要な箇所では
          明示的に R(quat_WB) を掛けること。
    @note tau_meas / wrench_feet は既定値つきの末尾の項目で、センサの無いモデルでは 0 のままになる。
    """

    p_WB: np.ndarray
    quat_WB: np.ndarray
    v_B: np.ndarray
    w_B: np.ndarray
    q: np.ndarray
    dq: np.ndarray
    p_WF: np.ndarray
    contact: np.ndarray
    f_meas: np.ndarray
    c_W: np.ndarray
    dc_W: np.ndarray
    t: float
    tau_meas: np.ndarray = field(default_factory=lambda: np.zeros(0))
    wrench_feet: np.ndarray = field(default_factory=lambda: np.zeros((0, 6)))
    wrench_hands: np.ndarray = field(default_factory=lambda: np.zeros((0, 6)))
    box_pose: np.ndarray = field(default_factory=lambda: np.zeros(0))
    p_hands: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))

    @staticmethod
    def zeros(n_joint: int, n_leg: int) -> "RobotState":
        """! @brief 全要素ゼロの状態を作る（初期化・テスト用）。
        @param n_joint 駆動関節数 n [-]
        @param n_leg   脚数 nf [-]
        @return RobotState
        """
        return RobotState(
            p_WB=np.zeros(3),
            quat_WB=np.array([1.0, 0.0, 0.0, 0.0]),
            v_B=np.zeros(3),
            w_B=np.zeros(3),
            q=np.zeros(n_joint),
            dq=np.zeros(n_joint),
            p_WF=np.zeros((n_leg, 3)),
            contact=np.zeros(n_leg, dtype=bool),
            f_meas=np.zeros((n_leg, 3)),
            c_W=np.zeros(3),
            dc_W=np.zeros(3),
            t=0.0,
            tau_meas=np.zeros(n_joint),
            wrench_feet=np.zeros((n_leg, 6)),
        )

    def copy(self) -> "RobotState":
        """! @brief 深いコピーを返す。
        @return RobotState
        """
        return RobotState(
            p_WB=self.p_WB.copy(),
            quat_WB=self.quat_WB.copy(),
            v_B=self.v_B.copy(),
            w_B=self.w_B.copy(),
            q=self.q.copy(),
            dq=self.dq.copy(),
            p_WF=self.p_WF.copy(),
            contact=self.contact.copy(),
            f_meas=self.f_meas.copy(),
            c_W=self.c_W.copy(),
            dc_W=self.dc_W.copy(),
            t=float(self.t),
            tau_meas=np.asarray(self.tau_meas).copy(),
            wrench_feet=np.asarray(self.wrench_feet).copy(),
        )


@dataclass
class GaitState:
    """! @brief 歩容スケジューラの出力。

    @param phase         全体位相 [-] in [0,1)
    @param leg_phase     各脚の位相 shape (nf,) [-] in [0,1)
    @param contact       接触予定フラグ shape (nf,) [bool]
    @param swing_ratio   遊脚区間内の進捗 shape (nf,) [-] in [0,1]（立脚中は 0）
    @param contact_table MPC ホライズン分の接触表 shape (N,nf) [bool]
    @param t_remain      現区間の残り時間 shape (nf,) [s]
    """

    phase: float
    leg_phase: np.ndarray
    contact: np.ndarray
    swing_ratio: np.ndarray
    contact_table: np.ndarray
    t_remain: np.ndarray


@dataclass
class ControlOutput:
    """! @brief 制御器の出力。実効トルクは tau_j = tau + kp*(q_des-q) + kd*(dq_des-dq)。

    @param tau    フィードフォワードトルク shape (n,) [N*m]
    @param q_des  関節目標角 shape (n,) [rad]
    @param dq_des 関節目標角速度 shape (n,) [rad/s]
    @param kp     関節 P ゲイン shape (n,) [N*m/rad]
    @param kd     関節 D ゲイン shape (n,) [N*m*s/rad]
    @note 立脚はトルク主導 (kp=0, kd 小)、遊脚は位置追従 (kp 大) を想定。
    """

    tau: np.ndarray
    q_des: np.ndarray
    dq_des: np.ndarray
    kp: np.ndarray
    kd: np.ndarray

    @staticmethod
    def zeros(n_joint: int) -> "ControlOutput":
        """! @brief 全要素ゼロの出力を作る。
        @param n_joint 駆動関節数 n [-]
        @return ControlOutput
        """
        return ControlOutput(
            tau=np.zeros(n_joint),
            q_des=np.zeros(n_joint),
            dq_des=np.zeros(n_joint),
            kp=np.zeros(n_joint),
            kd=np.zeros(n_joint),
        )

    def effective_torque(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        """! @brief 関節に実際に加わるトルクを計算する。
        @param q  関節角 shape (n,) [rad]
        @param dq 関節角速度 shape (n,) [rad/s]
        @return tau_j shape (n,) [N*m]
        """
        return self.tau + self.kp * (self.q_des - q) + self.kd * (self.dq_des - dq)
