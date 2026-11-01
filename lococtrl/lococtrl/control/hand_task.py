"""! @file hand_task.py
@brief 手先タスク（腕のある機体の手先インピーダンス）と、手の接触力の全身制御への反映。

受動性ベース全身制御（`wbc_passivity`）に差し込む。遊脚インピーダンスと同じ形で

    F_a = kp (p_des,a - p_a) + kd (v_des,a - v_a) + f_ff,a      （|F_a| <= f_max）
    tau += (J_hand,a^T [F_a; 0])[6:]

を関節トルクに足す。手の位置はベースと腕の関節だけで決まるので、F_a は腕の関節にだけ
トルクを生む。腕の関節姿勢タスクは手先ヤコビアン（腕の列）の零空間に射影する。

手が環境から受ける力 f_env（手首 F/T の実測、`RobotState.wrench_hands`）は
重心の力学に入る: m ddc = sum f_feet + sum f_env + m g。重心に要求するレンチ W_des の
うち手が受け持つ分を差し引いた `W_des - sum_a G_a f_env,a` を足に配分する
（Mesesan 2019 の相互作用レンチの扱い）。また DCM 追従則では、足裏 CoP が
`p_cmd + f_env,xy / (m omega^2)` にずれるので、ずれた先を支持矩形に clamp する
（`biped_stack.com_wrench_from_dcm` の f_ext）。

手先の目標 p_des / f_ff は「どこを押すか」を知っている上位（箱押しなら
armtasks/boxpush のフック）が毎周期書き換える。本モジュールは目標の置き場所と計算だけを持つ。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["HandTask"]


@dataclass
class HandTask:
    """! @brief 手先インピーダンスの目標と係数（腕ごと）。
    @param n_arm   腕の本数 [-]
    @param active  タスクを有効にする腕 shape (n_arm,) [bool]
    @param p_des   手先目標位置 {W} shape (n_arm,3) [m]
    @param v_des   手先目標速度 {W} shape (n_arm,3) [m/s]
    @param f_ff    フィードフォワード力 {W} shape (n_arm,3) [N]（押付け力の指令。箱の面に向けて与える）
    @param kp      手先の剛性 [N/m]
    @param kd      手先の粘性 [N*s/m]
    @param f_max   手先力の上限 [N]（腕のトルク上限 25 N*m を手先で超えない程度）
    @param comp_f_env 手が受ける力（実測）を足のレンチ配分と CoP 指令に反映するか [-]
    @param null_damping 腕の零空間射影の減衰 [-]
    @param kp_axis 軸ごとの剛性・粘性の倍率 {W} shape (n_arm,3) [-]。押す向きを 0 にすると
                   その軸は f_ff だけの力制御になる（位置と力のハイブリッド）
    @param hold_rot 手先の姿勢も保持するか [-]。位置 3 だけだと腕 7 自由度の残り 4 のうち
                   手首が「最も軽い経路」として曲がり、手首 pitch が 60 deg 以上流れた（実測）。
                   姿勢目標 R_des は最初に有効になった周期の実姿勢にラッチする（None のとき）
    @param kp_rot, kd_rot 姿勢インピーダンス [N*m/rad, N*m*s/rad]（手首は 5 N*m が上限）
    @param m_rot_max 姿勢モーメントの上限 [N*m]
    @param R_des   手先の姿勢目標 shape (n_arm,3,3)（NaN なら未ラッチ）
    @param last_F  直前周期の手先力指令 {W} shape (n_arm,3) [N]（ログ用、出力）
    @note kp 300 N/m, kd 30 N*s/m は手先の等価質量 ~1 kg に対して固有振動数 ~2.8 Hz、
          減衰比 ~0.9。腕のトルク上限（肩 25 N*m、腕長 ~0.4 m → 手先 ~60 N）に対して
          f_max は 60 N を既定にする。
    """

    n_arm: int
    active: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    p_des: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    v_des: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    f_ff: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    kp: float = 300.0
    kd: float = 30.0
    f_max: float = 60.0
    comp_f_env: bool = True
    null_damping: float = 1.0e-3
    kp_axis: np.ndarray = field(default_factory=lambda: np.ones((0, 3)))
    hold_rot: bool = True
    kp_rot: float = 8.0
    kd_rot: float = 0.8
    m_rot_max: float = 4.0
    R_des: np.ndarray = field(default_factory=lambda: np.full((0, 3, 3), np.nan))
    last_F: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))

    def __post_init__(self) -> None:
        n = int(self.n_arm)
        if self.active.size != n:
            self.active = np.zeros(n, dtype=bool)
        for name in ("p_des", "v_des", "f_ff", "last_F"):
            if getattr(self, name).shape != (n, 3):
                setattr(self, name, np.zeros((n, 3)))
        if self.kp_axis.shape != (n, 3):
            self.kp_axis = np.ones((n, 3))
        if self.R_des.shape != (n, 3, 3):
            self.R_des = np.full((n, 3, 3), np.nan)

    def set_target(self, arm: int, p_des, f_ff=None, v_des=None, active: bool = True, kp_axis=None) -> None:
        """! @brief 腕 arm の目標を書く（毎周期呼んでよい）。
        @param arm    腕インデックス [-]
        @param p_des  目標位置 {W} shape (3,) [m]
        @param f_ff   フィードフォワード力 {W} shape (3,) [N]（None で 0）
        @param v_des  目標速度 {W} shape (3,) [m/s]（None で 0）
        @param active 有効 [-]
        @param kp_axis 軸ごとの剛性倍率 shape (3,) [-]（None で全軸 1）
        @return なし
        """
        self.active[arm] = bool(active)
        self.kp_axis[arm] = 1.0 if kp_axis is None else np.asarray(kp_axis, dtype=np.float64).reshape(3)
        self.p_des[arm] = np.asarray(p_des, dtype=np.float64).reshape(3)
        self.f_ff[arm] = 0.0 if f_ff is None else np.asarray(f_ff, dtype=np.float64).reshape(3)
        self.v_des[arm] = 0.0 if v_des is None else np.asarray(v_des, dtype=np.float64).reshape(3)

    def force(self, arm: int, p: np.ndarray, v: np.ndarray) -> np.ndarray:
        """! @brief 腕 arm の手先力指令（インピーダンス + FF、上限つき）。
        @param arm 腕インデックス [-]
        @param p   手先位置 {W} shape (3,) [m]
        @param v   手先速度 {W} shape (3,) [m/s]
        @return F shape (3,) [N]
        """
        s = self.kp_axis[arm]
        F = s * (self.kp * (self.p_des[arm] - p) + self.kd * (self.v_des[arm] - v)) + self.f_ff[arm]
        n = float(np.linalg.norm(F))
        if n > self.f_max:
            F *= self.f_max / n
        self.last_F[arm] = F
        return F

    def moment(self, arm: int, R: np.ndarray, w: np.ndarray) -> np.ndarray:
        """! @brief 腕 arm の手先姿勢モーメント指令（SO(3) 対数写像の PD、上限つき）。
        @param arm 腕インデックス [-]
        @param R   手先姿勢 {W} shape (3,3) [-]
        @param w   手先角速度 {W} shape (3,) [rad/s]
        @return M shape (3,) [N*m]（hold_rot が False なら 0）
        """
        if not self.hold_rot:
            return np.zeros(3)
        if not np.all(np.isfinite(self.R_des[arm])):
            self.R_des[arm] = np.asarray(R, dtype=np.float64)   # 最初の周期の姿勢にラッチ
        from ..common.frames import log_so3

        e = log_so3(self.R_des[arm] @ np.asarray(R).T)
        M = self.kp_rot * e - self.kd_rot * np.asarray(w)
        n = float(np.linalg.norm(M))
        if n > self.m_rot_max:
            M *= self.m_rot_max / n
        return M

    @staticmethod
    def com_wrench_of_env(wrench_hands: np.ndarray, p_hands: np.ndarray, c: np.ndarray) -> np.ndarray:
        """! @brief 手が受ける接触レンチ（手先 site まわり）を重心まわりのレンチに移して合計する。
        @param wrench_hands shape (n_arm,6) [N, N*m]（`RobotState.wrench_hands`）
        @param p_hands      手先位置 {W} shape (n_arm,3) [m]
        @param c            重心 {W} shape (3,) [m]
        @return W_env shape (6,) [N, N*m]
        """
        W = np.zeros(6)
        for a in range(wrench_hands.shape[0]):
            f = wrench_hands[a, 0:3]
            W[0:3] += f
            W[3:6] += wrench_hands[a, 3:6] + np.cross(p_hands[a] - c, f)
        return W
