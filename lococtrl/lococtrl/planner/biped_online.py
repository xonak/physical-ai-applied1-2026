"""! @file biped_online.py
@brief オンライン足配置計画（DCM ステップ調整 + 時間調整 + 後退ホライズン DCM 再計画）。

`gait.biped_schedule` のオフライン固定列の代わりに、足配置を
**毎制御周期に計画し直す**計画器である。入力は速度指令（`set_command`）か
外部からの足配置列（`push_footstep`。テレオペの足配置ストリーミングの入口）、
出力はオフライン計画と同じ `BipedReference` なので、安定化層・全身制御器は変更なしで載る。

## 1 周期の処理

1. 状態機械（STAND -> DS -> SS -> DS -> ... -> STAND）を進める。SS は早期接地で短縮、
   未接地で最大 `t_ss_extra` まで延長する
2. 公称の次の着地位置を作る。**実測の立脚ではなく、速度指令を積分した公称パターン座標系**
   `(x, y, psi)`（1 歩ごとに `R(psi) v T` 進み `wz T` 回る。T = t_ds + t_ss）の横 `±w/2` に置く。
   足と胴体の目標ヨーも `psi` に追従させる（旋回）。
   立脚の実測位置を基準にすると、1 歩の着地誤差がそのまま次の公称に乗って横幅が縮み、
   数歩で左右の間隔が崩れた（実測）。stream のときは待ち行列の先頭
3. **DCM ステップ調整**（Englsberger 2015 / Khadiv 2016）: 現在の DCM 誤差 `e = xi - xi_ref` を
   単脚支持の残り時間で発散させた `e_eos = e exp(omega T_rem)` を次の着地位置に足す:
   `p_next <- p_next + k_step e_eos`。可到達領域（前後の最大歩幅、左右の最小/最大間隔）に clamp
4. **時間調整**: `|e_eos|` が `timing_margin` を超えるなら、超えない残り時間
   `T_rem' = ln(margin / |e|) / omega` まで単脚支持を短縮する（`t_ss_min` を下限。伸ばさない）
5. **後退ホライズン DCM 再計画**: 現在区間の残り + `preview_steps` 歩ぶんの区分一定 CoP 列
   （DS は `ds_subdiv` 分割の線形補間）から終端条件で後ろ向き再帰し、現在時刻の
   `xi_ref`, `xi_dot_ref` を出す。重心参照は `c_{k+1} = xi_k + (c_k - xi_k) exp(-omega dt)` で内部積分
6. 遊脚軌道は離陸位置から（更新され続ける）着地目標への `swing_trajectory`。
   進捗が `latch_ratio` を超えたら目標を固定する（着地直前の横滑り防止）

## オフライン固定列との違い

固定列では横押しの限界（4.5 N*s）が足裏幅の capturability で決まる（足首戦略の限界）。
本計画器は DCM 誤差を**次の一歩で吸収**するので、限界は歩幅の可到達領域で決まる。
`clamp_cop` の飽和量を着地位置へ回す考え方を、DCM 誤差を回す形で実現している
（両者は LIPM では同じ情報）。

記号: nf 脚数（= 2）[-]、omega = sqrt(g / z_c) [rad/s]、T 1 歩の周期 [s]。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from ..common.config import RobotConfig
from ..common.frames import quat_to_euler_zyx
from ..common.types import RobotState
from ..control.biped_stack import BipedReference
from ..gait.biped_schedule import BipedGaitParams, Foot, SegKind
from ..gait.swing import swing_trajectory
from ..model_dynamics.robot_model import RobotModelBase

__all__ = ["OnlinePlannerParams", "FootstepCommand", "OnlineBipedPlanner"]


@dataclass
class OnlinePlannerParams:
    """! @brief オンライン計画器のパラメータ（実験設定の planner 節）。
    @param vx, vy          初期の速度指令 {H} [m/s]
    @param source          "velocity"（速度指令から生成）| "stream"（push_footstep の待ち行列）
    @param wz              旋回速度指令 [rad/s]（パターン座標系の向き psi を 1 歩ごとに wz T 回す）
    @param turn_max_step   1 歩で回してよい向きの上限 [rad]
    @param k_step          DCM 誤差の着地位置への反映率 [-]（0 で調整なし、1 で全量）
    @param adapt_timing    時間調整を使うか [-]
    @param timing_margin   時間調整を始める発散後 DCM 誤差 [m]
    @param t_ss_min        単脚支持の下限 [s]
    @param t_ss_extra      未接地のとき単脚支持を延長できる上限 [s]
    @param v_swing_max     遊脚の最大水平速度 [m/s]。着地目標は現在の足先から `v_swing_max T_rem` 以内に
                           clamp し、時間調整もこれで届く範囲までしか縮めない。無いと短縮した
                           単脚支持で足が届かず、DCM 誤差が一歩ごとに残る（実測: 前押し 14 N*s で転倒）
    @param step_max_length 前後の最大歩幅（立脚からの距離）[m]
    @param step_min_width  左右の最小間隔（内側。足同士の干渉を避ける）[m]
    @param step_max_width  左右の最大間隔 [m]
    @param touchdown_depth 遊脚の終端を地面より下に置く量 [m]。0 だと追従誤差ぶん接地が遅れ、
                           単脚支持が延長されて DCM 参照と実測がずれる
    @param latch_ratio     着地目標を固定する遊脚進捗 [-]
    @param preview_steps   先読みする歩数 [-]
    @param duration        stream のときの実行時間 [s]（velocity では歩数から決まる）
    @param closing_step    最後に足を揃える 1 歩を入れるか [-]
    @param z_c_offset      重心高さの公称姿勢からのずらし [m]（負で低く）。前押しの後に骨盤が先行すると
                           後ろ足が伸びきって持ち上がらない（膝の特異姿勢、実測）。低く歩くと届く範囲が広がる
    @param stop_margin     歩き終わりに移ってよい DCM 誤差（発散後）の上限 [m]。これより大きければ
                           その場で回復の 1 歩（速度 0 の公称 + 調整）を足す
    @param max_extra_steps 回復の歩数の上限 [-]
    @param endless         歩数で終わらず速度指令が続く限り歩く（テレオペ）。指令がほぼ 0 なら足を揃えて立ち、
                           指令が来たらまた歩き出す。実行時間は duration [-]
    @param idle_speed      これ未満の並進指令を「止まれ」とみなす [m/s]
    @param idle_turn       これ未満の旋回指令を「止まれ」とみなす [rad/s]
    @param lean_xy         DCM 参照の CoP 列（CMP）を足から前にずらす量（パターン座標系）[m]。前に正。
                           押し作業で重心を足より前に置く「前傾」の口（`set_lean` で毎周期変えられる）。
                           立位でも歩行でも同じ（足配置はずらさない）
    @param z_c_rate        重心高さの切替速度の上限 [m/s]（`set_com_height`。両脚支持・立位のときだけ進める）
    """

    vx: float = 0.2
    vy: float = 0.0
    wz: float = 0.0
    turn_max_step: float = 0.3
    source: str = "velocity"
    k_step: float = 1.0
    adapt_timing: bool = True
    timing_margin: float = 0.10
    t_ss_min: float = 0.25
    v_swing_max: float = 1.2
    t_ss_extra: float = 0.0
    touchdown_depth: float = 0.005
    step_max_length: float = 0.35
    step_min_width: float = 0.15
    step_max_width: float = 0.40
    latch_ratio: float = 0.7
    preview_steps: int = 3
    duration: float = 10.0
    closing_step: bool = True
    z_c_offset: float = -0.04
    stop_margin: float = 0.04
    max_extra_steps: int = 4
    endless: bool = False
    idle_speed: float = 0.02
    idle_turn: float = 0.05
    lean_xy: tuple[float, float] = (0.0, 0.0)
    z_c_rate: float = 0.15


@dataclass
class FootstepCommand:
    """! @brief 外部から与える足配置（テレオペの足配置ストリームの 1 要素）。
    @param foot 踏み出す足 [-]（0 = 左, 1 = 右）
    @param xy   着地位置 {W} shape (2,) [m]
    @param t_ss 単脚支持時間 [s]（None で公称）
    @param yaw  着地時の足の向き [rad]（None で今の向き）
    """

    foot: int
    xy: np.ndarray
    yaw: float | None = None
    t_ss: float | None = None


@dataclass
class _Segment:
    """! @brief 先読み用の区分一定 CoP 区間。"""

    kind: SegKind
    dur: float
    cop: np.ndarray


class OnlineBipedPlanner:
    """! @brief オンライン足配置計画器。`BipedPlannerLayer` と同じ口を持つ。"""

    def __init__(
        self,
        model: RobotModelBase,
        rcfg: RobotConfig,
        gait: BipedGaitParams,
        par: OnlinePlannerParams,
        dt: float,
    ) -> None:
        """! @brief 構築する。
        @param model RobotModelBase [-]
        @param rcfg  RobotConfig [-]
        @param gait  歩容（t_ds, t_ss, step_width, swing_height, n_step, t_start, t_end, ds_subdiv, first_swing）[-]
        @param par   OnlinePlannerParams [-]
        @param dt    制御周期 [s]
        @return なし
        """
        self.model = model
        self.rcfg = rcfg
        self.gait = gait
        self.par = par
        self.dt = float(dt)
        q_nom = model.make_q(np.array([0.0, 0.0, rcfg.h_nominal]), np.array([1.0, 0.0, 0.0, 0.0]), rcfg.q_nominal)
        self.q_nominal = np.asarray(rcfg.q_nominal, dtype=np.float64).copy()
        c_nom = model.com(q_nom)
        self.z_c_nominal = float(c_nom[2]) + float(par.z_c_offset)
        self.z_c = self.z_c_nominal
        self.z_c_target = self.z_c
        self.omega = float(np.sqrt(9.81 / self.z_c))
        self.nf = model.n_leg
        self.cmd = np.array([par.vx, par.vy, par.wz], dtype=np.float64)
        self.queue: deque[FootstepCommand] = deque()
        n_walk = int(gait.n_step) + (1 if par.closing_step and gait.n_step > 0 else 0)
        if par.source == "stream" or par.endless:
            self.total_time = float(par.duration)
        else:
            # 回復の歩数ぶんの余裕を持たせる（使わなければ最後は立位のまま）
            self.total_time = (gait.t_start + (n_walk + par.max_extra_steps) * (gait.t_ds + gait.t_ss)
                               + gait.t_ds + gait.t_end)
        self.n_t = int(round(self.total_time / self.dt))
        self.n_walk = n_walk
        self.n_extra = 0
        self.reset()

    # ------------------------------------------------------------------ 外部入力
    def set_command(self, vx: float, vy: float = 0.0, wz: float = 0.0) -> None:
        """! @brief 速度指令を更新する（歩行中に変えてよい）。パターン座標系（進行方向 psi）で与える。
        @param vx 前進速度 [m/s]
        @param vy 横速度 [m/s]（左が正）
        @param wz 旋回速度 [rad/s]（左旋回が正）
        @return なし
        """
        self.cmd = np.array([float(vx), float(vy), float(wz)], dtype=np.float64)

    def push_footstep(self, foot: int, xy: np.ndarray, t_ss: float | None = None, yaw: float | None = None) -> None:
        """! @brief 足配置を待ち行列に積む（source = stream で使う）。
        @param foot 踏み出す足 [-]
        @param xy   着地位置 {W} shape (2,) [m]
        @param t_ss 単脚支持時間 [s]（None で公称）
        @param yaw  着地時の足の向き [rad]（None で今の向き）
        @return なし
        """
        self.queue.append(FootstepCommand(int(foot), np.asarray(xy, dtype=np.float64).reshape(2), t_ss,
                                          None if yaw is None else float(yaw)))

    def set_step_residual(self, dxy: np.ndarray, dT: float = 0.0) -> None:
        """! @brief HL 方策の出力: 今の 1 歩の着地位置と単脚支持時間への残差（Egle 2024 型の口）。
        @param dxy 着地位置の残差 {W} shape (2,) [m]（公称 + DCM 則の調整に加算）
        @param dT  単脚支持時間の残差 [s]（公称 t_ss に加算。下限 t_ss_min、時間調整はこれより縮める方向にだけ働く）
        @return なし
        @note SS 中、目標を固定する（latch_ratio）前に呼んだぶんだけ効く。歩が変わるとクリアされる。
              DS / STAND で呼ぶと次の SS に持ち越す。
        """
        self.hl_dxy = np.asarray(dxy, dtype=np.float64).reshape(2).copy()
        self.hl_dT = float(dT)
        self.hl_active = True
        self._hl_set_phase = self.phase
        if self.phase is SegKind.SS and not self.latched:
            self._apply_hl_timing()

    def _apply_hl_timing(self) -> None:
        """! @brief HL の時間残差を単脚支持時間に反映する（既に経過した時間より短くはしない）。"""
        elapsed = self._t_now - self.t_phase0
        T_new = max(self.par.t_ss_min, elapsed + self.dt, min(self.T_ss_base + self.hl_dT, self.T_adapt))
        self.T_phase = float(T_new)

    # ------------------------------------------------------------------ 内部状態
    def reset(self) -> None:
        """! @brief 内部状態を初期化する（立位から）。
        @return なし
        """
        self.phase = SegKind.STAND
        self.t_phase0 = 0.0
        self.T_phase = float(self.gait.t_start)
        self.swing = int(self.gait.first_swing)
        self.p_feet = np.zeros((self.nf, 3))
        self.pat = np.zeros(3)  # 公称パターン座標系 (x, y, psi)。速度指令をここに積分する
        self.yaw_feet = np.zeros(self.nf)  # 各足の計画上の向き [rad]
        self.yaw_liftoff = 0.0
        self.yaw_target = 0.0
        self._init_feet = False
        self.p_liftoff = np.zeros(3)
        self.target = np.zeros(3)
        self.target_nominal = np.zeros(3)
        self.latched = False
        self.steps_done = 0
        self.finished = False
        self.c_ref = np.zeros(3)
        self.c_ref_init = False
        self.cop_ds_from = np.zeros(2)
        self.arriving = -1
        self.departing = -1
        self.last_e_eos = np.zeros(2)
        self.last_t_ss = float(self.gait.t_ss)
        self.n_timing_adapt = 0
        self.n_extra = 0
        self._idle_aligned = True  # endless: 足が揃って立っているか
        self.lean_xy = np.array(self.par.lean_xy, dtype=np.float64).reshape(2)
        self.z_c = self.z_c_nominal
        self.z_c_target = self.z_c
        self.omega = float(np.sqrt(9.81 / self.z_c))
        self.n_walk = int(self.gait.n_step) + (1 if self.par.closing_step and self.gait.n_step > 0 else 0)
        self.queue.clear()
        # HL 方策の口（1 歩ごとにクリア）
        self.hl_dxy = np.zeros(2)
        self.hl_dT = 0.0
        self.hl_active = False
        self._hl_set_phase = SegKind.STAND
        self._carry_hl = False
        self._t_now = 0.0
        self.T_ss_base = float(self.gait.t_ss)
        self.T_adapt = float("inf")
        # 観測用に毎周期更新する値
        self.last_e = np.zeros(2)
        self.last_xi_ref = np.zeros(2)
        self.last_T_rem = 0.0

    @property
    def t_now(self) -> float:
        """! @brief 最後に update した時刻 [s]"""
        return float(self._t_now)

    @property
    def psi(self) -> float:
        """! @brief パターン座標系の向き [rad]"""
        return float(self.pat[2])

    @property
    def initialized(self) -> bool:
        """! @brief 最初の update 済みか（足位置・パターンが入っている）[-]"""
        return bool(self._init_feet)

    def rot2(self, psi: float | None = None) -> np.ndarray:
        """! @brief 2 次元回転行列 R(psi)（None で今のパターンの向き）[-]"""
        return self._rot2(self.psi if psi is None else float(psi))

    def set_lean(self, dx: float, dy: float = 0.0) -> None:
        """! @brief 立位の DCM 目標のずらし（前傾）をパターン座標系で与える。
        @param dx 前方向 [m]
        @param dy 左方向 [m]
        @return なし
        """
        self.lean_xy = np.array([float(dx), float(dy)])

    def set_com_height(self, z_c: float) -> None:
        """! @brief 重心高さの目標を与える（区間ごとの切替。両脚支持・立位のときに z_c_rate で近づける）。
        @param z_c 重心高さ {W} [m]（公称は `z_c_nominal`。しゃがむなら小さく）
        @return なし
        @note omega = sqrt(g / z_c) も一緒に変わる。単脚支持中は変えない（DCM の発散則が区間内で一定であるように）。
        """
        self.z_c_target = float(z_c)

    def _update_com_height(self) -> None:
        """! @brief 両脚支持・立位なら z_c を目標へ速度制限つきで進め、omega を更新する。
        @return なし
        """
        if self.phase is SegKind.SS:
            return
        dz = self.z_c_target - self.z_c
        step = self.par.z_c_rate * self.dt
        if abs(dz) > step:
            dz = np.sign(dz) * step
        if dz != 0.0:
            self.z_c += float(dz)
            self.omega = float(np.sqrt(9.81 / self.z_c))
            self.c_ref[2] = self.z_c

    def _lean_W(self, psi: float) -> np.ndarray:
        """! @brief lean をワールド系 xy に回す。
        @param psi パターン座標系の向き [rad]
        @return shape (2,) [m]
        """
        lx, ly = self.lean_xy
        c, s = np.cos(psi), np.sin(psi)
        return np.array([c * lx - s * ly, s * lx + c * ly])

    def _idle(self) -> bool:
        """! @brief endless で速度指令がほぼ 0 か [-]"""
        p = self.par
        return bool(p.endless and p.source == "velocity" and float(np.linalg.norm(self.cmd[0:2])) < p.idle_speed
                    and abs(float(self.cmd[2])) < p.idle_turn)

    @staticmethod
    def _wrap(a: float) -> float:
        return float((a + np.pi) % (2.0 * np.pi) - np.pi)

    def _foot_sign(self, foot: int) -> float:
        return 1.0 if foot == int(Foot.LEFT) else -1.0

    @staticmethod
    def _rot2(psi: float) -> np.ndarray:
        c, s_ = np.cos(psi), np.sin(psi)
        return np.array([[c, -s_], [s_, c]])

    @staticmethod
    def _rotz(psi: float) -> np.ndarray:
        R = np.eye(3)
        R[0:2, 0:2] = OnlineBipedPlanner._rot2(psi)
        return R

    def _advance_pattern(self, pat: np.ndarray, t_step: float) -> np.ndarray:
        """! @brief 公称パターン座標系を 1 歩ぶん進める（速度指令はパターン系で与えられている）。
        @param pat    (x, y, psi) [m, m, rad]
        @param t_step 1 歩の周期 [s]
        @return 進めた (x, y, psi)
        @note 旋回は 1 歩あたり turn_max_step に制限。並進は向きの中間値で回す（円弧の近似）。
        """
        dpsi = float(np.clip(self.cmd[2] * t_step, -self.par.turn_max_step, self.par.turn_max_step))
        psi_mid = pat[2] + 0.5 * dpsi
        dxy = self._rot2(psi_mid) @ (self.cmd[0:2] * t_step)
        return np.array([pat[0] + dxy[0], pat[1] + dxy[1], pat[2] + dpsi])

    def _pattern_foot(self, pat: np.ndarray, foot: int) -> np.ndarray:
        """! @brief パターン座標系上の足の公称位置（中心から横に ±step_width/2）。
        @param pat  (x, y, psi) [-]
        @param foot 足 [-]
        @return xy shape (2,) [m]
        """
        s = self._foot_sign(foot)
        return pat[0:2] + self._rot2(pat[2]) @ np.array([0.0, s * 0.5 * self.gait.step_width])

    def _nominal_target(self, swing: int, pat: np.ndarray, t_step: float,
                        closing: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """! @brief 速度指令から公称の着地位置（x,y）を作る（パターン座標系を v T 進め、その横に置く）。
        @param swing   遊脚 [-]
        @param pat     公称パターン (x, y, psi) [-]
        @param t_step  1 歩の周期 [s]
        @param closing 最後の足揃え（パターンを進めず、横に揃える）[-]
        @return (xy shape (2,) [m], 進めた pat)
        @note 歩き始めの 1 歩は v T、以降は立脚から v T 前へ出る標準の交互歩容。
              1 歩ごとに足が揃う行進型にはしない。
        """
        pat_next = pat.copy() if closing else self._advance_pattern(pat, t_step)
        return self._pattern_foot(pat_next, swing), pat_next

    def _clamp_target(self, swing: int, stance_xy: np.ndarray, xy: np.ndarray,
                      psi: float | None = None) -> np.ndarray:
        """! @brief 可到達領域（前後の最大歩幅、左右の最小/最大間隔）に clamp する（パターン系で評価）。
        @param swing     遊脚 [-]
        @param stance_xy 立脚位置 shape (2,) [m]
        @param xy        着地位置 shape (2,) [m]
        @param psi       評価する向き [rad]（None で今のパターンの向き）
        @return clamp 後 shape (2,) [m]
        """
        p = self.par
        s = self._foot_sign(swing)
        R = self._rot2(self.pat[2] if psi is None else psi)
        d = R.T @ (np.asarray(xy, dtype=np.float64) - stance_xy)
        d[0] = np.clip(d[0], -p.step_max_length, p.step_max_length)
        lat = s * d[1]  # 外側が正
        lat = np.clip(lat, p.step_min_width, p.step_max_width)
        d[1] = s * lat
        return stance_xy + R @ d

    def _advance_phase(self, t: float, state: RobotState) -> None:
        """! @brief 状態機械を進める。
        @param t     時刻 [s]
        @param state 推定状態 [-]
        @return なし
        """
        while True:
            elapsed = t - self.t_phase0
            if self.phase is SegKind.STAND:
                if self.finished or elapsed < self.T_phase:
                    return
                if self.par.source != "stream" and not self.par.endless and self.n_walk == 0:
                    return
                if self.par.source == "stream" and not self.queue:
                    return  # 足配置が来るまで立つ
                if self._idle() and self._idle_aligned:
                    return  # endless: 指令が来るまで立つ
                self._start_ds(t, departing=self.swing)
                continue
            if self.phase is SegKind.DS:
                if elapsed < self.T_phase:
                    return
                self._start_ss(t, state)
                continue
            # SS
            u = elapsed / self.T_phase
            meas = bool(np.asarray(state.contact, dtype=bool)[self.swing])
            if u >= 1.0 and not meas and elapsed < self.T_phase + self.par.t_ss_extra:
                return  # 未接地: 少し延長
            if u >= 1.0 or (meas and u > self.par.latch_ratio):
                self._end_ss(t, state)
                continue
            return

    def _start_ds(self, t: float, departing: int) -> None:
        self.phase = SegKind.DS
        self.t_phase0 = t
        self.T_phase = float(self.gait.t_ds)
        self.departing = int(departing)
        self.swing = int(departing)
        stance = 1 - self.swing
        # DS の CoP は「直前の CoP」から次の立脚足へ動かす
        self.cop_ds_to = self.p_feet[stance, 0:2].copy()

    def _start_ss(self, t: float, state: RobotState) -> None:
        self.phase = SegKind.SS
        self.t_phase0 = t
        stance = 1 - self.swing
        self.p_liftoff = np.asarray(state.p_WF[self.swing], dtype=np.float64).copy()
        self.p_liftoff[2] = self.gait.foot_z
        self.yaw_liftoff = float(self.yaw_feet[self.swing])
        self.yaw_target = float(self.pat[2])
        t_step = self.gait.t_ds + self.gait.t_ss
        self.T_phase = float(self.gait.t_ss)
        self._carry_hl = self.hl_active and self._hl_set_phase is not SegKind.SS
        if not self._carry_hl:
            self.hl_dxy = np.zeros(2)
            self.hl_dT = 0.0
            self.hl_active = False
        if self.par.source == "stream" and self.queue:
            cmd = self.queue.popleft()
            if cmd.foot != self.swing:
                # 違う足の指令が先頭なら、足を揃えるために今の遊脚は公称で踏む
                self.queue.appendleft(cmd)
                xy, self.pat = self._nominal_target(self.swing, self.pat, t_step, closing=True)
            else:
                xy = cmd.xy.copy()
                if cmd.t_ss is not None:
                    self.T_phase = float(cmd.t_ss)
                if cmd.yaw is not None:
                    self.yaw_target = float(cmd.yaw)
                    self.pat[2] = self.yaw_target
                # パターン中心は着地足と立脚の中点へ（次の公称がここから出る）
                self.pat[0:2] = 0.5 * (xy + self.p_feet[stance, 0:2])
        elif self.par.endless:
            idle = self._idle()
            xy, self.pat = self._nominal_target(self.swing, self.pat, t_step, closing=idle)
            self._idle_aligned = idle
        else:
            closing = self.steps_done >= int(self.gait.n_step) and self.par.closing_step
            if closing and self.n_extra < self.par.max_extra_steps \
                    and float(np.linalg.norm(self.last_e_eos)) > self.par.stop_margin:
                # DCM 誤差が残ったまま足を揃えると止まれない。速度 0 の回復の 1 歩を足す
                closing = False
                self.n_extra += 1
                self.n_walk += 1
                xy, self.pat = self._nominal_target(self.swing, self.pat, t_step, closing=True)
            else:
                xy, self.pat = self._nominal_target(self.swing, self.pat, t_step, closing=closing)
        # パターンは進めたが、調整量は乗せない（調整は 1 歩で消える）
        xy = self._clamp_target(self.swing, self.p_feet[stance, 0:2], xy)
        self.target_nominal = np.array([xy[0], xy[1], self.gait.foot_z - self.par.touchdown_depth])
        self.target = self.target_nominal.copy()
        self.latched = False
        self.T_ss_base = float(self.T_phase)
        self.T_adapt = float("inf")  # 時間調整で縮めた値（HL の残差と独立に持つ）
        if self.hl_active:
            self._apply_hl_timing()
        self.cop_ds_from = self.cop_ds_to.copy()

    def _end_ss(self, t: float, state: RobotState) -> None:
        landed = self.swing
        # DCM 計画上の足位置は**目標**を使う。実測に置き換えると着地誤差ぶん参照が飛び、
        # その飛びが次の 1 歩の調整量に化ける。実測との差は CoP 指令の clamp（実測の足裏矩形）が吸収する
        self.p_feet[landed, 0:2] = self.target[0:2]
        self.p_feet[landed, 2] = self.gait.foot_z
        # 向きは接地時点の補間値で確定する（早期接地で目標まで回り切らないとき、立脚の姿勢参照が飛ばない）
        u_td = float(np.clip((t - self.t_phase0) / max(self.T_phase, 1.0e-9), 0.0, 1.0))
        su = 3.0 * u_td * u_td - 2.0 * u_td ** 3
        self.yaw_feet[landed] = self.yaw_liftoff + su * self._wrap(self.yaw_target - self.yaw_liftoff)
        self.p_meas_landed = np.asarray(state.p_WF[landed, 0:2], dtype=np.float64).copy()
        self.steps_done += 1
        self.arriving = landed
        self.last_t_ss = float(self.T_phase)
        if self.par.endless:
            if self._idle() and self._idle_aligned and float(np.linalg.norm(self.last_e_eos)) <= self.par.stop_margin:
                # 止まれ: 足が揃っていて DCM 誤差も小さい -> 立って待つ（誤差が残れば足踏みを続ける）
                self.phase = SegKind.STAND
                self.t_phase0 = t
                self.T_phase = 0.0
                self.swing = 1 - landed
                self.departing = -1
                return
            self._start_ds(t, departing=1 - landed)
            return
        done = (self.par.source != "stream" and self.steps_done >= self.n_walk) or (
            self.par.source == "stream" and not self.queue
        )
        if done and self.par.source != "stream":
            self.phase = SegKind.STAND
            self.t_phase0 = t
            self.T_phase = float("inf")
            self.finished = True
            self.departing = -1
            return
        if done:
            # stream で次が無い: 立位で待つ（荷重は着地足を t_ds で立ち上げる）
            self.phase = SegKind.STAND
            self.t_phase0 = t
            self.T_phase = 0.0
            self.swing = 1 - landed
            self.departing = -1
            return
        self._start_ds(t, departing=1 - landed)

    # ------------------------------------------------------------------ DCM 先読み
    def _preview_segments(self, t: float) -> tuple[list[_Segment], int]:
        """! @brief 現在区間の残り + 先読み歩のセグメント列（区分一定 CoP）を作る。
        @param t 時刻 [s]
        @return (segments, 現在区間の index)
        """
        g, p = self.gait, self.par
        segs: list[_Segment] = []
        feet = self.p_feet.copy()
        elapsed = t - self.t_phase0
        nsub = int(g.ds_subdiv)

        def push_ds(cop_from: np.ndarray, cop_to: np.ndarray, dur: float, u0: float = 0.0) -> None:
            # u0 から 1 までの残りを nsub 分割で近似
            n = max(1, int(np.ceil(nsub * (1.0 - u0))))
            for k in range(1, n + 1):
                a = u0 + (1.0 - u0) * k / n
                segs.append(_Segment(SegKind.DS, dur / n, (1.0 - a) * cop_from + a * cop_to))

        cur = 0
        pat = self.pat.copy()
        skip_first_ds = self.phase is SegKind.DS  # 現在が DS なら、先読みの 1 歩目は SS から始まる
        lean = self._lean_W(pat[2])
        if self.phase is SegKind.STAND:
            mid = 0.5 * (feet[0, 0:2] + feet[1, 0:2])
            rem = max(self.T_phase - elapsed, 0.0) if np.isfinite(self.T_phase) else 1.0
            if self.finished or (p.source != "stream" and not p.endless and self.n_walk == 0) \
                    or (p.source == "stream" and not self.queue) or (self._idle() and self._idle_aligned):
                segs.append(_Segment(SegKind.STAND, max(rem, 1.0), mid))
                return self._shift_cmp(segs, lean), 0
            segs.append(_Segment(SegKind.STAND, max(rem, 1.0e-3), mid))
            swing = self.swing
            cop_prev = mid
            steps_ahead = p.preview_steps
        elif self.phase is SegKind.DS:
            u0 = float(np.clip(elapsed / self.T_phase, 0.0, 1.0))
            push_ds(self.cop_ds_from, self.cop_ds_to, max(self.T_phase - elapsed, 1.0e-3), u0)
            swing = self.swing
            cop_prev = self.cop_ds_to
            steps_ahead = p.preview_steps
        else:  # SS: 現在区間 = 立脚の CoP
            stance = 1 - self.swing
            # 残りは負になってよい（接地が遅れて延長中）。負なら参照は立脚 CoP から発散を続け、
            # 実測の DCM と同じ力学に従う。0 で止めると参照だけが止まり誤差が溜まる
            segs.append(_Segment(SegKind.SS, self.T_phase - elapsed, feet[stance, 0:2].copy()))
            feet[self.swing, 0:2] = self.target[0:2]
            swing = 1 - self.swing
            cop_prev = feet[stance, 0:2].copy()
            steps_ahead = p.preview_steps - 1
        if p.source == "stream":
            steps_left = len(self.queue)
        elif p.endless:
            steps_left = p.preview_steps if not self._idle() else (0 if self._idle_aligned else 1)
        else:
            steps_left = self.n_walk - self.steps_done - (1 if self.phase is SegKind.SS else 0)
        n_ahead = int(max(0, min(steps_ahead, steps_left)))
        t_step = g.t_ds + g.t_ss
        qi = 0
        for j in range(n_ahead):
            stance = 1 - swing
            if not (skip_first_ds and j == 0):
                push_ds(cop_prev, feet[stance, 0:2], g.t_ds)
            segs.append(_Segment(SegKind.SS, g.t_ss, feet[stance, 0:2].copy()))
            if p.source == "stream":
                cmd = self.queue[qi] if qi < len(self.queue) else None
                qi += 1
                if cmd is not None and cmd.foot == swing:
                    xy = cmd.xy.copy()
                else:
                    xy, pat = self._nominal_target(swing, pat, t_step, closing=True)
            else:
                idx = self.steps_done + (1 if self.phase is SegKind.SS else 0) + j
                closing = self._idle() if p.endless else (idx >= int(g.n_step) and p.closing_step)
                xy, pat = self._nominal_target(swing, pat, t_step, closing=closing)
            xy = self._clamp_target(swing, feet[stance, 0:2], xy, psi=pat[2])
            feet[swing, 0:2] = xy
            cop_prev = feet[stance, 0:2].copy()
            swing = 1 - swing
        # 終端: 両足の中点へ DS、そこで停止
        mid = 0.5 * (feet[0, 0:2] + feet[1, 0:2])
        push_ds(cop_prev, mid, g.t_ds)
        segs.append(_Segment(SegKind.STAND, g.t_end, mid))
        return self._shift_cmp(segs, lean), cur

    @staticmethod
    def _shift_cmp(segs: list, lean: np.ndarray) -> list:
        """! @brief 全セグメントの CoP（DCM 参照の CMP）を lean だけずらす。
        @param segs セグメント列 [-]
        @param lean {W} shape (2,) [m]
        @return segs（その場で書き換え）
        @note 外力 f（手の反力）が掛かると、足裏 CoP は CMP から f_xy / (m omega^2) ずれる
              （`biped_stack.com_wrench_from_dcm`）。lean = −f_xy / (m omega^2) と置けば、足裏 CoP は
              足の中心に留まったまま重心が足より前に出る（立位でも歩行でも同じ）。足配置はずらさない:
              歩行中の重心は CoP 列を追うので、足を一様に後ろへずらしても重心と足の相対位置は変わらない。
        """
        if not np.any(lean):
            return segs
        for sg in segs:
            sg.cop = sg.cop + lean
        return segs

    def _dcm_reference(self, segs: list[_Segment], tau: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """! @brief セグメント列から現在時刻（区間 0 の先頭から tau = 0）の DCM 参照を出す。
        @param segs セグメント列 [-]
        @param tau  区間 0 内の経過（常に 0。残り時間で区間を切ってあるため）[s]
        @return (xi_ref shape (2,), xi_dot_ref shape (2,), cop_now shape (2,))
        """
        w = self.omega
        n = len(segs)
        xi_eos = np.zeros((n, 2))
        xi_eos[n - 1] = segs[n - 1].cop
        for i in range(n - 1, 0, -1):
            xi_eos[i - 1] = segs[i].cop + (xi_eos[i] - segs[i].cop) * np.exp(-w * segs[i].dur)
        p0 = segs[0].cop
        xi = p0 + (xi_eos[0] - p0) * np.exp(w * (tau - segs[0].dur))
        return xi, w * (xi - p0), p0

    # ------------------------------------------------------------------ 本体
    def update(self, state: RobotState, k: int) -> BipedReference:
        """! @brief 1 制御周期ぶん計画を進め、参照を返す。
        @param state 推定状態 [-]
        @param k     周期番号 [-]（t = k dt）
        @return BipedReference
        """
        t = k * self.dt
        self._t_now = t
        g, p = self.gait, self.par
        if not self._init_feet:
            self.p_feet = np.asarray(state.p_WF, dtype=np.float64).copy()
            self.p_feet[:, 2] = g.foot_z
            psi0 = float(quat_to_euler_zyx(np.asarray(state.quat_WB))[0])
            self.pat = np.array([*(0.5 * (self.p_feet[0, 0:2] + self.p_feet[1, 0:2])), psi0])
            self.yaw_feet[:] = psi0
            self.yaw_target = psi0
            self.cop_ds_to = 0.5 * (self.p_feet[0, 0:2] + self.p_feet[1, 0:2])
            self.cop_ds_from = self.cop_ds_to.copy()
            self._init_feet = True
        if not self.c_ref_init:
            self.c_ref = np.array([state.c_W[0], state.c_W[1], self.z_c])
            self.c_ref_init = True

        self._advance_phase(t, state)
        self._update_com_height()
        elapsed = t - self.t_phase0

        # --- DCM ステップ調整と時間調整（SS で、目標を固定する前だけ） ---
        if self.phase is SegKind.SS:
            u = float(np.clip(elapsed / self.T_phase, 0.0, 1.0))
            if not self.latched and u > p.latch_ratio:
                self.latched = True
            if not self.latched:
                segs, _ = self._preview_segments(t)
                xi_ref, _, _ = self._dcm_reference(segs, 0.0)
                xi = np.asarray(state.c_W[0:2]) + np.asarray(state.dc_W[0:2]) / self.omega
                e = xi - xi_ref
                T_rem = max(self.T_phase - elapsed, 0.0)
                self.last_e, self.last_xi_ref, self.last_T_rem = e.copy(), xi_ref.copy(), float(T_rem)
            if not self.latched and (p.k_step > 0.0 or self.hl_active):
                # 可到達性は「離陸位置から単脚支持の全時間で届く距離」で判定する。
                # 現在の足先位置と残り時間で判定すると、足が軌道に遅れたときに目標が
                # 足先へ向かって縮み、遊脚が後退する（実測）。
                p_sw = self.p_liftoff[0:2]
                stance = 1 - self.swing

                def reach_clamp(xy: np.ndarray, T_total: float) -> np.ndarray:
                    d = xy - p_sw
                    r_max = p.v_swing_max * max(T_total, 0.0)
                    n = float(np.linalg.norm(d))
                    return p_sw + d * (r_max / n) if n > r_max and n > 1.0e-9 else xy

                if p.adapt_timing:
                    en = float(np.linalg.norm(e))
                    if en > 1.0e-6 and en * np.exp(self.omega * T_rem) > p.timing_margin:
                        T_rem_new = max(np.log(p.timing_margin / en) / self.omega, 0.0)
                        # 縮めた時間で目標まで足が届くことを条件にする
                        xy_try = self._clamp_target(self.swing, self.p_feet[stance, 0:2],
                                                    self.target_nominal[0:2] + p.k_step * e * np.exp(self.omega * T_rem_new) + self.hl_dxy)
                        T_reach = float(np.linalg.norm(xy_try - p_sw)) / p.v_swing_max  # 全時間で届く条件
                        T_new = max(p.t_ss_min, elapsed + T_rem_new, T_reach)
                        if T_new < self.T_phase:
                            self.T_phase = float(T_new)
                            self.T_adapt = float(T_new)
                            self.n_timing_adapt += 1
                            T_rem = max(self.T_phase - elapsed, 0.0)
                e_eos = e * np.exp(self.omega * T_rem)
                self.last_e_eos = e_eos
                xy = self.target_nominal[0:2] + p.k_step * e_eos + self.hl_dxy
                xy = self._clamp_target(self.swing, self.p_feet[stance, 0:2], xy)
                xy = reach_clamp(xy, self.T_phase)
                self.target[0:2] = xy

        # --- 参照（再計画した足配置で後ろ向き再帰） ---
        segs, _ = self._preview_segments(t)
        xi_ref, xi_dot_ref, cop_now = self._dcm_reference(segs, 0.0)
        # 重心参照の内部積分（安定極 -omega）
        e_dt = np.exp(-self.omega * self.dt)
        self.c_ref[0:2] = xi_ref + (self.c_ref[0:2] - xi_ref) * e_dt
        dc_ref = np.array([*(self.omega * (xi_ref - self.c_ref[0:2])), 0.0])
        ddc_ref = np.array([*((self.omega ** 2) * (self.c_ref[0:2] - cop_now)), 0.0])

        # --- 足先 ---
        nf = self.nf
        p_foot = self.p_feet.copy()
        v_foot = np.zeros((nf, 3))
        a_foot = np.zeros((nf, 3))
        contact = np.ones(nf, dtype=bool)
        swing_ratio = np.zeros(nf)
        load = np.ones(nf)
        yaw_ref = self.yaw_feet.copy()
        if self.phase is SegKind.SS:
            u = float(np.clip(elapsed / self.T_phase, 0.0, 1.0))
            pf, vf, af = swing_trajectory(u, self.p_liftoff, self.target, g.swing_height, float(self.T_phase))
            p_foot[self.swing], v_foot[self.swing], a_foot[self.swing] = pf, vf, af
            su = 3.0 * u * u - 2.0 * u ** 3  # 向きは滑らかに補間
            yaw_ref[self.swing] = self.yaw_liftoff + su * self._wrap(self.yaw_target - self.yaw_liftoff)
            contact[self.swing] = False
            swing_ratio[self.swing] = u
            load[self.swing] = 0.0
        elif self.phase is SegKind.DS:
            u = float(np.clip(elapsed / self.T_phase, 0.0, 1.0))
            if self.arriving >= 0:
                load[self.arriving] = u
            if self.departing >= 0:
                load[self.departing] = 1.0 - u
        else:  # STAND
            if self.arriving >= 0 and self.steps_done > 0:
                load[self.arriving] = float(np.clip(elapsed / max(g.t_ds, 1.0e-9), 0.0, 1.0))

        return BipedReference(
            t=t,
            kind=self.phase,
            c_ref=self.c_ref.copy(),
            dc_ref=dc_ref,
            ddc_ref=ddc_ref,
            xi_ref=xi_ref,
            xi_dot_ref=xi_dot_ref,
            cop_sched=cop_now,
            p_foot=p_foot,
            v_foot=v_foot,
            a_foot=a_foot,
            R_foot=np.array([self._rotz(y) for y in yaw_ref]),
            contact=contact,
            load=load,
            swing_ratio=swing_ratio,
            q_posture=self.q_nominal,
            R_base=self._rotz(yaw_ref[0] + 0.5 * self._wrap(yaw_ref[1] - yaw_ref[0])),
        )

    # 互換（BipedPlannerLayer と同じ属性名）
    def reference_at(self, k: int) -> BipedReference:  # pragma: no cover - 状態が要るので使わない
        """! @brief オンライン計画器は状態が要るので `update(state, k)` を使うこと。"""
        raise RuntimeError("OnlineBipedPlanner は update(state, k) で呼ぶ")
