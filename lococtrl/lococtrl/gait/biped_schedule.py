"""! @file biped_schedule.py
@brief 二足の歩容スケジューラ（両脚支持期つき）と足配置・CoP 参照の生成。

二足の歩容を**両脚支持期（DS）と単脚支持期（SS）の列**として時間を区切る。

## 両脚支持期を 0 にしない理由

区分一定の CoP 参照は踏み替えの瞬間に歩幅ぶん不連続に飛ぶ。予見制御での試算では、
歩幅 0.3 m・両脚支持期ゼロで ZMP が足（±0.09 m）を 25 mm 超えた。
両脚支持期を 0.2 s 入れて参照を繋ぐと、どの条件でも 85 mm 以上内側に収まった
よって `t_ds > 0` を強制する。

## CoP 参照の区分一定近似

`planner/dcm.DcmPlanner` は区分一定 CoP を前提に閉形式で解く。両脚支持期の
CoP は前の支持足から次の支持足へ連続に動かしたいので、**DS を `ds_subdiv` 等分して
各小区間を一定値で近似する**。分割を細かくすれば線形移動に収束する。

記号: nf 脚数（= 2）[-]、n_step 歩数 [-]、n_seg 区間数 [-]。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from ..common.types import GaitState

__all__ = ["Foot", "SegKind", "BipedGaitParams", "BipedSchedule", "make_biped_schedule"]


class Foot(int, Enum):
    """! @brief 二足の脚インデックス（configs/robot_g1.yaml の leg_names と一致）。"""

    LEFT = 0
    RIGHT = 1


class SegKind(str, Enum):
    """! @brief 区間の種類。"""

    STAND = "STAND"  # 静止（両脚支持）
    DS = "DS"  # 両脚支持（踏み替え）
    SS = "SS"  # 単脚支持（遊脚あり）


@dataclass
class BipedGaitParams:
    """! @brief 二足歩容のパラメータ。

    @param t_ds         両脚支持期 [s]。**0 にしてはならない**（上記の理由）
    @param t_ss         単脚支持期 [s]
    @param step_length  1 歩の踏み出し距離 [m]。平均前進速度は step_length/2/(t_ds+t_ss)
    @param step_width   左右の足の y 間隔 [m]
    @param swing_height 遊脚の持ち上げ高さ [m]
    @param n_step       歩数 [-]
    @param t_start      歩き出す前の静止時間 [s]
    @param t_end        歩き終えた後の静止時間 [s]
    @param first_swing  最初に踏み出す足 [-]
    @param ds_subdiv    両脚支持期の区分一定分割数 [-]
    @param foot_z       足裏の目標高さ [m]（接地面）
    """

    t_ds: float = 0.15
    t_ss: float = 0.35
    step_length: float = 0.20
    step_width: float = 0.237
    swing_height: float = 0.05
    n_step: int = 10
    t_start: float = 1.0
    t_end: float = 2.0
    first_swing: int = int(Foot.LEFT)
    ds_subdiv: int = 10
    foot_z: float = 0.0

    def __post_init__(self) -> None:
        """! @brief 両脚支持期がゼロでないことを検査する。
        @return なし
        @note t_ds = 0 では踏み替えで ZMP が支持多角形を出る（モジュール冒頭の説明を参照）。
        """
        if self.t_ds <= 0.0:
            raise ValueError(
                f"t_ds は正でなければならない（踏み替えで ZMP が足を出る）: {self.t_ds}"
            )
        if self.t_ss <= 0.0:
            raise ValueError(f"t_ss は正でなければならない: {self.t_ss}")

    @property
    def period(self) -> float:
        """! @brief 1 歩の周期 [s]"""
        return self.t_ds + self.t_ss

    @property
    def mean_speed(self) -> float:
        """! @brief 平均前進速度 [m/s]
        @note 1 歩で遊脚が step_length 進むと、重心は その半分だけ進む。
        """
        return 0.5 * self.step_length / self.period


@dataclass
class BipedSchedule:
    """! @brief 生成した歩容スケジュール。

    @param params      パラメータ [-]
    @param seg_kind    区間の種類 shape (n_seg,) [-]
    @param seg_t0      区間の開始時刻 shape (n_seg,) [s]
    @param seg_dur     区間長 shape (n_seg,) [s]
    @param seg_cop     区間の CoP（区分一定）shape (n_seg,2) [m]
    @param seg_swing   区間の遊脚（SS のみ有効、他は -1）shape (n_seg,) [-]
    @param foot_pos    各区間開始時の両足位置 shape (n_seg,nf,3) [m]
    @param swing_from  遊脚の離陸位置 shape (n_seg,3) [m]
    @param swing_to    遊脚の着地目標 shape (n_seg,3) [m]
    @note CoP 列と区間長をそのまま `DcmPlanner.plan()` に渡せる。
    """

    params: BipedGaitParams
    seg_kind: list[SegKind]
    seg_t0: np.ndarray
    seg_dur: np.ndarray
    seg_cop: np.ndarray
    seg_swing: np.ndarray
    foot_pos: np.ndarray
    swing_from: np.ndarray
    swing_to: np.ndarray

    @property
    def total_time(self) -> float:
        """! @brief 全体の時間 [s]"""
        return float(self.seg_t0[-1] + self.seg_dur[-1])

    def segment_at(self, t: float) -> int:
        """! @brief 時刻 t が属する区間の番号を返す。
        @param t 時刻 [s]
        @return 区間番号 [-]（範囲外は端に丸める）
        """
        if t <= self.seg_t0[0]:
            return 0
        i = int(np.searchsorted(self.seg_t0, t, side="right") - 1)
        return int(np.clip(i, 0, len(self.seg_kind) - 1))

    def state_at(self, t: float) -> tuple[GaitState, np.ndarray, np.ndarray, np.ndarray]:
        """! @brief 時刻 t の接触状態と足先目標を返す。
        @param t 時刻 [s]
        @return (GaitState, 両足の目標位置 shape (nf,3) [m],
                 遊脚の目標速度 shape (nf,3) [m/s], 立脚フラグ shape (nf,) [bool])
        @note 立脚の目標位置はその区間の足位置で固定する。遊脚は
              `gait.swing.swing_trajectory` の 5 次 + 3 次エルミート軌道に従う。
        """
        from .swing import swing_trajectory

        i = self.segment_at(t)
        kind = self.seg_kind[i]
        nf = self.foot_pos.shape[1]
        p_des = self.foot_pos[i].copy()
        v_des = np.zeros((nf, 3), dtype=np.float64)
        contact = np.ones(nf, dtype=bool)
        swing_ratio = np.zeros(nf, dtype=np.float64)
        leg_phase = np.zeros(nf, dtype=np.float64)

        if kind is SegKind.SS:
            sw = int(self.seg_swing[i])
            u = float(np.clip((t - self.seg_t0[i]) / self.seg_dur[i], 0.0, 1.0))
            p, v, _ = swing_trajectory(
                u, self.swing_from[i], self.swing_to[i],
                self.params.swing_height, float(self.seg_dur[i]),
            )
            p_des[sw] = p
            v_des[sw] = v
            contact[sw] = False
            swing_ratio[sw] = u
            leg_phase[sw] = u

        gs = GaitState(
            phase=float((t % self.params.period) / self.params.period),
            leg_phase=leg_phase,
            contact=contact,
            swing_ratio=swing_ratio,
            contact_table=np.zeros((0, nf), dtype=bool),  # 二足は MPC を使わない
            t_remain=np.full(nf, float(self.seg_t0[i] + self.seg_dur[i] - t)),
        )
        return gs, p_des, v_des, contact


def make_biped_schedule(
    params: BipedGaitParams, foot_init: np.ndarray | None = None
) -> BipedSchedule:
    """! @brief 足配置と CoP 参照を含む歩容スケジュールを作る。

    @param params    パラメータ [-]
    @param foot_init 初期の足裏中心位置 shape (nf,3) [m]。None なら
                     `(0, ±step_width/2, foot_z)` を使う
    @return BipedSchedule
    @note **foot_init は実機の公称姿勢で測った足先位置を渡すこと。**
          ここが実機とずれていると、IK が足を初期位置へ引き戻そうとして
          重心が前後に振られ、立っているだけで転倒する。
          G1 の公称姿勢では足裏中心が x = 0.046 m にあり、原点ではない。
    @note 構成は [静止] [DS, SS] x n_step [静止]。
          歩 i では `first_swing` から交互に足を踏み出し、着地位置は
          「その足の現在位置 + step_length」とする。1 歩で重心は半分進む。
          DS の CoP は前の支持足から次の支持足へ `ds_subdiv` 等分で動かす。
    """
    p = params
    w2 = 0.5 * p.step_width
    if foot_init is None:
        foot = np.array(
            [[0.0, +w2, p.foot_z], [0.0, -w2, p.foot_z]], dtype=np.float64
        )  # (nf,3) left, right
    else:
        foot = np.asarray(foot_init, dtype=np.float64).reshape(-1, 3).copy()

    kinds: list[SegKind] = []
    t0s: list[float] = []
    durs: list[float] = []
    cops: list[np.ndarray] = []
    swings: list[int] = []
    foots: list[np.ndarray] = []
    sw_from: list[np.ndarray] = []
    sw_to: list[np.ndarray] = []

    t = 0.0

    def push(kind: SegKind, dur: float, cop: np.ndarray, swing: int,
             f_from: np.ndarray | None = None, f_to: np.ndarray | None = None) -> None:
        nonlocal t
        kinds.append(kind)
        t0s.append(t)
        durs.append(float(dur))
        cops.append(np.asarray(cop, dtype=np.float64).reshape(2).copy())
        swings.append(int(swing))
        foots.append(foot.copy())
        sw_from.append(np.zeros(3) if f_from is None else np.asarray(f_from, float).copy())
        sw_to.append(np.zeros(3) if f_to is None else np.asarray(f_to, float).copy())
        t += float(dur)

    mid = 0.5 * (foot[0, 0:2] + foot[1, 0:2])
    push(SegKind.STAND, p.t_start, mid, -1)

    for i in range(int(p.n_step)):
        swing = p.first_swing if i % 2 == 0 else 1 - p.first_swing
        stance = 1 - swing
        cop_prev = cops[-1]
        cop_stance = foot[stance, 0:2].copy()

        # --- 両脚支持: CoP を前の値から支持足へ ds_subdiv 等分で動かす ---
        for k in range(1, int(p.ds_subdiv) + 1):
            a = k / float(p.ds_subdiv)
            push(SegKind.DS, p.t_ds / p.ds_subdiv,
                 (1.0 - a) * cop_prev + a * cop_stance, -1)

        # --- 単脚支持: 遊脚を踏み出す ---
        f_from = foot[swing].copy()
        f_to = f_from + np.array([p.step_length, 0.0, 0.0])
        push(SegKind.SS, p.t_ss, cop_stance, swing, f_from, f_to)
        foot[swing] = f_to  # 着地

    # --- 最後に両足を揃えてから静止（歩いていないときは何もしない）---
    if int(p.n_step) == 0:
        mid = 0.5 * (foot[0, 0:2] + foot[1, 0:2])
        push(SegKind.STAND, p.t_end, mid, -1)
        return BipedSchedule(
            params=p,
            seg_kind=kinds,
            seg_t0=np.asarray(t0s, dtype=np.float64),
            seg_dur=np.asarray(durs, dtype=np.float64),
            seg_cop=np.asarray(cops, dtype=np.float64),
            seg_swing=np.asarray(swings, dtype=int),
            foot_pos=np.asarray(foots, dtype=np.float64),
            swing_from=np.asarray(sw_from, dtype=np.float64),
            swing_to=np.asarray(sw_to, dtype=np.float64),
        )
    swing = p.first_swing if int(p.n_step) % 2 == 0 else 1 - p.first_swing
    stance = 1 - swing
    cop_prev = cops[-1]
    cop_stance = foot[stance, 0:2].copy()
    for k in range(1, int(p.ds_subdiv) + 1):
        a = k / float(p.ds_subdiv)
        push(SegKind.DS, p.t_ds / p.ds_subdiv,
             (1.0 - a) * cop_prev + a * cop_stance, -1)
    f_from = foot[swing].copy()
    f_to = np.array([foot[stance, 0], foot[swing, 1], p.foot_z])  # x を揃える
    push(SegKind.SS, p.t_ss, cop_stance, swing, f_from, f_to)
    foot[swing] = f_to

    mid = 0.5 * (foot[0, 0:2] + foot[1, 0:2])
    push(SegKind.STAND, p.t_end, mid, -1)

    return BipedSchedule(
        params=p,
        seg_kind=kinds,
        seg_t0=np.asarray(t0s, dtype=np.float64),
        seg_dur=np.asarray(durs, dtype=np.float64),
        seg_cop=np.asarray(cops, dtype=np.float64),
        seg_swing=np.asarray(swings, dtype=int),
        foot_pos=np.asarray(foots, dtype=np.float64),
        swing_from=np.asarray(sw_from, dtype=np.float64),
        swing_to=np.asarray(sw_to, dtype=np.float64),
    )


def gait_params_from_config(overrides: dict | None = None, section: str = "walk_3d") -> "BipedGaitParams":
    """! @brief configs/biped.yaml の歩容節から BipedGaitParams を作る（既定値を YAML に置く方針）。
    @param overrides 上書きする値（None の値は無視）[-]
    @param section   読む節 [-]
    @return BipedGaitParams
    """
    from ..common.config import load_yaml

    d = load_yaml("biped.yaml").get(section, {})
    kw = {k: d[k] for k in ("t_ds", "t_ss", "step_length", "step_width", "swing_height",
                            "n_step", "t_start", "t_end", "ds_subdiv") if k in d}
    if overrides:
        kw.update({k: v for k, v in overrides.items() if v is not None})
    return BipedGaitParams(**kw)
