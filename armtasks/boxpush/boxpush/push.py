"""! @file push.py
@brief 箱押しの手書きスケジュール（比較基準）を、方策の口（`boxpush.policy`）の上に載せたもの。
実験設定 YAML の `push` 節（力・タイミング・前傾）を `PushParams` に読み込む。

`PushParams` の中身は `ScheduleBoxPolicy`（時刻でランプする行動）と
`BoxPushApplier`（行動を計画器と手先タスクに適用）に分かれた。`run_experiment(extra_on_step=hook,
hand_task=hook.hand_task)` の使い方は変わらない。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .policy import BoxActionSpec, BoxPushApplier, ReactiveBoxPolicy, ScheduleBoxPolicy

__all__ = ["PushParams", "BoxPushHook"]


@dataclass
class PushParams:
    """! @brief 箱押しの手書き指令（比較基準）。
    @param f_push     目標押付け力（両手の合計）[N]
    @param t_start    押し始める時刻 [s]（それまでは手を面の手前 `standoff` に置く）
    @param t_ramp     押付け力・姿勢の立ち上がり時間 [s]
    @param standoff   接触前の手先 site の目標を面から手前に置く距離 [m]（ゴムの手は site より 0.067 m 前）
    @param dz         手先高さの構えからのずらし [m]（行動 hand_dz）
    @param dy         （未使用。左右のずらしは行動に無い）
    @param box_half_x 箱の x 半長 [m]
    @param lean_x     前傾 [m]（行動 lean_x）。None なら 0
    @param dz_com     重心高さの公称からのずらし [m]（行動 com_dz、負でしゃがむ）
    @param vx         押しながら歩く前進速度 [m/s]（0 で立位）。t_start から
    @param t_lean_lead 前傾・重心高さを押し始めより先に入れる時間 [s]
    @param lean_pre_max 先行して入れる前傾の上限 [m]（残りは押付け力と一緒にランプ）
    @param reactive   True なら前傾を手首 F/T の実測から決める（`ReactiveBoxPolicy`。lean_x は無視）
    @param ki, ki_limit 押付け力の積分補正 [1/s], 上限（指令比）
    @param kp, kd, f_max 手先インピーダンスの係数（`HandTask`）
    @param decimation 方策（スケジュール）を評価する周期 [-]
    """

    f_push: float = 40.0
    t_start: float = 2.0
    t_ramp: float = 1.0
    standoff: float = 0.10
    dz: float = 0.0
    dy: float = 0.0
    box_half_x: float = 0.25
    lean_x: float | None = None
    dz_com: float = 0.0
    vx: float = 0.0
    t_lean_lead: float = 1.5
    lean_pre_max: float = 0.05
    reactive: bool = False
    ki: float = 2.0
    ki_limit: float = 1.0
    kp: float = 300.0
    kd: float = 30.0
    f_max: float = 60.0
    decimation: int = 50

    def to_spec_and_policy(self) -> tuple[BoxActionSpec, ScheduleBoxPolicy]:
        """! @brief 行動仕様（7 行動を方策が出す形）と手書きスケジュール方策にする。"""
        spec = BoxActionSpec(actions=("step_dx", "step_dy", "step_dt", "f_push", "hand_dz", "lean_x", "com_dz"),
                             decimation=int(self.decimation), f_push_max=max(80.0, 2.0 * self.f_push))
        targets = {"f_push": float(self.f_push), "hand_dz": float(self.dz),
                   "lean_x": 0.0 if self.lean_x is None else float(self.lean_x), "com_dz": float(self.dz_com)}
        cls = ReactiveBoxPolicy if self.reactive else ScheduleBoxPolicy
        pol = cls(spec, targets=targets, t_start=self.t_start, t_ramp=self.t_ramp,
                  t_lean_lead=self.t_lean_lead, lean_pre_max=self.lean_pre_max)
        if self.reactive:
            pol.lean_min = self.lean_pre_max
            pol.dt_decide = self.decimation * 0.002
        return spec, pol


class BoxPushHook(BoxPushApplier):
    """! @brief 手書きスケジュールの毎周期フック: PushParams から Schedule 方策と適用側を作る。
    @param par PushParams [-]
    @param dt  制御周期 [s]
    @note 属性 `par`、`f_push_at(t)`、`hand_task` は run_box とテストが使う。
    """

    def __init__(self, par: PushParams, dt: float = 0.002, tau_max: np.ndarray | None = None, log: list | None = None) -> None:
        from lococtrl.control.hand_task import HandTask

        spec, pol = par.to_spec_and_policy()
        super().__init__(spec=spec, policy=pol, hand_task=HandTask(n_arm=2, kp=par.kp, kd=par.kd, f_max=par.f_max),
                         dt=dt, box_half_x=par.box_half_x, standoff=par.standoff, ki=par.ki, ki_limit=par.ki_limit,
                         vx=par.vx, t_walk=par.t_start, log=log, tau_max=tau_max)
        self.par = par
        self.schedule = pol

    def f_push_at(self, t: float) -> float:
        """! @brief 時刻 t の押付け力指令 [N]（スケジュール）。"""
        return float(self.schedule.values_at(t)["f_push"])
