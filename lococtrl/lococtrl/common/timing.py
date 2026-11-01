"""! @file timing.py
@brief 実時間ループのペーサと周期ジッタ統計。

実時間で走らせる場合は LoopPacer.sleep() を毎周期呼ぶ。
シミュレーション時間で走らせる場合（バッチ評価）は realtime=False とし、
統計だけを記録する。
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

__all__ = ["JitterStats", "LoopPacer"]


@dataclass
class JitterStats:
    """! @brief 周期ジッタの統計量。

    @param n        計測周期数 [-]
    @param dt_mean  実測周期の平均 [s]
    @param dt_std   実測周期の標準偏差 [s]
    @param dt_max   実測周期の最大 [s]
    @param overrun  周期超過（dt > period）の回数 [-]
    @param ratio_std 標準偏差 / 公称周期 [-]
    """

    n: int
    dt_mean: float
    dt_std: float
    dt_max: float
    overrun: int
    ratio_std: float


class LoopPacer:
    """! @brief 一定周期でループを回すためのペーサ。"""

    def __init__(
        self,
        period: float,
        realtime: bool = True,
        capacity: int = 200_000,
        spin_margin: float | None = None,
    ) -> None:
        """! @brief ペーサを構築する。
        @param period      制御周期 [s]
        @param realtime    True なら実時間で待つ。False なら統計のみ記録
        @param capacity    ジッタ記録の最大周期数 [-]
        @param spin_margin 周期境界の手前でスピン待ちに切り替える時間 [s]
                           （None なら min(period/2, 1 ms)）
        @return なし
        @note time.sleep の分解能（Linux で数百 us、Windows でさらに粗い）では
              制御周期 2 ms のジッタ 5% を満たせない。境界直前だけスピンする
              ハイブリッド待機とし、CPU 占有時間を spin_margin に限定する。
              margin を大きくするほどジッタは減るが CPU を食う。
        """
        if period <= 0.0:
            raise ValueError("LoopPacer: period は正でなければならない")
        self.period = float(period)
        self.realtime = bool(realtime)
        self.spin_margin = (
            min(0.5 * self.period, 1.0e-3) if spin_margin is None else float(spin_margin)
        )
        self._dt = np.full(capacity, np.nan, dtype=np.float64)
        self._n = 0
        self._t_next: float | None = None
        self._t_prev: float | None = None

    def reset(self) -> None:
        """! @brief 基準時刻と統計を初期化する。
        @return なし
        """
        self._t_next = None
        self._t_prev = None
        self._dt[:] = np.nan
        self._n = 0

    def sleep(self) -> float:
        """! @brief 次の周期境界まで待ち、実測周期を返す。
        @return dt 直前の周期からの経過時間 [s]（初回は period）
        @note 遅延が 1 周期を超えた場合は基準時刻を現在時刻に張り直す
              （遅れを取り戻そうとして連続実行するのを防ぐ）。
        """
        now = time.perf_counter()
        if self._t_next is None:
            self._t_next = now + self.period
            self._t_prev = now
            return self.period

        if self.realtime:
            remain = self._t_next - now - self.spin_margin
            if remain > 0.0:
                time.sleep(remain)
            while time.perf_counter() < self._t_next:  # 境界直前のみスピン
                pass
            now = time.perf_counter()
            self._t_next += self.period
            if now > self._t_next:  # 大きく遅れた: 位相を張り直す
                self._t_next = now + self.period

        dt = now - float(self._t_prev)
        self._t_prev = now
        if self._n < self._dt.size:
            self._dt[self._n] = dt
        self._n += 1
        return dt

    def stats(self) -> JitterStats:
        """! @brief 記録した実測周期からジッタ統計を計算する。
        @return JitterStats
        """
        d = self._dt[: min(self._n, self._dt.size)]
        d = d[np.isfinite(d)]
        if d.size == 0:
            return JitterStats(0, float("nan"), float("nan"), float("nan"), 0, float("nan"))
        std = float(np.std(d))
        return JitterStats(
            n=int(d.size),
            dt_mean=float(np.mean(d)),
            dt_std=std,
            dt_max=float(np.max(d)),
            overrun=int(np.sum(d > self.period)),
            ratio_std=std / self.period,
        )
