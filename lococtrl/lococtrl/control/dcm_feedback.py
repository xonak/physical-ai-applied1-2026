"""! @file dcm_feedback.py
@brief DCM フィードバック安定化則と、2D での外乱応答評価。

予見制御（Kajita 2003）や DCM 閉形式（`planner.dcm`）が軌道生成だけを扱うのに対し、
本モジュールは**外乱**に対する安定化を扱う（Englsberger 2015）。

## 制御則

LIPM の DCM を `xi = c + c_dot / omega`、`omega = sqrt(g / z_c)` とすると

    xi_dot = omega (xi - p)

であり、極 `+omega` の不安定系である。これに対し

    p_cmd = xi_hat + k (xi_hat - xi_ref) - xi_ref_dot / omega      (k > 0)

を与えると、誤差 `e = xi - xi_ref` の閉ループは

    e_dot = -k omega e

となり、極が `-k omega` に移る。`K = 1 + k` と書いたときの `K > 1` が安定条件であり、
**`K < 1` では「見た目は追従しているのに発散する」**という最悪の挙動になる
（`closed_loop_pole()` が正を返すので、必ず計算してログに出すこと）。

## 外乱の統一表現

本モジュールは外乱を**等価 CoP シフト `d` [m]** に統一して扱う。

    xi_dot = omega (xi - p_cmd + d)

- 水平外力 `f` [N]: `d = f / (m omega^2) = f z_c / (m g)`
- 上体の角運動量変化率 `Ldot` [N·m]: `d = -Ldot / (m g)`

後者は、上半身（腕）の運動が**そのまま下半身への外乱として入る**経路である。
腕を振れば `Ldot` が立ち、等価 CoP がずれる。`UpperBodyMomentumDisturbance` がこの経路を表し、
トルク制御の試験や学習時の外乱注入で共通に使える。

@note 本モジュールは MuJoCo に依存しない（制御層は sim/ と bench/ 以外
      からシミュレータを触らない）。検証用の `simulate_lipm_2d()` も純粋な数値積分である。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np

# 重力加速度 [m/s^2]。configs/biped.yaml と一致させること。
_G_DEFAULT = 9.81


# ======================================================================
# 制御則（状態を持たない純関数）
# ======================================================================
def closed_loop_pole(k: float, omega: float) -> float:
    """! @brief DCM フィードバックの閉ループ極を返す。

    @param k      フィードバックゲイン [-]。`K = 1 + k` の `k` の側
    @param omega  LIPM 固有振動数 sqrt(g / z_c) [rad/s]
    @return 閉ループ極 [1/s]。`k > 0` で負（安定）、`k < 0`（= K < 1）で正（発散）
    @note 時定数は `1 / (k omega)` [s]。
    """
    return -k * omega


def dcm_feedback_cop(
    xi_hat: np.ndarray,
    xi_ref: np.ndarray,
    xi_ref_dot: np.ndarray,
    omega: float,
    k: float,
) -> np.ndarray:
    """! @brief DCM フィードバック則による CoP 指令を計算する。

        p_cmd = xi_hat + k (xi_hat - xi_ref) - xi_ref_dot / omega

    @param xi_hat      推定 DCM {W} の水平成分 shape (2,) [m]
    @param xi_ref      参照 DCM           shape (2,) [m]
    @param xi_ref_dot  参照 DCM の時間微分 shape (2,) [m/s]
    @param omega       LIPM 固有振動数 [rad/s]
    @param k           フィードバックゲイン [-]
    @return CoP 指令 {W} 水平成分 shape (2,) [m]（支持多角形への clamp は呼び出し側）
    @note 等価形 `p_cmd = xi_ref + (1+k)(xi_hat - xi_ref) - xi_ref_dot/omega` と
          数値的に一致する。
    """
    xi_hat = np.asarray(xi_hat, dtype=np.float64)
    xi_ref = np.asarray(xi_ref, dtype=np.float64)
    xi_ref_dot = np.asarray(xi_ref_dot, dtype=np.float64)
    return xi_hat + k * (xi_hat - xi_ref) - xi_ref_dot / omega


def clamp_cop(
    p_cmd: np.ndarray,
    support_center: np.ndarray,
    half_extent: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """! @brief CoP 指令を支持多角形（矩形近似）に clamp する。

    @param p_cmd          CoP 指令 shape (2,) [m]
    @param support_center 支持多角形の中心 shape (2,) [m]
    @param half_extent    前後・左右の半長 shape (2,) [m]
    @return (clamp 後の CoP shape (2,) [m], 飽和量 shape (2,) [m])
    @note 飽和量 `p_cmd - p_clamped` は着地位置・タイミング修正へ回す量そのもの。
          捨てずに返すのはそのためである。
    """
    p_cmd = np.asarray(p_cmd, dtype=np.float64)
    lo = support_center - half_extent
    hi = support_center + half_extent
    p_clamped = np.clip(p_cmd, lo, hi)
    return p_clamped, p_cmd - p_clamped


# ======================================================================
# 外乱モデル（等価 CoP シフト [m] に統一）
# ======================================================================
class Disturbance(Protocol):
    """! @brief 外乱の共通インタフェース。等価 CoP シフトを返す。"""

    def cop_shift(self, t: float) -> np.ndarray:
        """! @brief 時刻 t における等価 CoP シフトを返す。
        @param t 時刻 [s]
        @return 等価 CoP シフト {W} 水平成分 shape (2,) [m]
        """
        ...


@dataclass
class ImpulseDisturbance:
    """! @brief 重心への水平インパルス外乱。

    力積を主変数にする（継続時間を変えると比較にならないため、
    継続時間は固定し力積で掃引する）。

    @param impulse   力積の大きさ [N*s]
    @param direction 方向 {W} 水平 shape (2,) 単位ベクトル [-]
    @param t_apply   印加開始時刻 [s]
    @param duration  継続時間 [s]
    @param mass      全質量 [kg]
    @param omega     LIPM 固有振動数 [rad/s]
    """

    impulse: float
    direction: np.ndarray
    t_apply: float
    duration: float
    mass: float
    omega: float

    def cop_shift(self, t: float) -> np.ndarray:
        """! @brief 等価 CoP シフト `d = f / (m omega^2)` を返す。
        @param t 時刻 [s]
        @return 等価 CoP シフト shape (2,) [m]
        """
        if not (self.t_apply <= t < self.t_apply + self.duration):
            return np.zeros(2, dtype=np.float64)
        force = self.impulse / self.duration  # [N]
        d = np.asarray(self.direction, dtype=np.float64)
        return force * d / (self.mass * self.omega**2)


@dataclass
class UpperBodyMomentumDisturbance:
    """! @brief 上体の角運動量変化率による外乱（腕の運動の等価表現）。

    上半身（腕）を動かすと、その運動が重心まわりの角運動量
    変化率 `Ldot` を生み、ZMP を `-Ldot / (m g)` だけずらす。LIPM の「角運動量ゼロ」
    仮定が破れる経路そのものであり、**下半身の安定化器から見れば外乱である**。

    @param ldot_fn 時刻 -> 重心まわり角運動量変化率 shape (2,) [N*m]
                    （x 成分がロール軸、y 成分がピッチ軸まわり）
    @param mass    全質量 [kg]
    @param gravity 重力加速度 [m/s^2]
    @note 符号規約: `d_x = -Ldot_y / (m g)`、`d_y = +Ldot_x / (m g)`。
          前方（+x）へ ZMP をずらすのはピッチ軸まわりの負の角運動量変化である。
    @note 学習時にこれを外乱として注入する場合、`ldot_fn` に操作ログの再生や
          ランダム過程を与えれば、そのままドメインランダム化の一要素になる。
    """

    ldot_fn: Callable[[float], np.ndarray]
    mass: float
    gravity: float = _G_DEFAULT

    def cop_shift(self, t: float) -> np.ndarray:
        """! @brief 等価 CoP シフトを返す。
        @param t 時刻 [s]
        @return 等価 CoP シフト shape (2,) [m]
        """
        ldot = np.asarray(self.ldot_fn(t), dtype=np.float64)  # [N*m]
        scale = self.mass * self.gravity
        return np.array([-ldot[1] / scale, ldot[0] / scale], dtype=np.float64)


@dataclass
class SumDisturbance:
    """! @brief 複数の外乱の重ね合わせ。
    @param parts 外乱の列
    """

    parts: list

    def cop_shift(self, t: float) -> np.ndarray:
        """! @brief 各外乱の等価 CoP シフトの総和を返す。
        @param t 時刻 [s]
        @return 等価 CoP シフト shape (2,) [m]
        """
        total = np.zeros(2, dtype=np.float64)
        for part in self.parts:
            total += part.cop_shift(t)
        return total


# ======================================================================
# 安定化器（状態を持つ）
# ======================================================================
@dataclass
class DcmFeedbackParams:
    """! @brief DCM フィードバックの設定。

    @param k            フィードバックゲイン [-]。安定条件は k > 0（K = 1 + k > 1）
    @param omega        LIPM 固有振動数 [rad/s]
    @param half_extent  支持多角形の半長（前後, 左右） shape (2,) [m]
    @param enabled      False でフィードバックを切る（p_cmd = p_ref。比較基準用）
    """

    k: float
    omega: float
    half_extent: np.ndarray
    enabled: bool = True


@dataclass
class DcmStabilizer:
    """! @brief DCM フィードバック安定化器。飽和量を累積して着地位置の修正へ渡す。

    @param params 設定
    """

    params: DcmFeedbackParams
    saturation: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=np.float64)
    )
    n_saturated: int = 0

    def __post_init__(self) -> None:
        """! @brief 安定条件を起動時に検査する。
        @note `k <= 0`（K <= 1）は見た目が追従しているのに発散する最悪の挙動を招くため、
              黙って通さず例外にする。掃引試験で意図的に使う場合は `allow_unstable` を
              別途設ける（現状は設けていない）。
        """
        if self.params.enabled and self.params.k <= 0.0:
            raise ValueError(
                f"k must be > 0 for stability (K = 1 + k > 1); got k = {self.params.k}. "
                f"closed-loop pole would be {closed_loop_pole(self.params.k, self.params.omega):+.3f} 1/s"
            )

    def compute(
        self,
        xi_hat: np.ndarray,
        xi_ref: np.ndarray,
        xi_ref_dot: np.ndarray,
        p_ref: np.ndarray,
        support_center: np.ndarray,
    ) -> np.ndarray:
        """! @brief CoP 指令を計算し、支持多角形へ clamp して返す。

        @param xi_hat         推定 DCM shape (2,) [m]
        @param xi_ref         参照 DCM shape (2,) [m]
        @param xi_ref_dot     参照 DCM の微分 shape (2,) [m/s]
        @param p_ref          参照 CoP shape (2,) [m]（フィードバック無効時に使う）
        @param support_center 支持多角形中心 shape (2,) [m]
        @return clamp 後の CoP 指令 shape (2,) [m]
        """
        if not self.params.enabled:
            p_raw = np.asarray(p_ref, dtype=np.float64)
        else:
            p_raw = dcm_feedback_cop(
                xi_hat, xi_ref, xi_ref_dot, self.params.omega, self.params.k
            )
        p_cop, sat = clamp_cop(p_raw, support_center, self.params.half_extent)
        self.saturation = sat
        if np.any(np.abs(sat) > 1e-12):
            self.n_saturated += 1
        return p_cop


# ======================================================================
# 検証用 2D LIPM シミュレータ
# ======================================================================
@dataclass
class LipmSimResult:
    """! @brief `simulate_lipm_2d` の返り値。

    @param t        時刻 shape (N,) [s]
    @param xi       DCM shape (N, 2) [m]
    @param c        重心 shape (N, 2) [m]
    @param p_cop    実現 CoP shape (N, 2) [m]
    @param d        等価 CoP 外乱 shape (N, 2) [m]
    @param captured 支持多角形内に収束したか [-]
    @param n_saturated CoP が飽和した制御周期数 [-]
    """

    t: np.ndarray
    xi: np.ndarray
    c: np.ndarray
    p_cop: np.ndarray
    d: np.ndarray
    captured: bool
    n_saturated: int


def simulate_lipm_2d(
    params: DcmFeedbackParams,
    mass: float,
    duration: float,
    dt: float = 0.002,
    disturbance: Disturbance | None = None,
    xi0: np.ndarray | None = None,
    c0: np.ndarray | None = None,
    support_center: np.ndarray | None = None,
    settle_tol: float = 0.01,
    settle_time: float = 2.0,
) -> LipmSimResult:
    """! @brief 静止立位（参照 DCM = 支持多角形中心）での外乱応答を数値積分する。

    支配方程式は等価 CoP 外乱 `d` を含む形

        xi_dot = omega (xi - p_cop + d)
        c_dot  = -omega (c - xi)

    @param params         安定化器の設定
    @param mass           全質量 [kg]
    @param duration       シミュレーション長 [s]
    @param dt             刻み [s]
    @param disturbance    外乱。None なら無外乱
    @param xi0            初期 DCM shape (2,) [m]。None なら支持多角形中心
    @param c0             初期重心 shape (2,) [m]。None なら支持多角形中心
    @param support_center 支持多角形中心 shape (2,) [m]。None なら原点
    @param settle_tol     収束判定の閾値 [m]
    @param settle_time    外乱印加後、この時間までに収束すれば成功 [s]
    @return LipmSimResult
    @note 「転倒」は 2D LIPM では明示的に表せないので、**DCM が支持多角形を出たまま
          戻らない**ことを失敗の判定に使う。CoP は支持多角形内に clamp されるので、
          `|xi| > half_extent` の領域では `xi_dot` の符号が外向きに固定され、
          原理的に引き戻せない（capturability の定義そのもの）。
    """
    omega = params.omega
    n = int(round(duration / dt))
    sc = np.zeros(2) if support_center is None else np.asarray(support_center, float)
    xi = sc.copy() if xi0 is None else np.asarray(xi0, dtype=np.float64).copy()
    c = sc.copy() if c0 is None else np.asarray(c0, dtype=np.float64).copy()

    stab = DcmStabilizer(params)
    # 静止立位なので参照 DCM は支持多角形中心で固定、その微分はゼロ。
    xi_ref = sc.copy()
    xi_ref_dot = np.zeros(2, dtype=np.float64)
    p_ref = sc.copy()

    ts = np.zeros(n)
    xis = np.zeros((n, 2))
    cs = np.zeros((n, 2))
    ps = np.zeros((n, 2))
    ds = np.zeros((n, 2))

    for i in range(n):
        t = i * dt
        d = (
            np.zeros(2, dtype=np.float64)
            if disturbance is None
            else disturbance.cop_shift(t)
        )
        p_cop = stab.compute(xi, xi_ref, xi_ref_dot, p_ref, sc)

        ts[i], xis[i], cs[i], ps[i], ds[i] = t, xi, c, p_cop, d

        # 前進オイラー（dt = 2 ms、omega ~ 3.5 rad/s なので十分）
        xi = xi + dt * omega * (xi - p_cop + d)
        c = c + dt * (-omega) * (c - xi)

        if not np.all(np.isfinite(xi)):
            break

    # 収束判定: 最後の 0.2 s で参照 DCM の近傍に留まっているか
    tail = max(1, int(0.2 / dt))
    err_tail = np.linalg.norm(xis[-tail:] - xi_ref, axis=1)
    captured = bool(
        np.all(np.isfinite(xis[-tail:])) and np.max(err_tail) < settle_tol
    )

    return LipmSimResult(
        t=ts, xi=xis, c=cs, p_cop=ps, d=ds,
        captured=captured, n_saturated=stab.n_saturated,
    )


def max_tolerable_impulse(
    params: DcmFeedbackParams,
    mass: float,
    direction: np.ndarray,
    t_apply: float = 0.5,
    pulse_duration: float = 0.03,
    sim_duration: float = 4.0,
    dt: float = 0.002,
    j_lo: float = 0.0,
    j_hi: float = 200.0,
    n_bisect: int = 24,
) -> float:
    """! @brief 許容最大外乱力積を二分探索で求める。

    @param params         安定化器の設定
    @param mass           全質量 [kg]
    @param direction      外乱方向 {W} 水平 shape (2,) 単位ベクトル [-]
    @param t_apply        印加時刻 [s]
    @param pulse_duration パルス継続時間 [s]（既定 30 ms）
    @param sim_duration   シミュレーション長 [s]
    @param dt             刻み [s]
    @param j_lo           探索下限 [N*s]
    @param j_hi           探索上限 [N*s]
    @param n_bisect       二分回数 [-]
    @return 許容最大力積 [N*s]。これを超えると支持多角形内に収束しない
    @note 単調性（力積が大きいほど失敗しやすい）を仮定している。上限で成功して
          しまった場合は探索範囲が不足しているので例外を出す。
    """

    def succeeds(j: float) -> bool:
        dist = ImpulseDisturbance(
            impulse=j, direction=np.asarray(direction, dtype=np.float64),
            t_apply=t_apply, duration=pulse_duration,
            mass=mass, omega=params.omega,
        )
        return simulate_lipm_2d(
            params, mass, sim_duration, dt=dt, disturbance=dist
        ).captured

    if succeeds(j_hi):
        raise ValueError(
            f"upper bound {j_hi} N*s still succeeds; widen the search range"
        )
    if not succeeds(j_lo + 1e-9):
        return 0.0

    for _ in range(n_bisect):
        mid = 0.5 * (j_lo + j_hi)
        if succeeds(mid):
            j_lo = mid
        else:
            j_hi = mid
    return j_lo
