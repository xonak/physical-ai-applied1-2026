"""! @file swing.py
@brief 遊脚軌道。

**必ず位相 [0,1] の関数として書く。** 時間の関数として書くと、周期や duty を
変えた瞬間に軌道が破綻する。

形状:
- 水平: 5 次のスムーズステップ s(u) = 6u^5 - 15u^4 + 10u^3
  s'(0)=s'(1)=0, s''(0)=s''(1)=0 なので、**離陸時と着地時の水平速度がゼロ**になる。
- 鉛直: 頂点 u=0.5 で 2 分割した 3 次エルミート。
  前半は z0 -> z_apex（両端で鉛直速度 0）、後半は z_apex -> z1（着地速度は指定の負値）。
  **着地時の垂直速度を負にする**ことで、接地検出の前に足が地面を探る形になる。

記号: u 遊脚区間内の進捗 [-] in [0,1]、T_sw 遊脚時間 [s]。
"""

from __future__ import annotations

import numpy as np

__all__ = ["smoothstep5", "hermite_cubic", "swing_trajectory", "SwingParams"]


class SwingParams:
    """! @brief 遊脚軌道のパラメータ。"""

    def __init__(self, height: float = 0.08, touchdown_speed: float = 0.15) -> None:
        """! @brief パラメータを保持する。
        @param height          遊脚高さ（頂点の持ち上げ量）[m]
        @param touchdown_speed 着地時の下向き速度の大きさ [m/s]
        @return なし
        """
        self.height = float(height)
        self.touchdown_speed = float(touchdown_speed)


def smoothstep5(u: float) -> tuple[float, float, float]:
    """! @brief 5 次スムーズステップとその 1・2 階微分。
    @param u 進捗 [-] in [0,1]
    @return (s, ds/du, d2s/du2) [-]
    @note s(0)=0, s(1)=1, s'(0)=s'(1)=0, s''(0)=s''(1)=0。
    """
    u = float(np.clip(u, 0.0, 1.0))
    s = u * u * u * (10.0 + u * (-15.0 + 6.0 * u))
    ds = 30.0 * u * u * (1.0 + u * (-2.0 + u))
    dds = 60.0 * u * (1.0 + u * (-3.0 + 2.0 * u))
    return s, ds, dds


def hermite_cubic(
    u: float, p0: float, p1: float, m0: float, m1: float
) -> tuple[float, float, float]:
    """! @brief 3 次エルミート補間とその 1・2 階微分。
    @param u  進捗 [-] in [0,1]
    @param p0 始端値 [m]
    @param p1 終端値 [m]
    @param m0 始端の du 微分 [m]
    @param m1 終端の du 微分 [m]
    @return (p, dp/du, d2p/du2) [m, m, m]
    """
    u = float(np.clip(u, 0.0, 1.0))
    u2, u3 = u * u, u * u * u
    h00, h10 = 2 * u3 - 3 * u2 + 1.0, u3 - 2 * u2 + u
    h01, h11 = -2 * u3 + 3 * u2, u3 - u2
    d00, d10 = 6 * u2 - 6 * u, 3 * u2 - 4 * u + 1.0
    d01, d11 = -6 * u2 + 6 * u, 3 * u2 - 2 * u
    e00, e10 = 12 * u - 6.0, 6 * u - 4.0
    e01, e11 = -12 * u + 6.0, 6 * u - 2.0
    return (
        h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1,
        d00 * p0 + d10 * m0 + d01 * p1 + d11 * m1,
        e00 * p0 + e10 * m0 + e01 * p1 + e11 * m1,
    )


def swing_trajectory(
    phase: float,
    p_start: np.ndarray,
    p_end: np.ndarray,
    height: float,
    t_swing: float,
    touchdown_speed: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """! @brief 遊脚の目標位置・速度・加速度を返す。
    @param phase           遊脚区間内の進捗 [-] in [0,1]
    @param p_start         離陸位置 {W} shape (3,) [m]
    @param p_end           着地目標 {W} shape (3,) [m]
    @param height          遊脚高さ [m]
    @param t_swing         遊脚時間 [s]
    @param touchdown_speed 着地時の下向き速度の大きさ [m/s]
    @return (p shape (3,) [m], v shape (3,) [m/s], a shape (3,) [m/s^2])
    @note 位相微分から時間微分への換算は d/dt = (1/T_sw) d/du。T_sw を変えても
          形状は変わらず、速度・加速度だけがスケールする。
    """
    u = float(np.clip(phase, 0.0, 1.0))
    p0 = np.asarray(p_start, dtype=np.float64).reshape(3)
    p1 = np.asarray(p_end, dtype=np.float64).reshape(3)
    T = max(float(t_swing), 1.0e-6)

    s, ds, dds = smoothstep5(u)
    d = p1 - p0
    p = p0 + d * s
    v = d * (ds / T)
    a = d * (dds / (T * T))

    z_apex = max(p0[2], p1[2]) + float(height)
    m_td = 0.5 * (-abs(float(touchdown_speed)) * T)  # 終端の dz/du（着地は下向き）
    if u < 0.5:
        z, dz, ddz = hermite_cubic(u / 0.5, p0[2], z_apex, 0.0, 0.0)
    else:
        z, dz, ddz = hermite_cubic((u - 0.5) / 0.5, z_apex, p1[2], 0.0, m_td)
    p[2] = z
    v[2] = dz * 2.0 / T
    a[2] = ddz * 4.0 / (T * T)
    return p, v, a
