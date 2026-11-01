"""! @file dcm.py
@brief 発散成分（DCM）による重心軌道生成と、予見制御との等価性検証。

**このファイルは MuJoCo に触れない。** NumPy だけで完結する。

LIPM の水平運動は

```
c_ddot = omega^2 (c - p),   omega = sqrt(g / z_c)
```

であり、これは固有値 `+omega`（発散）と `-omega`（収束）に分解できる。
発散側の座標が DCM（Divergent Component of Motion, Capture Point）である。

```
xi      = c + c_dot / omega          # DCM [m]
xi_dot  = omega (xi - p)             # 発散（不安定極 +omega）
c_dot   = -omega (c - xi)            # 収束（安定極 -omega）
```

**DCM 参照は終端条件から後ろ向きに構成する。** 前向きに数値積分すると
不安定極 `+omega` がそのまま出て発散するので、必ず再帰で解く。
各歩 i の CoP を `p_i`（区間内で一定）、区間長を `T_i` とすると

```
xi(t)        = p_i + (xi_eos_i - p_i) exp(omega (t - t_eos_i))
xi_eos_{i-1} = p_i + (xi_eos_i - p_i) exp(-omega T_i)
xi_eos_N     = p_N                   # 最終歩は最終 CoP で停止する
```

重心は上の `xi(t)` を入力とする 1 次系として**閉形式で**解ける。
`tau = t - t_0`、`A = xi(t_0) - p_i` とおくと

```
c(t) = p_i + (c(t_0) - p_i) exp(-omega tau) + A sinh(omega tau)
```

（`c_dot = omega (xi - c)` を定数変化法で解いた結果。数値積分は不要。）

記号: n_step 歩数 [-]、T 時刻ステップ数 [-]。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["DcmPlan", "DcmPlanner"]


@dataclass
class DcmPlan:
    """! @brief DCM による軌道生成の結果。

    @param t      時刻 shape (T,) [s]
    @param c      重心位置 shape (T,) [m]
    @param c_dot  重心速度 shape (T,) [m/s]
    @param xi     DCM shape (T,) [m]
    @param xi_dot DCM 速度 shape (T,) [m/s]
    @param p      CoP（= ZMP）shape (T,) [m]
    @param xi_eos 各歩の終端 DCM shape (n_step,) [m]
    """

    t: np.ndarray
    c: np.ndarray
    c_dot: np.ndarray
    xi: np.ndarray
    xi_dot: np.ndarray
    p: np.ndarray
    xi_eos: np.ndarray

    @property
    def c_ddot(self) -> np.ndarray:
        """! @brief 重心加速度 shape (T,) [m/s^2]（`omega^2 (c - p)`）"""
        raise NotImplementedError  # omega を持たないので DcmPlanner 側で出す


class DcmPlanner:
    """! @brief 区分一定 CoP に対する DCM 参照と重心軌道の閉形式生成。"""

    def __init__(self, z_c: float = 0.8, g: float = 9.81) -> None:
        """! @brief 重心高さから omega を決める。
        @param z_c 重心高さ [m]
        @param g   重力加速度 [m/s^2]
        @return なし
        """
        self.z_c = float(z_c)
        self.g = float(g)
        self.omega = float(np.sqrt(self.g / self.z_c))

    # ------------------------------------------------------------------ 後ろ向き
    def end_of_step_dcm(self, cop: np.ndarray, duration: np.ndarray) -> np.ndarray:
        """! @brief 各歩の終端 DCM を**最終歩から後ろ向きに**再帰で求める。
        @param cop      各歩の CoP shape (n_step,) [m]
        @param duration 各歩の区間長 shape (n_step,) [s]
        @return xi_eos shape (n_step,) [m]
        @note `xi_eos_{i-1} = p_i + (xi_eos_i - p_i) exp(-omega T_i)`、
              `xi_eos_{N-1} = p_{N-1}`（最終 CoP で停止）。

              **前向きに積分してはいけない。** DCM の極は `+omega` なので、
              初期条件から前向きに解くと必ず発散する。後ろ向き再帰では
              `exp(-omega T_i) < 1` が掛かるので数値的にも安定である。
        """
        p = np.asarray(cop, dtype=np.float64).ravel()
        T = np.asarray(duration, dtype=np.float64).ravel()
        if p.size != T.size:
            raise ValueError(f"cop と duration の長さが違う: {p.size} vs {T.size}")
        n = p.size
        xi_eos = np.zeros(n, dtype=np.float64)
        xi_eos[n - 1] = p[n - 1]  # 終端条件: 最終 CoP で停止
        for i in range(n - 1, 0, -1):
            xi_eos[i - 1] = p[i] + (xi_eos[i] - p[i]) * np.exp(-self.omega * T[i])
        return xi_eos

    # ------------------------------------------------------------------ 前向き
    def plan(
        self,
        cop: np.ndarray,
        duration: np.ndarray,
        dt: float,
        c0: float | None = None,
        c_dot0: float = 0.0,
    ) -> DcmPlan:
        """! @brief 区分一定 CoP から DCM 参照と重心軌道を閉形式で作る。
        @param cop      各歩の CoP shape (n_step,) [m]
        @param duration 各歩の区間長 shape (n_step,) [s]
        @param dt       出力の時間刻み [s]
        @param c0       初期重心位置 [m]（None で `cop[0]`）
        @param c_dot0   初期重心速度 [m/s]
        @return DcmPlan
        @note 重心は各区間で閉形式
              `c = p + (c_0 - p) exp(-omega tau) + (xi_0 - p) sinh(omega tau)`
              で進める。数値積分ではないので dt を粗くしても誤差が増えない。
        """
        p = np.asarray(cop, dtype=np.float64).ravel()
        T = np.asarray(duration, dtype=np.float64).ravel()
        xi_eos = self.end_of_step_dcm(p, T)
        w = self.omega

        n_k = [int(round(float(ti) / float(dt))) for ti in T]
        n_t = int(sum(n_k))
        t = np.arange(n_t, dtype=np.float64) * dt
        c = np.zeros(n_t)
        cd = np.zeros(n_t)
        xi = np.zeros(n_t)
        xid = np.zeros(n_t)
        pp = np.zeros(n_t)

        c_cur = float(p[0] if c0 is None else c0)
        cd_cur = float(c_dot0)
        k0 = 0
        for i in range(p.size):
            # 区間開始時の DCM。xi_eos[i] は区間終端なので T_i だけ戻す
            xi_0 = p[i] + (xi_eos[i] - p[i]) * np.exp(-w * T[i])
            for k in range(n_k[i]):
                tau = k * dt
                e_m = np.exp(-w * tau)
                xi_k = p[i] + (xi_0 - p[i]) * np.exp(w * tau)
                c_k = p[i] + (c_cur - p[i]) * e_m + (xi_0 - p[i]) * np.sinh(w * tau)
                cd_k = w * (xi_k - c_k)
                idx = k0 + k
                c[idx], cd[idx], xi[idx], xid[idx], pp[idx] = (
                    c_k,
                    cd_k,
                    xi_k,
                    w * (xi_k - p[i]),
                    p[i],
                )
            # 区間終端の値を次区間の初期条件にする
            tau = n_k[i] * dt
            c_cur = p[i] + (c_cur - p[i]) * np.exp(-w * tau) + (xi_0 - p[i]) * np.sinh(w * tau)
            xi_end = p[i] + (xi_0 - p[i]) * np.exp(w * tau)
            cd_cur = w * (xi_end - c_cur)
            k0 += n_k[i]

        return DcmPlan(t=t, c=c, c_dot=cd, xi=xi, xi_dot=xid, p=pp, xi_eos=xi_eos)

    def com_accel(self, plan: DcmPlan) -> np.ndarray:
        """! @brief 重心加速度を返す。
        @param plan DcmPlan [-]
        @return c_ddot shape (T,) [m/s^2]
        @note LIPM の運動方程式 `c_ddot = omega^2 (c - p)` そのもの。
        """
        return (self.omega**2) * (plan.c - plan.p)
