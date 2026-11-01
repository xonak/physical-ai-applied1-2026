"""! @file metrics.py
@brief 評価指標（CoP・支持多角形・摩擦錐余裕・姿勢誤差ほか）。

本モジュールは MuJoCo にもモデルにも依存しない純関数の集合とする。

記号: nf 脚数 [-]、{W} ワールド系。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..common.frames import log_so3, quat_to_rotmat

__all__ = [
    "center_of_pressure",
    "foot_support_polygon",
    "convex_hull_2d",
    "polygon_centroid",
    "point_in_polygon",
    "distance_to_polygon_edges",
    "support_polygon_inradius",
    "cop_margin",
    "friction_margin",
    "orientation_error",
    "rms",
    "cost_of_transport",
    "ImpulseTolerance",
    "bisect_max_impulse",
    "traversal_rate",
    "fall_rate",
    "peak_torque",
    "peak_contact_force",
]

_EPS = 1.0e-12


def center_of_pressure(
    p_WF: np.ndarray, f_W: np.ndarray, contact: np.ndarray, z_plane: float = 0.0
) -> np.ndarray:
    """! @brief 接触力から圧力中心 (CoP) を求める。
    @param p_WF    足先位置 {W} shape (nf,3) [m]
    @param f_W     足先接触力 {W} shape (nf,3) [N]
    @param contact 接触フラグ shape (nf,) [bool]
    @param z_plane CoP を評価する水平面の高さ [m]
    @return cop shape (3,) [m]（接触が無ければ NaN）
    @note 平面 z=z_plane まわりのモーメントを 0 とおく:
          x = sum(x_i f_zi - (z_i - z0) f_xi) / sum(f_zi)
          y = sum(y_i f_zi - (z_i - z0) f_yi) / sum(f_zi)
          接線力が生むモーメントを無視しないため、足先が浮いた高さにあっても正しい。
    """
    p = np.asarray(p_WF, dtype=np.float64).reshape(-1, 3)
    f = np.asarray(f_W, dtype=np.float64).reshape(-1, 3)
    m = np.asarray(contact, dtype=bool).reshape(-1)
    fz = float(np.sum(f[m, 2]))
    if not np.any(m) or fz <= _EPS:
        return np.full(3, np.nan)
    dz = p[m, 2] - z_plane
    x = float(np.sum(p[m, 0] * f[m, 2] - dz * f[m, 0])) / fz
    y = float(np.sum(p[m, 1] * f[m, 2] - dz * f[m, 1])) / fz
    return np.array([x, y, z_plane], dtype=np.float64)


def convex_hull_2d(pts: np.ndarray) -> np.ndarray:
    """! @brief 2 次元点集合の凸包（Andrew の monotone chain）。
    @param pts 点集合 shape (k,2) [m]
    @return 凸包の頂点 shape (h,2) [m]（反時計回り、重複なし）
    @note SciPy に依存しないのは、3 点以下でも例外を出さずに扱いたいためである。
    """
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if p.shape[0] <= 2:
        return p.copy()
    order = np.lexsort((p[:, 1], p[:, 0]))
    p = p[order]

    def _cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    lower: list[np.ndarray] = []
    for q in p:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], q) <= 0.0:
            lower.pop()
        lower.append(q)
    upper: list[np.ndarray] = []
    for q in p[::-1]:
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], q) <= 0.0:
            upper.pop()
        upper.append(q)
    hull = np.array(lower[:-1] + upper[:-1], dtype=np.float64)
    return hull if hull.size else p


def polygon_centroid(poly: np.ndarray) -> np.ndarray:
    """! @brief 多角形の頂点重心を返す。
    @param poly 頂点 shape (h,2) [m]
    @return 重心 shape (2,) [m]
    """
    return np.asarray(poly, dtype=np.float64).reshape(-1, 2).mean(axis=0)


def point_in_polygon(pt: np.ndarray, poly: np.ndarray) -> bool:
    """! @brief 点が凸多角形（反時計回り）の内部にあるか判定する。
    @param pt   点 shape (2,) [m]
    @param poly 凸多角形の頂点 shape (h,2) [m]
    @return True なら内部（辺上を含む）
    """
    q = np.asarray(pt, dtype=np.float64).reshape(2)
    v = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    h = v.shape[0]
    if h < 3:
        return False
    for i in range(h):
        a, b = v[i], v[(i + 1) % h]
        if (b[0] - a[0]) * (q[1] - a[1]) - (b[1] - a[1]) * (q[0] - a[0]) < -1.0e-9:
            return False
    return True


def distance_to_polygon_edges(pt: np.ndarray, poly: np.ndarray) -> float:
    """! @brief 点から凸多角形の各辺までの最短距離。
    @param pt   点 shape (2,) [m]
    @param poly 凸多角形の頂点 shape (h,2) [m]
    @return 距離 [m]（頂点が 3 未満なら 0）
    """
    q = np.asarray(pt, dtype=np.float64).reshape(2)
    v = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    h = v.shape[0]
    if h < 3:
        return 0.0
    d = np.inf
    for i in range(h):
        a, b = v[i], v[(i + 1) % h]
        e = b - a
        n = float(np.linalg.norm(e))
        if n < _EPS:
            continue
        u = e / n
        r = q - a
        d = min(d, abs(float(u[0] * r[1] - u[1] * r[0])))
    return float(d)


def foot_support_polygon(
    p_WF: np.ndarray,
    R_WF: np.ndarray,
    half_length: float,
    half_width: float,
    contact: np.ndarray,
) -> np.ndarray:
    """! @brief 足裏の矩形から支持多角形を作る（二足・面接触用）。
    @param p_WF        足先位置 {W}（足裏中心）shape (nf,3) [m]
    @param R_WF        足先姿勢 {W} shape (nf,3,3) [-]
    @param half_length 足裏の前後半長 [m]
    @param half_width  足裏の左右半長 [m]
    @param contact     接触フラグ shape (nf,) [bool]
    @return 凸包の頂点 shape (k,2) [m]（接触が無ければ空）
    @note **二足は足裏が面**で接するので、足先 2 点の凸包は線分になって面積を持たない。足裏の 4 隅を
          足先姿勢で回してから凸包を取る。単脚支持でも矩形が残るため、
          CoP の内包判定が意味を持つ。
    """
    p = np.asarray(p_WF, dtype=np.float64).reshape(-1, 3)
    R = np.asarray(R_WF, dtype=np.float64).reshape(-1, 3, 3)
    m = np.asarray(contact, dtype=bool).reshape(-1)
    corners: list[np.ndarray] = []
    local = np.array(
        [
            [+half_length, +half_width, 0.0],
            [+half_length, -half_width, 0.0],
            [-half_length, -half_width, 0.0],
            [-half_length, +half_width, 0.0],
        ],
        dtype=np.float64,
    )
    for i in range(p.shape[0]):
        if not m[i]:
            continue
        for v in local:
            corners.append((p[i] + R[i] @ v)[0:2])
    if not corners:
        return np.zeros((0, 2), dtype=np.float64)
    return convex_hull_2d(np.asarray(corners, dtype=np.float64))


def support_polygon_inradius(p_WF: np.ndarray, contact: np.ndarray) -> float:
    """! @brief 支持多角形の内接円半径（重心から辺までの最短距離で近似）。
    @param p_WF    足先位置 {W} shape (nf,3) [m]
    @param contact 接触フラグ shape (nf,) [bool]
    @return 半径 [m]（接触脚が 3 未満なら 0）
    @note 厳密な最大内接円ではなく「頂点重心を中心とする内接円」である。
          支持多角形が細長い場合に保守側（小さめ）の値を返す。
    """
    p = np.asarray(p_WF, dtype=np.float64).reshape(-1, 3)
    m = np.asarray(contact, dtype=bool).reshape(-1)
    if int(np.count_nonzero(m)) < 3:
        return 0.0
    hull = convex_hull_2d(p[m, :2])
    return distance_to_polygon_edges(polygon_centroid(hull), hull)


def cop_margin(
    p_WF: np.ndarray, f_W: np.ndarray, contact: np.ndarray, z_plane: float = 0.0
) -> tuple[float, float, bool]:
    """! @brief CoP の支持多角形内での余裕を評価する。
    @param p_WF    足先位置 {W} shape (nf,3) [m]
    @param f_W     足先接触力 {W} shape (nf,3) [N]
    @param contact 接触フラグ shape (nf,) [bool]
    @param z_plane CoP 評価平面の高さ [m]
    @return (d_center, r_in, inside)
            d_center 支持多角形重心から CoP までの距離 [m]、
            r_in 内接円半径 [m]、inside CoP が多角形内部か [-]
    @note 「CoP が内接円半径の 30% 以内」の判定は d_center < 0.3 * r_in で行う。
    """
    cop = center_of_pressure(p_WF, f_W, contact, z_plane)
    p = np.asarray(p_WF, dtype=np.float64).reshape(-1, 3)
    m = np.asarray(contact, dtype=bool).reshape(-1)
    if np.any(np.isnan(cop)) or int(np.count_nonzero(m)) < 3:
        return float("nan"), 0.0, False
    hull = convex_hull_2d(p[m, :2])
    ctr = polygon_centroid(hull)
    return (
        float(np.linalg.norm(cop[:2] - ctr)),
        distance_to_polygon_edges(ctr, hull),
        point_in_polygon(cop[:2], hull),
    )


def friction_margin(f_W: np.ndarray, mu: float) -> np.ndarray:
    """! @brief 各脚の摩擦錐余裕 mu*f_z - ||f_xy||。
    @param f_W 足先接触力 {W} shape (nf,3) [N]
    @param mu  摩擦係数 [-]
    @return margin shape (nf,) [N]（負なら滑りうる）
    """
    f = np.asarray(f_W, dtype=np.float64).reshape(-1, 3)
    return mu * f[:, 2] - np.linalg.norm(f[:, :2], axis=1)


def orientation_error(quat_WB: np.ndarray, quat_des: np.ndarray) -> np.ndarray:
    """! @brief 姿勢誤差を回転ベクトルで返す（log(R_des R^T)）。
    @param quat_WB 実姿勢 (w,x,y,z) shape (4,) [-]
    @param quat_des 目標姿勢 (w,x,y,z) shape (4,) [-]
    @return e shape (3,) [rad]（{W} 表現）
    """
    return log_so3(quat_to_rotmat(quat_des) @ quat_to_rotmat(quat_WB).T)


def rms(x: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    """! @brief 二乗平均平方根（NaN を無視）。
    @param x    配列 shape 任意 [-]
    @param axis 集約する軸（None で全体）[-]
    @return RMS [-]
    """
    a = np.asarray(x, dtype=np.float64)
    return np.sqrt(np.nanmean(a**2, axis=axis))


def cost_of_transport(
    tau: np.ndarray,
    dq: np.ndarray,
    mass: float,
    distance: float,
    dt: float,
    g: float = 9.81,
) -> float:
    """! @brief 輸送コスト CoT = E / (m g d)。
    @param tau      関節トルク時系列 shape (T,n) [N*m]
    @param dq       関節角速度時系列 shape (T,n) [rad/s]
    @param mass     全質量 [kg]
    @param distance 移動距離 [m]
    @param dt       時系列のサンプリング周期 [s]
    @param g        重力加速度 [m/s^2]
    @return CoT [-]
    @note 機械仕事の正部分のみを積算する（回生を仮定しない保守的な定義）。
    """
    p = np.sum(np.maximum(np.asarray(tau) * np.asarray(dq), 0.0), axis=1)
    e = float(np.nansum(p)) * float(dt)
    if distance <= _EPS:
        return float("nan")
    return e / (mass * g * distance)


# ==========================================================================
# 頑健性の指標
# ==========================================================================
@dataclass
class ImpulseTolerance:
    """! @brief 許容最大外乱力積の探索結果。

    @param impulse    許容最大力積の推定値 [N*s]（= 成功が確認された最大値）
    @param lower      成功が確認された最大値 [N*s]
    @param upper      失敗が確認された最小値 [N*s]（すべて成功なら inf）
    @param n_trial    試行回数 [-]
    @param monotonic  単調性の仮定（大きいほど失敗しやすい）が保たれたか [-]
    @param trials     (力積 [N*s], 成功 [-]) の全記録
    @note `monotonic=False` は「小さい外乱で転び、大きい外乱で転ばなかった」
          ことを意味する。二分探索の前提が崩れているので、**その値をそのまま
          耐性として報告してはいけない**。歩容位相のずれや数値的な分岐が疑われる。
    """

    impulse: float
    lower: float
    upper: float
    n_trial: int
    monotonic: bool
    trials: list = field(default_factory=list)


def bisect_max_impulse(
    trial_fn,
    lo: float = 1.0,
    hi: float = 40.0,
    n_iter: int = 8,
    tol_ratio: float = 0.05,
    verify_below: int = 1,
) -> ImpulseTolerance:
    """! @brief 許容最大外乱力積を対数スケールの二分探索で求める。
    @param trial_fn  力積 [N*s] を受け取り、成功なら True を返す関数 [-]
    @param lo        探索下限 [N*s]（ここは成功する想定）
    @param hi        探索上限 [N*s]（ここは失敗する想定）
    @param n_iter    二分回数 [-]
    @param tol_ratio 上下限の比がこれ以内になったら打ち切る [-]
    @param verify_below 探索後、境界より小さい側を追加で試す回数 [-]（0 で無効）
    @return ImpulseTolerance
    @note **対数スケールで刻む。** 耐性は機体質量と速度で 1 桁変わるので、
          線形に刻むと小さい側の分解能が足りない。
    @note 端点も必ず試す。`lo` で失敗するなら耐性は lo 未満、`hi` で成功するなら
          hi 以上であり、いずれも「二分探索の外側」であることを結果に残す。
    @note **二分探索そのものは単調性の破れを検出できない。** 失敗したら必ず下へ
          進むので、境界より下は一度も試されないからである。そこで探索後に
          境界より小さい側を `verify_below` 点だけ追試し、そこで失敗したら
          `monotonic=False` を立てる。歩容位相のずれや接触の分岐で
          「小さい外乱で転ぶ」ことは実際に起こりうる。
    """
    trials: list[tuple[float, bool]] = []

    def _run(x: float) -> bool:
        ok = bool(trial_fn(float(x)))
        trials.append((float(x), ok))
        return ok

    lo_f, hi_f = float(lo), float(hi)
    if not _run(lo_f):  # 下限でも失敗
        return ImpulseTolerance(0.0, 0.0, lo_f, len(trials), True, trials)
    if _run(hi_f):  # 上限でも成功
        return ImpulseTolerance(hi_f, hi_f, float("inf"), len(trials), True, trials)

    for _ in range(int(n_iter)):
        if hi_f / max(lo_f, 1e-9) - 1.0 <= tol_ratio:
            break
        mid = float(np.sqrt(lo_f * hi_f))  # 対数スケールの中点
        if _run(mid):
            lo_f = mid
        else:
            hi_f = mid

    # 単調性の追試: 境界より小さい側を対数等間隔で追加サンプルする
    for k in range(int(verify_below)):
        x = float(lo) * (lo_f / float(lo)) ** ((k + 1) / (int(verify_below) + 1))
        if x < lo_f * (1.0 - 1e-9):
            _run(x)

    ok_max = max((x for x, ok in trials if ok), default=0.0)
    ng_min = min((x for x, ok in trials if not ok), default=float("inf"))
    return ImpulseTolerance(
        impulse=lo_f,
        lower=lo_f,
        upper=hi_f,
        n_trial=len(trials),
        monotonic=ok_max <= ng_min,
        trials=trials,
    )


def traversal_rate(results) -> float:
    """! @brief 地形踏破率。
    @param results 試行結果の列（`traversed` 属性を持つもの）[-]
    @return 踏破率 [-] in [0,1]
    """
    rs = list(results)
    if not rs:
        return float("nan")
    return float(np.mean([1.0 if bool(getattr(r, "traversed", False)) else 0.0 for r in rs]))


def fall_rate(results) -> float:
    """! @brief 転倒率。
    @param results 試行結果の列（`fell` 属性を持つもの）[-]
    @return 転倒率 [-] in [0,1]
    """
    rs = list(results)
    if not rs:
        return float("nan")
    return float(np.mean([1.0 if bool(getattr(r, "fell", False)) else 0.0 for r in rs]))


def peak_torque(tau: np.ndarray) -> float:
    """! @brief 関節トルクのピーク（絶対値の最大）。
    @param tau トルク時系列 shape (T,n) [N*m]
    @return ピーク [N*m]
    """
    a = np.asarray(tau, dtype=np.float64)
    return float(np.nanmax(np.abs(a))) if a.size else float("nan")


def peak_contact_force(f: np.ndarray) -> float:
    """! @brief 接触力のピーク（法線成分の最大）。
    @param f 接触力時系列 shape (T,nf,3) [N]
    @return ピーク [N]
    """
    a = np.asarray(f, dtype=np.float64)
    return float(np.nanmax(a[..., 2])) if a.size else float("nan")
