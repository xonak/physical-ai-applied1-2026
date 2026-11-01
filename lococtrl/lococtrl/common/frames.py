"""! @file frames.py
@brief 座標系・姿勢表現の変換をここ 1 箇所に閉じ込めるモジュール。

規約:
- クォータニオンは (w, x, y, z) スカラー先頭。MuJoCo qpos[3:7] と同順。
- オイラー角は ZYX、配列順は [yaw, pitch, roll] [rad]。
  R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
- {H}（水平系）は {B} からロール・ピッチを除きヨーのみ残した系。
- SciPy Rotation は (x, y, z, w) 順であるため、その変換は本ファイルにのみ書く。
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "quat_normalize",
    "quat_conj",
    "quat_mul",
    "quat_to_rotmat",
    "rotmat_to_quat",
    "rotmat_to_euler_zyx",
    "euler_zyx_to_rotmat",
    "quat_to_euler_zyx",
    "euler_zyx_to_quat",
    "yaw_from_quat",
    "yaw_only_rotmat",
    "quat_rotate",
    "quat_rotate_inv",
    "skew",
    "log_so3",
    "exp_so3",
    "quat_wxyz_to_xyzw",
    "quat_xyzw_to_wxyz",
    "to_scipy_rotation",
    "from_scipy_rotation",
]

_EPS = 1.0e-12


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """! @brief クォータニオンを正規化し、スカラー部を非負に揃える。
    @param q クォータニオン (w,x,y,z) shape (4,) [-]
    @return 正規化済みクォータニオン shape (4,) [-]
    @note スカラー部の符号を固定することで、姿勢誤差の連続性を保つ。
    """
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(q))
    if n < _EPS:
        raise ValueError("quat_normalize: ノルムが 0 のクォータニオンが渡された")
    q = q / n
    return -q if q[0] < 0.0 else q


def quat_conj(q: np.ndarray) -> np.ndarray:
    """! @brief クォータニオンの共役（単位長なら逆回転）。
    @param q クォータニオン (w,x,y,z) shape (4,) [-]
    @return 共役クォータニオン shape (4,) [-]
    """
    q = np.asarray(q, dtype=np.float64).reshape(4)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """! @brief クォータニオン積 q1 (x) q2（回転の合成 R(q1) @ R(q2)）。
    @param q1 クォータニオン (w,x,y,z) shape (4,) [-]
    @param q2 クォータニオン (w,x,y,z) shape (4,) [-]
    @return 積 shape (4,) [-]
    """
    w1, x1, y1, z1 = np.asarray(q1, dtype=np.float64).reshape(4)
    w2, x2, y2, z2 = np.asarray(q2, dtype=np.float64).reshape(4)
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """! @brief クォータニオンから回転行列へ変換する。
    @param q クォータニオン (w,x,y,z) shape (4,) [-]
    @return R 回転行列 shape (3,3) [-]（列は {B} 基底の {W} 表現）
    """
    w, x, y, z = quat_normalize(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """! @brief 回転行列からクォータニオンへ変換する（Shepperd 法）。
    @param R 回転行列 shape (3,3) [-]
    @return q クォータニオン (w,x,y,z) shape (4,) [-]、スカラー部は非負
    @note 4 つの分岐から最大成分を選ぶことで、いずれの姿勢でも桁落ちを避ける。
    """
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        q = np.array(
            [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
        )
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q = np.array(
            [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
        )
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        q = np.array(
            [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
        )
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        q = np.array(
            [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
        )
    return quat_normalize(q)


def rotmat_to_euler_zyx(R: np.ndarray) -> np.ndarray:
    """! @brief 回転行列を ZYX オイラー角に分解する。
    @param R 回転行列 shape (3,3) [-]
    @return theta [yaw, pitch, roll] shape (3,) [rad]、pitch in [-pi/2, pi/2]
    @note pitch = +-pi/2 でジンバルロックになる。SRB-MPC の線形化以外に使わない。
    """
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    sp = -R[2, 0]
    sp = float(np.clip(sp, -1.0, 1.0))
    pitch = np.arcsin(sp)
    if abs(sp) > 1.0 - 1.0e-10:  # ジンバルロック近傍
        yaw = np.arctan2(-R[0, 1], R[1, 1])
        roll = 0.0
    else:
        yaw = np.arctan2(R[1, 0], R[0, 0])
        roll = np.arctan2(R[2, 1], R[2, 2])
    return np.array([yaw, pitch, roll], dtype=np.float64)


def euler_zyx_to_rotmat(theta: np.ndarray) -> np.ndarray:
    """! @brief ZYX オイラー角から回転行列を作る。R = Rz(yaw) Ry(pitch) Rx(roll)。
    @param theta [yaw, pitch, roll] shape (3,) [rad]
    @return R 回転行列 shape (3,3) [-]
    """
    yaw, pitch, roll = np.asarray(theta, dtype=np.float64).reshape(3)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def quat_to_euler_zyx(q: np.ndarray) -> np.ndarray:
    """! @brief クォータニオンを ZYX オイラー角に変換する。
    @param q クォータニオン (w,x,y,z) shape (4,) [-]
    @return theta [yaw, pitch, roll] shape (3,) [rad]
    """
    return rotmat_to_euler_zyx(quat_to_rotmat(q))


def euler_zyx_to_quat(theta: np.ndarray) -> np.ndarray:
    """! @brief ZYX オイラー角をクォータニオンに変換する。
    @param theta [yaw, pitch, roll] shape (3,) [rad]
    @return q クォータニオン (w,x,y,z) shape (4,) [-]
    """
    return rotmat_to_quat(euler_zyx_to_rotmat(theta))


def yaw_from_quat(q: np.ndarray) -> float:
    """! @brief 姿勢からヨー角のみを取り出す。
    @param q クォータニオン (w,x,y,z) shape (4,) [-]
    @return yaw [rad]
    @note x 軸を水平面に射影して求めるため、ピッチが大きい場合も定義される。
    """
    R = quat_to_rotmat(q)
    return float(np.arctan2(R[1, 0], R[0, 0]))


def yaw_only_rotmat(q: np.ndarray) -> np.ndarray:
    """! @brief {H}（水平系）の回転行列 Rz(yaw) を返す。
    @param q ベース姿勢クォータニオン (w,x,y,z) shape (4,) [-]
    @return R_WH shape (3,3) [-]
    @note 操縦指令はすべて {H} で解釈する。v_W = R_WH @ v_H。
    """
    yaw = yaw_from_quat(q)
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """! @brief ベクトルを q で回す（v_W = R(q) v_B）。
    @param q クォータニオン (w,x,y,z) shape (4,) [-]
    @param v ベクトル shape (3,) [任意単位]
    @return 回転後ベクトル shape (3,) [同単位]
    """
    return quat_to_rotmat(q) @ np.asarray(v, dtype=np.float64).reshape(3)


def quat_rotate_inv(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """! @brief ベクトルを q の逆回転で回す（v_B = R(q)^T v_W）。
    @param q クォータニオン (w,x,y,z) shape (4,) [-]
    @param v ベクトル shape (3,) [任意単位]
    @return 回転後ベクトル shape (3,) [同単位]
    """
    return quat_to_rotmat(q).T @ np.asarray(v, dtype=np.float64).reshape(3)


def skew(v: np.ndarray) -> np.ndarray:
    """! @brief 3 次元ベクトルの歪対称行列（外積行列）。
    @param v ベクトル shape (3,) [任意単位]
    @return S shape (3,3)、S @ a = v x a
    """
    x, y, z = np.asarray(v, dtype=np.float64).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def log_so3(R: np.ndarray) -> np.ndarray:
    """! @brief SO(3) の対数写像。姿勢誤差を回転ベクトルで表す。
    @param R 回転行列 shape (3,3) [-]
    @return omega 回転ベクトル shape (3,) [rad]、|omega| <= pi
    @note 姿勢 PD の誤差項 kp_R * log(R_des R^T) に用いる。
    """
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    cos_th = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    th = np.arccos(cos_th)
    if th < 1.0e-8:  # 微小回転: 1 次近似
        return 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if np.pi - th < 1.0e-6:  # pi 近傍: 対称成分から復元
        A = 0.5 * (R + np.eye(3))
        axis = np.sqrt(np.clip(np.diag(A), 0.0, None))
        k = int(np.argmax(axis))
        axis = A[:, k] / max(axis[k], _EPS)
        axis = axis / np.linalg.norm(axis)
        return th * axis
    return (th / (2.0 * np.sin(th))) * np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]
    )


def exp_so3(omega: np.ndarray) -> np.ndarray:
    """! @brief SO(3) の指数写像（Rodrigues）。
    @param omega 回転ベクトル shape (3,) [rad]
    @return R 回転行列 shape (3,3) [-]
    """
    omega = np.asarray(omega, dtype=np.float64).reshape(3)
    th = float(np.linalg.norm(omega))
    if th < 1.0e-12:
        return np.eye(3) + skew(omega)
    K = skew(omega / th)
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


# --------------------------------------------------------------------------
# SciPy 連携（xyzw 順序の変換はここだけに書く）
# --------------------------------------------------------------------------
def quat_wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    """! @brief (w,x,y,z) -> (x,y,z,w)。
    @param q クォータニオン shape (4,) [-]
    @return クォータニオン shape (4,) [-]
    """
    q = np.asarray(q, dtype=np.float64).reshape(4)
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)


def quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    """! @brief (x,y,z,w) -> (w,x,y,z)。
    @param q クォータニオン shape (4,) [-]
    @return クォータニオン shape (4,) [-]
    """
    q = np.asarray(q, dtype=np.float64).reshape(4)
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def to_scipy_rotation(q: np.ndarray):
    """! @brief (w,x,y,z) クォータニオンから scipy Rotation を作る。
    @param q クォータニオン shape (4,) [-]
    @return scipy.spatial.transform.Rotation
    @note SciPy は遅延 import（依存を common に持ち込まないため）。
    """
    from scipy.spatial.transform import Rotation

    return Rotation.from_quat(quat_wxyz_to_xyzw(q))


def from_scipy_rotation(rot) -> np.ndarray:
    """! @brief scipy Rotation から (w,x,y,z) クォータニオンを得る。
    @param rot scipy.spatial.transform.Rotation
    @return q クォータニオン shape (4,) [-]
    """
    return quat_normalize(quat_xyzw_to_wxyz(rot.as_quat()))
