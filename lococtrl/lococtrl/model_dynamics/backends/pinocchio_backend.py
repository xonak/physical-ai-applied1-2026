"""! @file pinocchio_backend.py
@brief Pinocchio バックエンド（未実装のプレースホルダ）。

Windows 環境では Pinocchio の導入コストが高いため、既定は MuJoCo バックエンドとした。
Pinocchio が使える環境（Linux / conda）では、ロボット設定の backend を "pinocchio" にし、本ファイルを実装する。

実装時の注意（MuJoCo バックエンドとの規約差）:
- Pinocchio の浮遊ベース速度は **ボディ表現**（並進も {B}）。本リポジトリの規約は
  [v_W; w_B; dq_j] であるため、**この境界で必ず変換する**こと。
    v_pin[0:3] = R_WB^T v_W,  v_pin[3:6] = w_B
  そのため M, h, J, Ag も相似変換 T = blkdiag(R_WB, I3, I_n) を挟む:
    J_ours = J_pin @ T^{-1},  M_ours = T^{-T} M_pin T^{-1}
- クォータニオン順序が (x,y,z,w) であるため、common.frames の変換関数を通すこと。
- 対応関数: forwardKinematics / computeJointJacobians / crba / nonLinearEffects /
  centerOfMass / computeCentroidalMap
"""

from __future__ import annotations

import numpy as np

from ...common.config import RobotConfig
from ..robot_model import RobotModelBase

__all__ = ["PinocchioRobotModel"]

_MSG = (
    "Pinocchio バックエンドは未実装。configs の backend を 'mujoco' にするか、"
    "lococtrl/model_dynamics/backends/pinocchio_backend.py を実装すること。"
)


class PinocchioRobotModel(RobotModelBase):
    """! @brief Pinocchio による動力学（未実装）。"""

    def __init__(self, cfg: RobotConfig) -> None:
        """! @brief 依存の有無を確認して失敗を早期に知らせる。
        @param cfg RobotConfig [-]
        @return なし
        """
        super().__init__(cfg)
        raise NotImplementedError(_MSG)

    @property
    def nq(self) -> int:
        """! @brief 一般化座標次元 [-]"""
        raise NotImplementedError(_MSG)

    @property
    def nv(self) -> int:
        """! @brief 一般化速度次元 [-]"""
        raise NotImplementedError(_MSG)

    @property
    def total_mass(self) -> float:
        """! @brief 全質量 [kg]"""
        raise NotImplementedError(_MSG)

    def integrate(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        """! @brief pin.integrate 相当 @param q (nq,) @param dq (nv,) @return (nq,)"""
        raise NotImplementedError(_MSG)

    def fk(self, q: np.ndarray) -> np.ndarray:
        """! @brief 足先位置 @param q (nq,) @return (nf,3) [m]"""
        raise NotImplementedError(_MSG)

    def foot_position_base(self, q: np.ndarray, leg: int) -> np.ndarray:
        """! @brief 足先位置 {B} @param q (nq,) @param leg [-] @return (3,) [m]"""
        raise NotImplementedError(_MSG)

    def jacobian_foot(self, q: np.ndarray, leg: int) -> np.ndarray:
        """! @brief 足先ヤコビアン @param q (nq,) @param leg [-] @return (3,nv)"""
        raise NotImplementedError(_MSG)

    def jacobian_foot_leg(self, q: np.ndarray, leg: int) -> np.ndarray:
        """! @brief 脚のみヤコビアン @param q (nq,) @param leg [-] @return (3,3)"""
        raise NotImplementedError(_MSG)

    def jacobian_foot_dot_v(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """! @brief Jdot v @param q (nq,) @param v (nv,) @return (nf,3) [m/s^2]"""
        raise NotImplementedError(_MSG)

    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        """! @brief 慣性行列 @param q (nq,) @return (nv,nv)"""
        raise NotImplementedError(_MSG)

    def nle(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """! @brief 非線形項 @param q (nq,) @param v (nv,) @return (nv,)"""
        raise NotImplementedError(_MSG)

    def com(self, q: np.ndarray) -> np.ndarray:
        """! @brief 重心 @param q (nq,) @return (3,) [m]"""
        raise NotImplementedError(_MSG)

    def Ag(self, q: np.ndarray) -> np.ndarray:
        """! @brief 重心運動量行列 @param q (nq,) @return (6,nv)"""
        raise NotImplementedError(_MSG)
