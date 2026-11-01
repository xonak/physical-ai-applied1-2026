"""! @file robot_model.py
@brief 剛体動力学モデルのインタフェース（バックエンド非依存）。

背景:
Pinocchio は Windows 環境での導入コストが高い。そこで **動力学計算をインタフェースの背後に隠し、MuJoCo バックエンドを既定** とする。
Pinocchio が入った環境では backend: pinocchio に切り替えるだけでよい。

規約（重要）:
- 一般化座標  q  : shape (nq,) = [p_WB(3); quat_WB(4, wxyz); q_joint(n)]
- 一般化速度  v  : shape (nv,) = [v_W(3); w_B(3); dq_joint(n)]
  **並進はワールド表現、回転はボディ表現**（MuJoCo qvel と同順・同表現）。
  Pinocchio の既定は並進もボディ表現だが、既定バックエンドを
  MuJoCo にしたため本実装ではこの混合表現を唯一の正とする。
  RobotState.v_B（ボディ表現）との変換は body_velocity() / world_velocity() を使う。
- 制御層はこのインタフェースのみを参照し、mjModel / mjData に触れてはならない。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..common.config import RobotConfig
from ..common.frames import quat_to_rotmat

__all__ = ["RobotModelBase", "create_robot_model", "body_velocity", "world_velocity"]


def body_velocity(quat_WB: np.ndarray, v_W: np.ndarray) -> np.ndarray:
    """! @brief ワールド表現の並進速度をボディ表現に変換する。
    @param quat_WB ベース姿勢 (w,x,y,z) shape (4,) [-]
    @param v_W     並進速度 {W} shape (3,) [m/s]
    @return v_B 並進速度 {B} shape (3,) [m/s]
    """
    return quat_to_rotmat(quat_WB).T @ np.asarray(v_W, dtype=np.float64).reshape(3)


def world_velocity(quat_WB: np.ndarray, v_B: np.ndarray) -> np.ndarray:
    """! @brief ボディ表現の並進速度をワールド表現に変換する。
    @param quat_WB ベース姿勢 (w,x,y,z) shape (4,) [-]
    @param v_B     並進速度 {B} shape (3,) [m/s]
    @return v_W 並進速度 {W} shape (3,) [m/s]
    """
    return quat_to_rotmat(quat_WB) @ np.asarray(v_B, dtype=np.float64).reshape(3)


class RobotModelBase(ABC):
    """! @brief 剛体動力学モデルの共通インタフェース。"""

    def __init__(self, cfg: RobotConfig) -> None:
        """! @brief 設定を保持する。
        @param cfg RobotConfig [-]
        @return なし
        """
        self.cfg = cfg

    # ---- 次元 -------------------------------------------------------------
    @property
    @abstractmethod
    def nq(self) -> int:
        """! @brief 一般化座標次元 [-]"""

    @property
    @abstractmethod
    def nv(self) -> int:
        """! @brief 一般化速度次元 [-]"""

    @property
    def n_joint(self) -> int:
        """! @brief 駆動関節数 n [-]"""
        return self.cfg.n_joint

    @property
    def n_leg(self) -> int:
        """! @brief 脚数 nf [-]"""
        return self.cfg.n_leg

    @property
    @abstractmethod
    def total_mass(self) -> float:
        """! @brief 全質量 [kg]"""

    # ---- 座標の組み立て ---------------------------------------------------
    def make_q(self, p_WB: np.ndarray, quat_WB: np.ndarray, q_joint: np.ndarray) -> np.ndarray:
        """! @brief 一般化座標を組み立てる。
        @param p_WB    ベース位置 {W} shape (3,) [m]
        @param quat_WB ベース姿勢 (w,x,y,z) shape (4,) [-]
        @param q_joint 関節角 shape (n,) [rad]
        @return q shape (nq,) [m, -, rad]
        """
        return np.concatenate(
            [
                np.asarray(p_WB, dtype=np.float64).reshape(3),
                np.asarray(quat_WB, dtype=np.float64).reshape(4),
                np.asarray(q_joint, dtype=np.float64).reshape(self.n_joint),
            ]
        )

    def make_v(self, v_W: np.ndarray, w_B: np.ndarray, dq_joint: np.ndarray) -> np.ndarray:
        """! @brief 一般化速度を組み立てる。
        @param v_W      ベース並進速度 {W} shape (3,) [m/s]
        @param w_B      ベース角速度 {B} shape (3,) [rad/s]
        @param dq_joint 関節角速度 shape (n,) [rad/s]
        @return v shape (nv,) [m/s, rad/s, rad/s]
        """
        return np.concatenate(
            [
                np.asarray(v_W, dtype=np.float64).reshape(3),
                np.asarray(w_B, dtype=np.float64).reshape(3),
                np.asarray(dq_joint, dtype=np.float64).reshape(self.n_joint),
            ]
        )

    def leg_joint_indices(self, leg: int) -> np.ndarray:
        """! @brief 脚 leg に属する関節の（自前順の）番号を返す。
        @param leg 脚インデックス [-]
        @return idx shape (n_leg_joint,) [-]
        @note 自前順は cfg.joint_names の並び（脚順 x 脚内順）そのものであり、
              バックエンドに依らず設定だけで決まる。索引写像の二重定義を避けるため
              ここで一元化する。
        """
        counts = [len(j) for j in self.cfg.joint_names]
        if not 0 <= leg < len(counts):
            raise IndexError(f"脚インデックスが範囲外: {leg}")
        start = int(sum(counts[:leg]))
        return np.arange(start, start + counts[leg], dtype=int)

    @abstractmethod
    def integrate(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        """! @brief 一般化座標を速度ぶん進める（多様体上の加算）。
        @param q  一般化座標 shape (nq,) [m, -, rad]
        @param dq 速度 x 時間 shape (nv,) [m, rad, rad]
        @return q_next shape (nq,) [m, -, rad]
        @note ヤコビアンの数値微分検証に用いる。
        """

    # ---- 運動学 -----------------------------------------------------------
    @abstractmethod
    def fk(self, q: np.ndarray) -> np.ndarray:
        """! @brief 順運動学。全脚の足先位置を返す。
        @param q 一般化座標 shape (nq,) [m, -, rad]
        @return p_WF 足先位置 {W} shape (nf,3) [m]
        """

    @abstractmethod
    def foot_position_base(self, q: np.ndarray, leg: int) -> np.ndarray:
        """! @brief 足先位置をベース系 {B} で返す。
        @param q   一般化座標 shape (nq,) [m, -, rad]
        @param leg 脚インデックス [-]
        @return p_BF shape (3,) [m]
        """

    @abstractmethod
    def jacobian_foot(self, q: np.ndarray, leg: int) -> np.ndarray:
        """! @brief 足先の並進ヤコビアン（全一般化速度に対する {W} 速度）。
        @param q   一般化座標 shape (nq,) [m, -, rad]
        @param leg 脚インデックス [-]
        @return J shape (3, nv) [-]、v_WF = J @ v
        """

    @abstractmethod
    def jacobian_foot_leg(self, q: np.ndarray, leg: int) -> np.ndarray:
        """! @brief 脚のみのヤコビアン（ベース固定、{B} 表現）。
        @param q   一般化座標 shape (nq,) [m, -, rad]
        @param leg 脚インデックス [-]
        @return J_leg shape (3,3) [-]、v_BF = J_leg @ dq_leg
        @note 立位保持の tau = -J_leg^T f に用いる。
        """

    def jacobian_foot_rot(self, q: np.ndarray, leg: int) -> np.ndarray:
        """! @brief 足先の回転ヤコビアン（全一般化速度に対する {W} 角速度）。
        @param q   一般化座標 shape (nq,) [m, -, rad]
        @param leg 脚インデックス [-]
        @return J_rot shape (3, nv) [-]、w_WF = J_rot @ v
        @note 二足では足裏を床に平らに保つ必要があり、位置だけでは拘束が足りない
              （足首が傾くと片側が浮き、CoP が足裏内を動けなくなる）。
              既定では未対応として例外を投げ、
              対応するバックエンドだけが上書きする。
        """
        raise NotImplementedError(
            f"{type(self).__name__} は足先の回転ヤコビアンに未対応"
        )

    def foot_rotation(self, q: np.ndarray, leg: int) -> np.ndarray:
        """! @brief 足先の姿勢行列 {W}。
        @param q   一般化座標 shape (nq,) [m, -, rad]
        @param leg 脚インデックス [-]
        @return R_WF shape (3,3) [-]
        @note `jacobian_foot_rot` と対で使う。姿勢誤差は log(R_des R^T) で取る。
        """
        raise NotImplementedError(
            f"{type(self).__name__} は足先姿勢の取得に未対応"
        )

    def hand_position(self, q: np.ndarray, arm: int) -> np.ndarray:
        """! @brief 手先 site の位置 {W}（腕のある機体のみ）。
        @param q   一般化座標 shape (nq,) [m, -, rad]
        @param arm 腕インデックス [-]
        @return p_WH shape (3,) [m]
        """
        raise NotImplementedError(f"{type(self).__name__} は手先位置の取得に未対応")

    def hand_rotation(self, q: np.ndarray, arm: int) -> np.ndarray:
        """! @brief 手先 site の姿勢行列 {W}（腕のある機体のみ）。
        @param q   一般化座標 shape (nq,) [m, -, rad]
        @param arm 腕インデックス [-]
        @return R_WH shape (3,3) [-]
        """
        raise NotImplementedError(f"{type(self).__name__} は手先姿勢の取得に未対応")

    def jacobian_hand6(self, q: np.ndarray, arm: int) -> np.ndarray:
        """! @brief 手先 site の 6 次元ヤコビアン（{W} 表現、自前 dof 順。上 3 並進、下 3 回転）。
        @param q   一般化座標 shape (nq,) [m, -, rad]
        @param arm 腕インデックス [-]
        @return J shape (6,nv) [-]
        @note 手先タスク（インピーダンス）のための口。脚の列は 0 になる（手の位置は
              ベースと腕の関節だけで決まる）。
        """
        raise NotImplementedError(f"{type(self).__name__} は手先ヤコビアンに未対応")

    @abstractmethod
    def jacobian_foot_dot_v(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """! @brief 足先ヤコビアンの時間微分と速度の積 Jdot @ v。
        @param q 一般化座標 shape (nq,) [m, -, rad]
        @param v 一般化速度 shape (nv,) [m/s, rad/s]
        @return acc_bias shape (nf,3) [m/s^2]
        @note 全身制御の接触維持タスクで使う。
        """

    # ---- 動力学 -----------------------------------------------------------
    @abstractmethod
    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        """! @brief 慣性行列。
        @param q 一般化座標 shape (nq,) [m, -, rad]
        @return M shape (nv,nv) [kg, kg*m^2]
        """

    @abstractmethod
    def nle(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """! @brief 非線形項（コリオリ・遠心力 + 重力）。
        @param q 一般化座標 shape (nq,) [m, -, rad]
        @param v 一般化速度 shape (nv,) [m/s, rad/s]
        @return h shape (nv,) [N, N*m]、M ddq + h = S^T tau + J^T f
        """

    @abstractmethod
    def com(self, q: np.ndarray) -> np.ndarray:
        """! @brief 全身重心位置。
        @param q 一般化座標 shape (nq,) [m, -, rad]
        @return c_W shape (3,) [m]
        """

    @abstractmethod
    def Ag(self, q: np.ndarray) -> np.ndarray:
        """! @brief 重心運動量行列（centroidal momentum matrix）。
        @param q 一般化座標 shape (nq,) [m, -, rad]
        @return A_G shape (6,nv)、h_G = A_G @ v
                上 3 行 = 並進運動量 [kg*m/s]、下 3 行 = 重心まわり角運動量 [kg*m^2/s]
        """

    def com_velocity(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """! @brief 重心速度。
        @param q 一般化座標 shape (nq,) [m, -, rad]
        @param v 一般化速度 shape (nv,) [m/s, rad/s]
        @return dc_W shape (3,) [m/s]
        """
        return (self.Ag(q)[0:3] @ np.asarray(v, dtype=np.float64)) / self.total_mass


def create_robot_model(cfg: RobotConfig) -> RobotModelBase:
    """! @brief 設定に従って動力学バックエンドを生成する。
    @param cfg RobotConfig（backend: "mujoco" | "pinocchio"）[-]
    @return RobotModelBase の具象インスタンス
    @note 制御層はこの返り値の型だけに依存すること。
    """
    backend = cfg.backend.lower()
    if backend == "mujoco":
        from .backends.mujoco_backend import MujocoRobotModel

        return MujocoRobotModel(cfg)
    if backend == "pinocchio":
        from .backends.pinocchio_backend import PinocchioRobotModel

        return PinocchioRobotModel(cfg)
    raise ValueError(f"未知の動力学バックエンド: {cfg.backend}")
