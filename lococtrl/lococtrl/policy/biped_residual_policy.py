"""! @file biped_residual_policy.py
@brief 学習 LL 方策の口: 残差トルク方策の観測・行動の定義と学習用環境。

設計は Yan & Egle 2026（"Efficiently Learning Robust Torque-based Locomotion Through
Reinforcement with Model-Based Supervision"）の構成に合わせる:

    tau = tau_base(モデルベース: DCM 計画 + 全身制御) + tau_res(学習方策)

- **ベース方策**は `BipedTorqueController`（ihmc / dlr のどちらでもよい）。
- **残差方策**は観測から残差トルク delta_tau in [-1, 1]^n * tau_scale を出す MLP。
- **オラクル方策**（真のダイナミクスを知るベース方策）による教師あり損失は、
  学習側（別スクリプト）で `BipedResidualEnv` の `info["tau_oracle"]` を使って付ける。
  本モジュールは環境と推論だけを持ち、学習器を含まない。

観測（`build_observation`）は実機で測れる量だけで構成する:
DCM 誤差、重心高さ誤差、重心速度、胴体ロール・ピッチ、角速度、関節角・角速度、
**関節トルクセンサ**、**足裏 F/T**、接触予定、荷重係数、区間進捗の sin/cos。
重心位置・速度はシミュレータの真値（Cheater）を使う。

記号: n 駆動関節数 [-]、nf 脚数 [-]。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..common.frames import quat_to_euler_zyx, quat_to_rotmat
from ..common.types import RobotState

__all__ = [
    "ResidualPolicySpec",
    "build_observation",
    "observation_size",
    "ZeroResidualPolicy",
    "MlpResidualPolicy",
    "BipedResidualEnv",
]


@dataclass
class ResidualPolicySpec:
    """! @brief 残差方策の正規化と上限。
    @param tau_scale   残差トルクの上限 [N*m]（行動 [-1,1] に掛ける）
    @param f_scale     足裏力の正規化 [N]（m g 程度）
    @param tau_meas_scale 関節トルク計測の正規化 [N*m]
    @param dq_scale    関節角速度の正規化 [rad/s]
    """

    tau_scale: float = 10.0
    f_scale: float = 330.0
    tau_meas_scale: float = 50.0
    dq_scale: float = 10.0


def observation_size(n_joint: int, n_leg: int) -> int:
    """! @brief 観測ベクトルの次元 [-]
    @param n_joint 駆動関節数 [-]
    @param n_leg   脚数 [-]
    @return 次元
    """
    return 2 + 1 + 3 + 2 + 3 + 2 * n_joint + n_joint + 6 * n_leg + 2 * n_leg + 2


def build_observation(state: RobotState, info, omega: float, spec: ResidualPolicySpec | None = None) -> np.ndarray:
    """! @brief 観測ベクトルを作る。
    @param state 推定状態 [-]
    @param info  `BipedTorqueController` の CycleInfo [-]
    @param omega LIPM 固有振動数 [rad/s]
    @param spec  正規化（None で既定）[-]
    @return obs shape (observation_size,) [-]
    """
    sp = ResidualPolicySpec() if spec is None else spec
    ref = info.ref
    rpy = quat_to_euler_zyx(state.quat_WB)[::-1]
    R = quat_to_rotmat(state.quat_WB)
    w_W = R @ np.asarray(state.w_B)
    nf = ref.contact.shape[0]
    wf = np.asarray(state.wrench_feet, dtype=np.float64).reshape(nf, 6).copy()
    wf[:, 0:3] /= sp.f_scale
    wf[:, 3:6] /= sp.f_scale * 0.1
    seg_u = float(ref.swing_ratio.max()) if ref.kind.value == "SS" else 0.0
    parts = [
        info.xi_hat - ref.xi_ref,                       # 2
        [state.c_W[2] - ref.c_ref[2]],                  # 1
        np.asarray(state.dc_W) - ref.dc_ref,            # 3
        rpy[0:2],                                       # 2
        w_W,                                            # 3
        np.asarray(state.q) - ref.q_posture,            # n
        np.asarray(state.dq) / sp.dq_scale,             # n
        np.asarray(state.tau_meas, dtype=np.float64).reshape(-1) / sp.tau_meas_scale,  # n
        wf.reshape(-1),                                 # 6 nf
        ref.contact.astype(np.float64),                 # nf
        np.asarray(ref.load, dtype=np.float64),         # nf
        [np.sin(2 * np.pi * seg_u), np.cos(2 * np.pi * seg_u)],  # 2
    ]
    obs = np.concatenate([np.atleast_1d(np.asarray(p, dtype=np.float64)) for p in parts])
    return obs


class ZeroResidualPolicy:
    """! @brief 残差ゼロ（ベース方策そのもの）。学習前の基準と試験に使う。"""

    def __init__(self, n_joint: int) -> None:
        """! @brief 構築する。
        @param n_joint 駆動関節数 [-]
        @return なし
        """
        self.n = int(n_joint)

    def act(self, obs: np.ndarray) -> np.ndarray:
        """! @brief ゼロを返す。
        @param obs 観測 [-]
        @return delta_tau shape (n,) [N*m]
        """
        del obs
        return np.zeros(self.n)


class MlpResidualPolicy:
    """! @brief NumPy だけで推論する tanh MLP（学習器に依存しない）。

    重みは npz（W0, b0, W1, b1, ...）、メタは meta.json（obs_mean, obs_std, tau_scale）。
    学習側（PyTorch / JAX）から `export_npz()` 相当で書き出す。
    """

    def __init__(self, model_dir: str | Path, n_joint: int) -> None:
        """! @brief 読み込む。
        @param model_dir params.npz と meta.json のあるディレクトリ [-]
        @param n_joint   駆動関節数 [-]
        @return なし
        """
        d = Path(model_dir)
        z = np.load(d / "params.npz")
        self.layers = []
        k = 0
        while f"W{k}" in z:
            self.layers.append((np.asarray(z[f"W{k}"]), np.asarray(z[f"b{k}"])))
            k += 1
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        self.obs_mean = np.asarray(meta.get("obs_mean", 0.0), dtype=np.float64)
        self.obs_std = np.asarray(meta.get("obs_std", 1.0), dtype=np.float64)
        self.tau_scale = float(meta.get("tau_scale", 10.0))
        self.n = int(n_joint)
        if self.layers[-1][0].shape[1] != self.n:
            raise ValueError("方策の出力次元が関節数と違う")

    def act(self, obs: np.ndarray) -> np.ndarray:
        """! @brief 推論する。
        @param obs 観測 [-]
        @return delta_tau shape (n,) [N*m]
        """
        x = (np.asarray(obs, dtype=np.float64) - self.obs_mean) / (self.obs_std + 1.0e-8)
        for i, (W, b) in enumerate(self.layers):
            x = x @ W + b
            if i < len(self.layers) - 1:
                x = np.tanh(x)
        return self.tau_scale * np.tanh(x)


class BipedResidualEnv:
    """! @brief 残差 RL の学習用環境（gym 風。依存を増やさないため gym には継承しない）。

    reset() -> obs、step(action) -> (obs, reward, done, info)。
    action は [-1, 1]^n、残差トルクは tau_scale を掛けたもの。
    報酬は DCM 誤差・重心高さ誤差・姿勢・残差トルクの二乗和の負 + 生存ボーナス。
    info["tau_base"] にベース方策のトルクを入れるので、オラクル教師付き損失
    （Yan 2026）は学習側で別環境（真のパラメータ）を回して差を取る。
    """

    def __init__(
        self,
        make_controller,
        env,
        dt_ctrl: float,
        spec: ResidualPolicySpec | None = None,
        policy_decimation: int = 1,
        max_steps: int | None = None,
    ) -> None:
        """! @brief 構築する。
        @param make_controller () -> (BipedTorqueController, 初期 RobotState) を返す関数 [-]
        @param env             MujocoEnv [-]
        @param dt_ctrl         制御周期 [s]
        @param spec            ResidualPolicySpec [-]
        @param policy_decimation 方策を評価する周期（制御周期の何倍か）[-]
        @param max_steps       エピソード長（None で計画の全体）[-]
        @return なし
        """
        self.make_controller = make_controller
        self.env = env
        self.dt_ctrl = float(dt_ctrl)
        self.spec = ResidualPolicySpec() if spec is None else spec
        self.decim_policy = int(policy_decimation)
        self.decim_sim = max(1, int(round(dt_ctrl / env.dt)))
        self.max_steps = max_steps
        self.ctrl = None
        self.state = None
        self.k = 0
        self._tau_res = None
        ## 毎制御周期に呼ぶ外乱フック (MujocoEnv, k) -> None（学習時の押し外乱注入。None で無し）
        self.disturbance = None

    def _obs(self) -> np.ndarray:
        return build_observation(self.state, self.ctrl.last_info, self.ctrl.omega, self.spec)

    def reset(self) -> np.ndarray:
        """! @brief エピソードを初期化する。
        @return obs
        """
        self.ctrl, self.state = self.make_controller()
        self.k = 0
        self._tau_res = np.zeros(self.ctrl.n)
        # 観測を作るために 1 周期ぶんの内部量が要る（トルクは適用しない）
        self.ctrl.residual = None
        self.ctrl.compute(self.state, 0)
        return self._obs()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        """! @brief 1 方策周期ぶん進める。
        @param action 行動 shape (n,) in [-1,1] [-]
        @return (obs, reward, done, info)
        """
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(self.ctrl.n), -1.0, 1.0)
        tau_res = self.spec.tau_scale * a
        reward = 0.0
        done = False
        info: dict = {}
        for _ in range(self.decim_policy):
            if self.disturbance is not None:
                self.disturbance(self.env, self.k)
            u = self.ctrl.compute(self.state, self.k)
            ci = self.ctrl.last_info
            u.tau = np.clip(u.tau + tau_res, -self.ctrl.rcfg.tau_max, self.ctrl.rcfg.tau_max)
            ci.tau_res = tau_res
            self.env.apply(u)
            self.env.step(self.decim_sim)
            self.state = self.env.observe()
            self.k += 1
            e_xi = ci.xi_hat - ci.ref.xi_ref
            e_z = self.state.c_W[2] - ci.ref.c_ref[2]
            rpy = quat_to_euler_zyx(self.state.quat_WB)[::-1]
            reward += (
                1.0
                - 20.0 * float(e_xi @ e_xi)
                - 10.0 * float(e_z * e_z)
                - 2.0 * float(rpy[0] ** 2 + rpy[1] ** 2)
                - 1.0e-3 * float(tau_res @ tau_res)
            ) * self.dt_ctrl / 0.002
            fell = abs(rpy[0]) > np.radians(40.0) or abs(rpy[1]) > np.radians(40.0)
            n_max = self.ctrl.planner.n_t if self.max_steps is None else self.max_steps
            if fell:
                reward -= 50.0
                done = True
                info["fell"] = True
                break
            if self.k >= n_max:
                done = True
                break
        info["tau_base"] = self.ctrl.last_info.tau_base.copy()
        info["k"] = self.k
        return self._obs(), float(reward), done, info
