"""! @file biped_hl_policy.py
@brief HL（上位）方策の口: オンライン足配置計画への着地位置・時間の残差（Egle 2024 型）。

`planner.biped_online.OnlineBipedPlanner` は、DCM 誤差 e から着地位置を p = p_nom + k_step e e^{omega T_rem} で直す
（解析則）。Egle et al. 2024（DCM ベースの歩行に RL で足配置・時間の修正を載せる構成）は、
この「修正量」を方策の出力にする。本モジュールはその口を作る。

    観測 o  =  計画器の内部量（DCM 誤差、残り時間、位相、目標、立脚相対の位置）+ 状態（速度、姿勢）+ 前回行動
    行動 a  in [-1,1]^3  ->  dxy = scale_xy * a[0:2] [m]（パターン座標系、{W} に回して渡す）,  dT = scale_t * a[2] [s]
    計画器   planner.set_step_residual(dxy, dT)   （SS 中、目標を固定する前だけ効く）

下層（全身制御）と計画器の解析則はそのまま残すので、方策は「残差」だけを学べばよい。
k_step = 0 にすれば方策が足配置修正を全部担う構成（純粋 HL 方策）、k_step = 1 のままなら
解析則の上の残差になる（`biped_residual_policy` の残差トルクと同じ発想）。

方策の実装は 3 つ: ZeroHlPolicy（残差ゼロ = 解析則のみと同一）、OracleHlPolicy（解析則を外から与える。
口が機能することの確認と、学習の教師に使う）、MlpHlPolicy（NumPy 推論。学習器は含まない）。
学習用環境 `BipedHlEnv` は `biped_residual_policy.BipedResidualEnv` と同じ gym 風 API。

記号: omega LIPM 固有振動数 [rad/s]、{W} ワールド系、立脚 = 1 - swing。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..common.frames import quat_to_euler_zyx
from ..common.types import RobotState
from ..gait.biped_schedule import SegKind

__all__ = [
    "HlPolicySpec",
    "HL_OBS_SIZE",
    "HL_ACT_SIZE",
    "build_hl_observation",
    "ZeroHlPolicy",
    "OracleHlPolicy",
    "MlpHlPolicy",
    "make_hl_on_step",
    "BipedHlEnv",
]

## 観測の次元 [-]
HL_OBS_SIZE = 29
## 行動の次元 [-]（dx, dy, dT）
HL_ACT_SIZE = 3


@dataclass
class HlPolicySpec:
    """! @brief HL 方策の正規化と行動のスケール。
    @param scale_xy    着地位置残差の上限 [m]（行動 [-1,1] に掛ける）
    @param scale_t     単脚支持時間残差の上限 [s]
    @param decimation  方策を呼ぶ周期（制御周期の整数倍）[-]。25 で 50 ms（2 ms 制御のとき）
    @param e_scale     DCM 誤差の正規化 [m]
    @param v_scale     速度の正規化 [m/s]
    @param p_scale     位置の正規化 [m]
    """

    scale_xy: float = 0.10
    scale_t: float = 0.10
    decimation: int = 25
    e_scale: float = 0.05
    v_scale: float = 0.5
    p_scale: float = 0.3


def build_hl_observation(planner, state: RobotState, prev_action: np.ndarray,
                         spec: HlPolicySpec | None = None) -> np.ndarray:
    """! @brief 観測を組む（HL_OBS_SIZE = 29 次元、正規化済み）。
    @param planner     OnlineBipedPlanner [-]
    @param state       推定状態 [-]
    @param prev_action 前回の行動 shape (3,) [-]
    @param spec        正規化 [-]
    @return obs shape (29,) [-]
    @note 位置・速度・誤差はパターン座標系（進行方向 psi）に回して与える（旋回しても分布が変わらない）。
    """
    s = HlPolicySpec() if spec is None else spec
    omega = float(planner.omega)
    stance = 1 - int(planner.swing)
    p_st = np.asarray(planner.p_feet[stance, 0:2], dtype=np.float64)
    c = np.asarray(state.c_W[0:2], dtype=np.float64)
    dc = np.asarray(state.dc_W[0:2], dtype=np.float64)
    xi = c + dc / omega
    in_ss = planner.phase is SegKind.SS
    if in_ss:
        e = np.asarray(planner.last_e, dtype=np.float64)
        T_rem = float(planner.last_T_rem)
        u = float(np.clip((planner.t_now - planner.t_phase0) / max(planner.T_phase, 1.0e-9), 0.0, 1.0))
    else:
        e = xi - np.asarray(planner.last_xi_ref, dtype=np.float64) if planner.steps_done > 0 else np.zeros(2)
        T_rem = float(planner.gait.t_ss)
        u = 0.0
    e_eos = e * np.exp(omega * T_rem)
    Rt = planner.rot2().T  # {W} -> パターン系
    e, e_eos, dc = Rt @ e, Rt @ e_eos, Rt @ dc
    phase_1h = np.array([planner.phase is SegKind.STAND, planner.phase is SegKind.DS, in_ss], dtype=np.float64)
    rpy = quat_to_euler_zyx(state.quat_WB)[::-1]
    w = np.asarray(state.w_B, dtype=np.float64)
    tgt = Rt @ (np.asarray(planner.target_nominal[0:2], dtype=np.float64) - p_st) if in_ss else np.zeros(2)
    lift = Rt @ (np.asarray(planner.p_liftoff[0:2], dtype=np.float64) - p_st) if in_ss else np.zeros(2)
    obs = np.concatenate([
        e / s.e_scale,                                   # 2 DCM 誤差
        np.clip(e_eos / s.e_scale, -10.0, 10.0),         # 2 終端 DCM 誤差（発散後）
        [T_rem / max(planner.gait.t_ss, 1.0e-9), u],     # 2 残り時間、進捗
        phase_1h,                                        # 3 区間
        [planner._foot_sign(int(planner.swing))],        # 1 遊脚（+1 左 / -1 右）
        tgt / s.p_scale,                                 # 2 公称目標（立脚相対）
        lift / s.p_scale,                                # 2 離陸位置（立脚相対）
        (Rt @ (c - p_st)) / s.p_scale,                   # 2 重心（立脚相対）
        dc / s.v_scale,                                  # 2 重心速度
        np.asarray(planner.cmd[0:2], dtype=np.float64) / s.v_scale,  # 2 速度指令
        [float(planner.cmd[2]) / 1.0],                   # 1 旋回指令
        [rpy[0], rpy[1]],                                # 2 roll, pitch
        w[0:2] / 2.0,                                    # 2 角速度 roll, pitch
        [(state.c_W[2] - planner.z_c) / 0.05],           # 1 重心高さ誤差
        np.asarray(prev_action, dtype=np.float64).reshape(HL_ACT_SIZE),  # 3 前回行動
    ]).astype(np.float64)
    assert obs.shape == (HL_OBS_SIZE,)
    return obs


class ZeroHlPolicy:
    """! @brief 残差ゼロ（解析則のみと同一の挙動）。"""

    def act(self, obs: np.ndarray) -> np.ndarray:  # noqa: ARG002
        return np.zeros(HL_ACT_SIZE)


class OracleHlPolicy:
    """! @brief 解析則（DCM ステップ調整）を方策の形で外から与える。

    計画器の k_step を 0 にして本方策を繋ぐと、解析則のみ（k_step = 1）とほぼ同じ歩きになる
    （1 周期の遅れだけ違う）。口の確認と、学習の教師（オラクル教師付き損失）に使う。
    """

    def __init__(self, k_step: float = 1.0, spec: HlPolicySpec | None = None) -> None:
        self.k_step = float(k_step)
        self.spec = HlPolicySpec() if spec is None else spec

    def act(self, obs: np.ndarray) -> np.ndarray:
        e_eos = np.asarray(obs[2:4], dtype=np.float64) * self.spec.e_scale  # パターン系
        a = np.zeros(HL_ACT_SIZE)
        a[0:2] = np.clip(self.k_step * e_eos / self.spec.scale_xy, -1.0, 1.0)
        return a


class MlpHlPolicy:
    """! @brief NumPy だけで推論する tanh MLP。重みは params.npz（W0, b0, ...）、メタは meta.json（obs_mean, obs_std）。"""

    def __init__(self, model_dir: str | Path) -> None:
        d = Path(model_dir)
        z = np.load(d / "params.npz")
        self.layers = []
        k = 0
        while f"W{k}" in z:
            self.layers.append((np.asarray(z[f"W{k}"]), np.asarray(z[f"b{k}"])))
            k += 1
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8")) if (d / "meta.json").exists() else {}
        self.obs_mean = np.asarray(meta.get("obs_mean", 0.0), dtype=np.float64)
        self.obs_std = np.asarray(meta.get("obs_std", 1.0), dtype=np.float64)
        if self.layers[0][0].shape[0] != HL_OBS_SIZE or self.layers[-1][0].shape[1] != HL_ACT_SIZE:
            raise ValueError(f"HL 方策の次元が違う（観測 {HL_OBS_SIZE}、行動 {HL_ACT_SIZE}）")

    def act(self, obs: np.ndarray) -> np.ndarray:
        x = (np.asarray(obs, dtype=np.float64) - self.obs_mean) / (self.obs_std + 1.0e-8)
        for i, (W, b) in enumerate(self.layers):
            x = x @ W + b
            if i < len(self.layers) - 1:
                x = np.tanh(x)
        return np.tanh(x)


def _apply_action(planner, a: np.ndarray, spec: HlPolicySpec) -> None:
    """! @brief 行動（パターン系）を {W} の残差に直して計画器へ。"""
    a = np.clip(np.asarray(a, dtype=np.float64).reshape(HL_ACT_SIZE), -1.0, 1.0)
    dxy_W = planner.rot2() @ (spec.scale_xy * a[0:2])
    planner.set_step_residual(dxy_W, spec.scale_t * a[2])


def make_hl_on_step(policy, spec: HlPolicySpec | None = None, log: list | None = None):
    """! @brief `run_biped_torque(on_step=...)` に渡すフック。decimation ごとに観測 -> 行動 -> 計画器へ。
    @param policy act(obs) -> a を持つもの [-]
    @param spec   HlPolicySpec [-]
    @param log    None でなければ (k, obs, a) を追記する（学習データ・解析用）[-]
    @return on_step(planner, state, k)
    """
    s = HlPolicySpec() if spec is None else spec
    prev = np.zeros(HL_ACT_SIZE)

    def on_step(planner, state, k):
        nonlocal prev
        if k % max(1, int(s.decimation)) != 0:
            return
        if not planner.initialized:
            return  # 最初の update 前（p_feet 未設定）
        obs = build_hl_observation(planner, state, prev, s)
        a = np.clip(np.asarray(policy.act(obs), dtype=np.float64).reshape(HL_ACT_SIZE), -1.0, 1.0)
        _apply_action(planner, a, s)
        prev = a
        if log is not None:
            log.append((k, obs, a))

    return on_step


class BipedHlEnv:
    """! @brief HL 方策の学習用環境（gym 風、依存を増やさないため gym には継承しない）。

    reset() -> obs、step(action) -> (obs, reward, done, info)。1 step = decimation 制御周期。
    報酬 = 生存 + 速度追従 − DCM 誤差 − 姿勢 − 行動の大きさ、転倒で −50。
    `make_controller()` は (BipedTorqueController, 初期 RobotState) を返す。計画器は OnlineBipedPlanner。
    """

    def __init__(self, make_controller, env, dt_ctrl: float, spec: HlPolicySpec | None = None,
                 max_steps: int | None = None) -> None:
        self.make_controller = make_controller
        self.env = env
        self.dt_ctrl = float(dt_ctrl)
        self.spec = HlPolicySpec() if spec is None else spec
        self.max_steps = max_steps
        self.decim_sim = max(1, int(round(dt_ctrl / env.dt)))
        self.ctrl = None
        self.state = None
        self.k = 0
        self.prev = np.zeros(HL_ACT_SIZE)
        ## 毎制御周期に呼ぶ外乱フック (MujocoEnv, k) -> None（学習時の押し外乱注入。None で無し）
        self.disturbance = None

    @property
    def planner(self):
        return self.ctrl.planner

    def _obs(self) -> np.ndarray:
        return build_hl_observation(self.planner, self.state, self.prev, self.spec)

    def reset(self) -> np.ndarray:
        self.ctrl, self.state = self.make_controller()
        self.k = 0
        self.prev = np.zeros(HL_ACT_SIZE)
        # 1 周期回して計画器に足位置を入れる
        u = self.ctrl.compute(self.state, 0)
        self.env.apply(u)
        self.env.step(self.decim_sim)
        self.state = self.env.observe()
        self.k = 1
        return self._obs()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(HL_ACT_SIZE), -1.0, 1.0)
        _apply_action(self.planner, a, self.spec)
        self.prev = a
        reward = 0.0
        done = False
        info: dict = {}
        pl = self.planner
        for _ in range(max(1, int(self.spec.decimation))):
            if self.disturbance is not None:
                self.disturbance(self.env, self.k)
            u = self.ctrl.compute(self.state, self.k)
            ci = self.ctrl.last_info
            self.env.apply(u)
            self.env.step(self.decim_sim)
            self.state = self.env.observe()
            self.k += 1
            e_xi = ci.xi_hat - ci.ref.xi_ref
            dv = pl.rot2().T @ np.asarray(self.state.dc_W[0:2]) - np.asarray(pl.cmd[0:2])
            rpy = quat_to_euler_zyx(self.state.quat_WB)[::-1]
            reward += (
                1.0
                - 20.0 * float(e_xi @ e_xi)
                - 2.0 * float(dv @ dv)
                - 2.0 * float(rpy[0] ** 2 + rpy[1] ** 2)
                - 0.05 * float(a @ a)
            ) * self.dt_ctrl / 0.002
            fell = abs(rpy[0]) > np.radians(40.0) or abs(rpy[1]) > np.radians(40.0)
            n_max = pl.n_t if self.max_steps is None else self.max_steps
            if fell:
                reward -= 50.0
                done = True
                info["fell"] = True
                break
            if self.k >= n_max or pl.finished:
                done = True
                break
        info["k"] = self.k
        info["steps_done"] = int(pl.steps_done)
        return self._obs(), float(reward), done, info
