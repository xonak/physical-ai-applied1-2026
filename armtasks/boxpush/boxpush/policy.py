"""! @file policy.py
@brief 箱押しの上位方策の口: 行動の仕様、観測、方策（Zero / Schedule / Reactive / Mlp）、計画器と手先タスクへの適用。
観測（方策への入力）と行動（方策の出力）のインタフェースを定義する。

方策は `BoxActionSpec.decimation` 制御周期ごと（50 で 10 Hz、5 で 100 Hz）に呼ばれ、次を出す（行動は [-1, 1]、仕様でスケール）。

| 名前      | 意味                                     | 受け口                                   | 保持 |
| --------- | ---------------------------------------- | ---------------------------------------- | ---- |
| step_dx   | 着地位置の残差（前）[m]                  | planner.set_step_residual（lococtrl HL） | 次の 1 歩に効く |
| step_dy   | 着地位置の残差（左）[m]                  | 同上                                     | 同上 |
| step_dt   | 単脚支持時間の残差 [s]                   | 同上                                     | 同上 |
| f_push    | 押付け力（両手合計）[N]                  | HandTask.f_ff（x 力制御）                 | 毎周期 |
| hand_dz   | 手先高さの構えからのずらし [m]           | HandTask.p_des z（重心に相対）            | 毎周期 |
| lean_x    | CMP 参照のオフセット x（前が正）[m]      | planner.set_lean                          | 毎周期 |
| cmp_dx    | lean_x の別名（向きを決めない名前）      | 同上                                      | 同上 |
| cmp_dy    | CMP 参照のオフセット y（左が正）[m]      | planner.set_lean                          | 毎周期 |
| com_dz    | 重心高さの公称からのずらし [m]           | planner.set_com_height（両脚支持で切替）  | 毎周期 |
| cop_dx/dy | 安定化層: CoP 指令の残差（パターン系）[m] | ctrl.set_external_residual(dcop)          | 判断周期 |
| mom_x/y   | 安定化層: 重心モーメントの残差 [N*m]     | ctrl.set_external_residual(dmom)          | 判断周期 |
| tau_leg0..11 | 全身制御: 脚関節トルクの残差 [N*m]    | ctrl.set_external_residual(tau_leg)       | 判断周期 |

説明資料の 3 方式との対応は次のとおり。

| 説明資料 | 方策の指定（eval_policy） | 学習の条件（PRESETS） | 行動 | 報酬の姿勢項 w_post |
| -------- | ------------------------- | --------------------- | ---- | ------------------- |
| Analytic | `analytic`                | A の行動の形を借用（学習なし） | 押付け力のスケジュール。前傾は脚側の解析則 | - |
| RL-M1    | `../../models/rl_m1`      | P                     | f_push, hand_dz, cmp_dx, cmp_dy, com_dz, step_dx, step_dy, step_dt（8） | 0 |
| RL-M2    | `../../models/rl_m2`      | legacy                | f_push, lean_x（2） | 1 |

学習の条件は `BoxActionSpec.preset(name)` で選ぶ。P と legacy 以外は、安定化層や全身制御への残差を
加えた追加の条件であり、説明資料では評価していない。

| 名前   | 行動 | 判断周期 | 脚 |
| ------ | ---- | -------- | -- |
| A      | f_push, hand_dz（腕の上位方策だけ） | 10 Hz | 解析則（前傾は押付け力の指令から） |
| legacy | f_push, lean_x（押付け力と前傾） | 10 Hz | 解析則 + 前傾の指令 |
| B      | A + cop_dx, cop_dy（安定化層の残差） | 100 Hz | 解析則 + CoP 残差 |
| B_mom  | B + mom_x, mom_y | 100 Hz | 同上 + モーメント残差 |
| C      | B + tau_leg0..11（全身制御の残差トルク ±10 N·m） | 500 Hz | 同上 + 脚トルク残差 |
| P      | f_push, hand_dz + 計画層の残差 cmp_dx, cmp_dy, com_dz, step_dx, step_dy, step_dt | 10 Hz | 解析則 + 計画の残差（姿勢の指示なし） |
| P_S    | P + cop_dx, cop_dy | 100 Hz | 同上 + CoP 残差 |
| P_S_T  | P_S + tau_leg0..11 | 500 Hz | 同上 + 脚トルク残差 |
| S      | cop_dx, cop_dy だけ | 100 Hz | 上位の指令は基底方策（--base）か固定値 |
| S_T    | cop_dx, cop_dy + tau_leg0..11 | 500 Hz | 同上 |

基底方策（`BoxPushEnv(base_policy=...)`、学習は `--base ../../models/rl_m2`）: 学習済みの上位方策（legacy など）を
凍結し、その出力（f_push, lean_x）を上位の指令にする。RL の行動に同じ口（f_push、cmp_dx = lean_x）があれば
残差として足し（`merge_base`）、なければ基底の値をそのまま使う。「上位の指令を残差学習するか（P_S_T --base）、
しないか（S_T --base）」の比較に使う。

残差を下層に足す条件は、下層の周期に合わせて推論する（安定化層の CoP 残差 100 Hz、全身制御のトルク残差は制御周期と
同じ 500 Hz）。残差を長く保持すると、2 ms ごとに状態を見て出す下層の出力に対して古い状態で決めた値を押し続け、
補正のつもりが遅れとして効く。一方、腕と計画層の出力（押付け力、手先高さ、CMP オフセット、重心高さ、足配置）は
推論が速くても `slow_period`（100 ms）ごとにしか更新しない（`BoxActionSpec.hold_slow`。速く揺らす意味がなく、
学習も難しくなる）。

P 系は姿勢を指示しない。前傾（CMP オフセット）・しゃがみ・足配置は方策に自由度として渡すだけで、
向きも量も初期値 0 から学ぶ（解析則の前傾は使わない。報酬の姿勢項 w_post も 0、`PRESET_REWARD`）。

lean_x を行動に含めないとき（A / B / C と Analytic）、前傾は脚側の解析則で決める（`BoxPushApplier.lean_mode = "analytic"`）。
押付け力の指令 f に釣り合う CMP のずれ f / (m omega^2) を前傾にする（係数 1.0）。
押す前から lean_pre（5 cm）だけ先に前へ出し、押付け力は「今の前傾で支えられる力 + f_pre」で頭を抑える。
手首 F/T の実測から前傾を決める版（`ReactiveBoxPolicy`）は 30 kg で転倒したので、Analytic には使わない。

使う行動は `BoxActionSpec.actions` で選ぶ（例: RL-M2 は ["f_push", "lean_x"]）。
使わない行動は `BoxActionSpec.fixed` の値（Schedule 方策と同じ手書き値）で埋める。

観測（`build_box_observation`）は lococtrl の HL 観測 29 次元に箱押しの 11 次元を足した 40 次元。
安定化層の残差を使うとき 6 + 残差の前回行動、全身制御の残差を使うとき 24 + 12 を足す（`box_obs_size(spec)`）。
質量・摩擦は観測に入れない（手先力と箱の動きから推定させる）。

方策はどれも `act(obs) -> a` を持つ:
- `ZeroBoxPolicy`    : 全行動 0（= fixed の値だけ。口の確認）
- `ScheduleBoxPolicy`: 手書きスケジュール（t_start からランプ）を行動の形で出す。学習の比較基準
- `ReactiveBoxPolicy`: 押付け力はスケジュール、前傾は手首 F/T の実測から決める手書き版
- `MlpBoxPolicy`     : NumPy だけで推論する tanh MLP（params.npz + meta.json。lococtrl の MlpHlPolicy と同じ形式）
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from lococtrl.common.frames import quat_to_euler_zyx
from lococtrl.control.hand_task import HandTask
from lococtrl.policy.biped_hl_policy import HL_ACT_SIZE, HL_OBS_SIZE, HlPolicySpec, build_hl_observation

__all__ = [
    "ACTION_NAMES", "BOX_OBS_SIZE", "PRESETS", "BoxActionSpec", "box_obs_size", "build_box_observation",
    "ZeroBoxPolicy", "ScheduleBoxPolicy", "ReactiveBoxPolicy", "MlpBoxPolicy", "BoxPushApplier",
]

## 脚関節の数（G1、脚 → 腕の並びの先頭）[-]
N_LEG_JOINT = 12
## 全身制御の残差トルクの行動名
TAU_LEG_NAMES = tuple(f"tau_leg{i}" for i in range(N_LEG_JOINT))
## 安定化層の残差の行動名
STAB_NAMES = ("cop_dx", "cop_dy", "mom_x", "mom_y")
## 計画層の残差の行動名。cmp_dx は lean_x の別名（実値は lean_x に書く）
PLAN_NAMES = ("cmp_dx", "cmp_dy", "com_dz", "step_dx", "step_dy", "step_dt")
## 行動の全名称（この順で `BoxActionSpec.scales` / `fixed` を引く）
ACTION_NAMES = ("step_dx", "step_dy", "step_dt", "f_push", "hand_dz", "lean_x", "cmp_dx", "cmp_dy", "com_dz") + STAB_NAMES + TAU_LEG_NAMES
## 箱押しで足す観測の次元 [-]
BOX_EXTRA_OBS = 11
## 観測の次元 [-]（HL 29 + 箱押し 11）。残差層を使うときは `box_obs_size` で増える
BOX_OBS_SIZE = HL_OBS_SIZE + BOX_EXTRA_OBS
## 安定化層の残差を使うときに足す観測（前回行動を除く）[-]
STAB_EXTRA_OBS = 6
## 全身制御の残差を使うときに足す観測（前回行動を除く）[-]
WBC_EXTRA_OBS = 2 * N_LEG_JOINT

## 学習の条件（RL-M1 は P、RL-M2 は legacy）。名前 -> (行動, 判断周期 decimation)
PRESETS: dict = {
    "A": (("f_push", "hand_dz"), 50),
    "B": (("f_push", "hand_dz", "cop_dx", "cop_dy"), 5),
    "B_mom": (("f_push", "hand_dz", "cop_dx", "cop_dy", "mom_x", "mom_y"), 5),
    "C": (("f_push", "hand_dz", "cop_dx", "cop_dy") + TAU_LEG_NAMES, 1),
    "legacy": (("f_push", "lean_x"), 50),
    "P": (("f_push", "hand_dz") + PLAN_NAMES, 50),
    "P_S": (("f_push", "hand_dz") + PLAN_NAMES + ("cop_dx", "cop_dy"), 5),
    "P_S_T": (("f_push", "hand_dz") + PLAN_NAMES + ("cop_dx", "cop_dy") + TAU_LEG_NAMES, 1),
    "S": (("cop_dx", "cop_dy"), 5),
    "S_T": (("cop_dx", "cop_dy") + TAU_LEG_NAMES, 1),
}
## 基底方策の出力と RL の行動で同じ口になる名前（基底 -> RL）
BASE_ALIAS = {"lean_x": "cmp_dx", "cmp_dx": "lean_x"}
## 条件ごとの報酬重みの上書き（P 系は姿勢を罰しない）
PRESET_REWARD: dict = {"P": {"w_post": 0.0}, "P_S": {"w_post": 0.0}, "P_S_T": {"w_post": 0.0}}


@dataclass
class BoxActionSpec:
    """! @brief 行動の仕様（どの行動を使うか、スケール、固定値）。
    @param actions    方策が出す行動の名前（ACTION_NAMES の部分列）[-]
    @param scales     行動 [-1,1] に掛ける量（名前 -> 値）。step_* は残差の上限、f_push は N、長さは m
    @param fixed      方策が出さない行動に使う値（名前 -> 実値）。Schedule 方策の目標値でもある
    @param decimation 方策を呼ぶ周期（制御周期の整数倍）[-]。50 で 10 Hz、5 で 100 Hz、1 で 500 Hz（2 ms 制御）
    @param slow_period 腕・計画層の行動を更新する周期（制御周期の整数倍）[-]。50 で 100 ms。decimation より短ければ無効
    @param f_push_max 押付け力の上限 [N]（f_push の実値 = f_push_max * (a + 1) / 2 で 0〜max）。40 kg の箱（摩擦 0.2）の
                      静止摩擦は 78 N、押し切るには ~90 N 要るので 120 N
    @param hl         lococtrl HL 観測の正規化 [-]
    @note f_push だけは [-1, 1] を [0, f_push_max] に写す（押す向きは決まっているので負は要らない）。
    """

    actions: tuple[str, ...] = ("f_push", "lean_x")
    scales: dict = field(default_factory=lambda: {
        "step_dx": 0.10, "step_dy": 0.10, "step_dt": 0.10,
        "f_push": 1.0, "hand_dz": 0.15, "lean_x": 0.20, "cmp_dx": 0.20, "cmp_dy": 0.10, "com_dz": 0.15,
        "cop_dx": 0.03, "cop_dy": 0.03, "mom_x": 10.0, "mom_y": 10.0,
        **{n: 10.0 for n in TAU_LEG_NAMES},
    })
    fixed: dict = field(default_factory=lambda: {
        "step_dx": 0.0, "step_dy": 0.0, "step_dt": 0.0,
        "f_push": 40.0, "hand_dz": 0.0, "lean_x": 0.04, "cmp_dx": 0.0, "cmp_dy": 0.0, "com_dz": 0.0,
        **{n: 0.0 for n in STAB_NAMES + TAU_LEG_NAMES},
    })
    decimation: int = 50
    slow_period: int = 50
    f_push_max: float = 120.0
    hl: HlPolicySpec = field(default_factory=lambda: HlPolicySpec(decimation=50))
    preset_name: str = ""

    def __post_init__(self) -> None:
        bad = [a for a in self.actions if a not in ACTION_NAMES]
        if bad:
            raise ValueError(f"未知の行動 {bad}（使える名前: {ACTION_NAMES}）")
        if "lean_x" in self.actions and "cmp_dx" in self.actions:
            raise ValueError("lean_x と cmp_dx は同じ口（どちらか一方）")
        self.actions = tuple(self.actions)

    @classmethod
    def preset(cls, name: str, **kw) -> "BoxActionSpec":
        """! @brief 学習の条件（PRESETS のキー。RL-M1 は P、RL-M2 は legacy）から作る。
        @param name 条件名 [-]
        @param kw   ほかの引数（f_push_max など）[-]
        @return BoxActionSpec
        """
        if name not in PRESETS:
            raise ValueError(f"未知の条件 {name}（{list(PRESETS)}）")
        acts, dec = PRESETS[name]
        kw.setdefault("decimation", dec)
        return cls(actions=acts, preset_name=name, **kw)

    @property
    def uses_lean_action(self) -> bool:
        """! @brief 前傾を方策が出すか（False なら解析則で決める）。"""
        return "lean_x" in self.actions or "cmp_dx" in self.actions

    @property
    def stab_actions(self) -> tuple[str, ...]:
        """! @brief 使う安定化層の残差の行動名。"""
        return tuple(a for a in self.actions if a in STAB_NAMES)

    @property
    def uses_tau_leg(self) -> bool:
        """! @brief 全身制御の残差トルクを使うか。"""
        return any(a in TAU_LEG_NAMES for a in self.actions)

    @property
    def residual_actions(self) -> tuple[str, ...]:
        """! @brief 下層への残差（安定化層 + 全身制御）の行動名（方策の行動の並び順）。"""
        return tuple(a for a in self.actions if a in STAB_NAMES or a in TAU_LEG_NAMES)

    def merge_base(self, vals: dict, base_vals: dict, base_actions: tuple[str, ...]) -> dict:
        """! @brief 基底方策の出力を上位の指令に合成する。RL に同じ口があれば残差として足し、なければ基底の値を使う。
        @param vals         RL の行動の実値（to_values）[-]
        @param base_vals    基底方策の行動の実値（基底の spec.to_values）[-]
        @param base_actions 基底方策が出す行動名 [-]
        @return dict（f_push は [0, f_push_max]、ほかは scale の範囲に clip）
        """
        out = dict(vals)
        for name in base_actions:
            b = float(base_vals[name])
            rl = name if name in self.actions else BASE_ALIAS.get(name)
            if rl is not None and rl in self.actions:
                if name == "f_push":
                    out[name] = float(np.clip(b + vals[name] - 0.5 * self.f_push_max, 0.0, self.f_push_max))  # RL の 0 点 = 基底
                else:
                    lim = self.scales[rl]
                    out[name] = float(np.clip(b + vals[name], -lim, lim))
            else:
                out[name] = b
            if name in BASE_ALIAS and BASE_ALIAS[name] in out:
                out[BASE_ALIAS[name]] = out[name]
        if "cmp_dx" in out:
            out["lean_x"] = out["cmp_dx"] if "cmp_dx" in self.actions else out["lean_x"]
        return out

    @property
    def slow_index(self) -> np.ndarray:
        """! @brief 行動ベクトルのうち slow_period ごとにしか更新しない位置（腕・計画層）[-]"""
        return np.array([i for i, a in enumerate(self.actions) if a not in STAB_NAMES and a not in TAU_LEG_NAMES], dtype=int)

    def hold_slow(self, a: np.ndarray, prev: np.ndarray, k: int) -> np.ndarray:
        """! @brief 腕・計画層の行動を slow_period ごとにだけ更新した行動を返す（残差はそのまま）。
        @param a    方策の出した行動 shape (n_act,) [-]
        @param prev 前回の行動（保持する値）[-]
        @param k    制御周期番号 [-]
        @return shape (n_act,) [-]
        """
        # 判断の時刻は decimation の倍数とは限らない（reset で 1 周期進む）ので、slow_period の区切りを跨いだ判断で更新する
        if self.decimation >= self.slow_period or prev.size != a.size or (k % self.slow_period) < self.decimation:
            return a
        out = np.array(a, dtype=np.float64)
        idx = self.slow_index
        out[idx] = prev[idx]
        return out

    @property
    def residual_index(self) -> np.ndarray:
        """! @brief 行動ベクトルのうち残差の位置 [-]"""
        return np.array([i for i, a in enumerate(self.actions) if a in STAB_NAMES or a in TAU_LEG_NAMES], dtype=int)

    @property
    def n_act(self) -> int:
        """! @brief 方策の行動次元 [-]"""
        return len(self.actions)

    def to_values(self, a: np.ndarray) -> dict:
        """! @brief 行動ベクトル [-1,1] を全項目の実値（名前 -> 値）にする。使わない項目は fixed。
        @param a shape (n_act,) [-]
        @return dict
        """
        a = np.clip(np.asarray(a, dtype=np.float64).reshape(self.n_act), -1.0, 1.0)
        vals = dict(self.fixed)
        if "cmp_dx" in self.actions:
            vals["lean_x"] = 0.0
        for name, ai in zip(self.actions, a):
            if name == "f_push":
                vals[name] = 0.5 * self.f_push_max * (float(ai) + 1.0)
            else:
                vals[name] = float(self.scales[name] * ai)
        if "cmp_dx" in self.actions:
            vals["lean_x"] = vals["cmp_dx"]       # 別名: 計画器には lean_x として渡す
        return vals

    def from_values(self, vals: dict) -> np.ndarray:
        """! @brief 実値（名前 -> 値）を行動ベクトル [-1,1] に戻す（Schedule 方策・教師信号用）。
        @param vals dict [-]
        @return shape (n_act,) [-]
        """
        out = np.zeros(self.n_act)
        for i, name in enumerate(self.actions):
            if name == "cmp_dx" and "lean_x" in vals:
                v = float(vals["lean_x"])            # 別名: 手書き（lean_x）を cmp_dx の行動に戻す
            else:
                v = float(vals.get(name, self.fixed[name]))
            out[i] = 2.0 * v / self.f_push_max - 1.0 if name == "f_push" else v / self.scales[name]
        return np.clip(out, -1.0, 1.0)


def box_obs_size(spec: BoxActionSpec) -> int:
    """! @brief 行動仕様に合う観測の次元 [-]（40 + 安定化層 6 + 前回残差 + 全身制御 24）。"""
    n = BOX_OBS_SIZE
    if spec.residual_actions:
        n += STAB_EXTRA_OBS + len(spec.residual_actions)
    if spec.uses_tau_leg:
        n += WBC_EXTRA_OBS
    return n


def build_residual_observation(planner, state, info, prev_action: np.ndarray, spec: BoxActionSpec,
                               q_nominal: np.ndarray | None, m_total: float) -> np.ndarray:
    """! @brief 残差層の観測（安定化層: 6 + 前回残差、全身制御: + 24）。
    @param planner     OnlineBipedPlanner [-]
    @param state       RobotState [-]
    @param info        直前周期の CycleInfo（None なら 0）[-]
    @param prev_action 前回の行動 shape (n_act,) [-]
    @param spec        BoxActionSpec [-]
    @param q_nominal   公称の関節角（脚の先頭 12）[rad]
    @param m_total     ロボットの質量 [kg]
    @return shape (box_obs_size(spec) - 40,) [-]
    @note 安定化層の 6: CoP 指令の支持中心からのずれ（パターン系、/0.1 m）、足裏鉛直力 L/R（/m g）、接触 L/R。
          全身制御の 24: 脚関節角の公称からの差（/0.3 rad）、脚関節速度（/5 rad/s）。
    """
    parts: list = []
    if spec.residual_actions:
        Rt = planner.rot2().T
        pf = np.asarray(state.p_WF, dtype=np.float64)
        con = np.asarray(state.contact, dtype=bool)
        center = pf[con, 0:2].mean(axis=0) if np.any(con) else pf[:, 0:2].mean(axis=0)
        cop = np.asarray(info.cop_cmd, dtype=np.float64) if info is not None else center
        wf = np.asarray(state.wrench_feet, dtype=np.float64).reshape(-1, 6)
        parts += [Rt @ (cop - center) / 0.1, wf[0:2, 2] / (m_total * 9.81), con[0:2].astype(np.float64)]
        idx = spec.residual_index
        pa = np.asarray(prev_action, dtype=np.float64)
        parts.append(pa[idx] if pa.size == spec.n_act else np.zeros(idx.size))
    if spec.uses_tau_leg:
        q = np.asarray(state.q, dtype=np.float64)[0:N_LEG_JOINT]
        qn = np.zeros(N_LEG_JOINT) if q_nominal is None else np.asarray(q_nominal, dtype=np.float64)[0:N_LEG_JOINT]
        dq = np.asarray(state.dq, dtype=np.float64)[0:N_LEG_JOINT]
        parts += [(q - qn) / 0.3, dq / 5.0]
    return np.concatenate([np.atleast_1d(np.asarray(x, dtype=np.float64)) for x in parts]) if parts else np.zeros(0)


def build_box_observation(planner, state, hand_task: HandTask, prev_action: np.ndarray, spec: BoxActionSpec,
                          box_half_x: float, hand_z_nom_rel: float, tau_sat: float, p_hands: np.ndarray) -> np.ndarray:
    """! @brief 観測を組む（BOX_OBS_SIZE = 40、正規化済み）。
    @param planner      OnlineBipedPlanner [-]
    @param state        RobotState（box_pose, wrench_hands）[-]
    @param hand_task    HandTask（直前の手先力指令）[-]
    @param prev_action  前回の行動 shape (n_act,) [-]（HL の 3 次元ぶんだけ lococtrl の観測に入れる）
    @param spec         BoxActionSpec [-]
    @param box_half_x   箱の x 半長 [m]
    @param hand_z_nom_rel 構えの手先高さ（重心に相対）[m]
    @param tau_sat      直前周期のトルク飽和率 [-]
    @param p_hands      手先位置 {W} shape (2,3) [m]
    @return obs shape (40,) [-]
    @note 箱押しの 11 次元（パターン座標系）: 箱の面の相対位置 x（重心から）、箱の相対 y、箱の速度 x、
          手の押付け力（左右、F/T 実測）、手先高さ（重心相対、構えからの差）、重心高さ（公称からの差）、
          CMP オフセット x（今の値）、トルク飽和率、押付け力の指令（f_ff 合計）、CMP オフセット y。
    """
    vals_prev = spec.to_values(prev_action) if prev_action.size == spec.n_act else dict(spec.fixed)
    hl_prev = np.array([vals_prev["step_dx"] / spec.scales["step_dx"], vals_prev["step_dy"] / spec.scales["step_dy"],
                        vals_prev["step_dt"] / spec.scales["step_dt"]])
    base = build_hl_observation(planner, state, hl_prev, spec.hl)
    Rt = planner.rot2().T
    c = np.asarray(state.c_W, dtype=np.float64)
    bp = np.asarray(state.box_pose, dtype=np.float64)
    if bp.size >= 10:
        d_face = Rt @ (bp[0:2] - c[0:2])
        d_face[0] -= box_half_x
        v_box = float((Rt @ bp[7:9])[0])
    else:
        d_face = np.zeros(2)
        v_box = 0.0
    wh = np.asarray(state.wrench_hands, dtype=np.float64)
    f_hands = -wh[:, 0] if wh.shape == (2, 6) else np.zeros(2)
    hand_z_rel = float(np.mean(p_hands[:, 2])) - c[2] - hand_z_nom_rel if p_hands.size else 0.0
    extra = np.array([
        d_face[0] / 0.3, d_face[1] / 0.3, v_box / 0.3,
        f_hands[0] / 40.0, f_hands[1] / 40.0,
        hand_z_rel / 0.15,
        (planner.z_c - planner.z_c_nominal) / 0.15,
        float(planner.lean_xy[0]) / 0.10,
        float(tau_sat),
        float(np.sum(hand_task.f_ff[:, 0])) / 40.0,
        float(planner.lean_xy[1]) / 0.10,
    ], dtype=np.float64)
    obs = np.concatenate([base, extra])
    assert obs.shape == (BOX_OBS_SIZE,)
    return obs


class ZeroBoxPolicy:
    """! @brief 全行動 0（fixed の値だけが入る。口の確認用）。"""

    needs_obs = False

    def __init__(self, spec: BoxActionSpec) -> None:
        self.spec = spec

    def act(self, obs: np.ndarray) -> np.ndarray:  # noqa: ARG002
        return np.zeros(self.spec.n_act)


@dataclass
class ScheduleBoxPolicy:
    """! @brief 手書きスケジュール: t_start − t_lean_lead から前傾・重心高さを、t_start から押付け力をランプする。
    @param spec    BoxActionSpec [-]
    @param targets 目標の実値（名前 -> 値）。None なら spec.fixed
    @param t_start 押し始め [s] / t_ramp ランプ時間 [s] / t_lean_lead 前傾の先行 [s]
    @param lean_pre_max 押す前に先行して入れる前傾の上限 [m]。外力の無い立位で CoP を足裏に保てる範囲
                   （足裏の前半長 0.085 m − 余裕）。残りは押付け力と同じランプで入れる。30 kg（68 N、前傾 13 cm）を
                   全部先行させると、押す前に CoP がつま先に張り付いて転倒した
    @note 学習の比較基準（ベースライン）。時刻は `set_time` で外から与える（obs には時刻が無い）。
    """

    spec: BoxActionSpec
    targets: dict | None = None
    t_start: float = 2.0
    t_ramp: float = 1.0
    t_lean_lead: float = 1.5
    lean_pre_max: float = 0.05
    t: float = 0.0
    needs_obs: bool = False

    def set_time(self, t: float) -> None:
        self.t = float(t)

    def values_at(self, t: float) -> dict:
        tg = dict(self.spec.fixed if self.targets is None else {**self.spec.fixed, **self.targets})
        r_push = 0.0 if t < self.t_start else min((t - self.t_start) / max(self.t_ramp, 1e-6), 1.0)
        t0 = self.t_start - self.t_lean_lead
        r_pose = 0.0 if t < t0 else min((t - t0) / max(self.t_ramp, 1e-6), 1.0)
        out = dict(tg)
        out["f_push"] = tg["f_push"] * r_push
        for k in ("com_dz", "hand_dz"):
            out[k] = tg[k] * r_pose
        L = float(tg["lean_x"])
        pre = float(np.clip(L, -self.lean_pre_max, self.lean_pre_max))
        out["lean_x"] = pre * r_pose + (L - pre) * r_push
        return out

    def act(self, obs: np.ndarray) -> np.ndarray:  # noqa: ARG002
        return self.spec.from_values(self.values_at(self.t))


@dataclass
class ReactiveBoxPolicy(ScheduleBoxPolicy):
    """! @brief 解析則のベースライン: 押付け力はスケジュール、前傾は手首 F/T の実測から決める。
    @param m_robot 体重 [kg] / omega LIPM 固有振動数 [rad/s]（前傾 = f_meas / (m omega^2)）
    @param tau_f   実測力のローパス時定数 [s]
    @param lean_min 押している間の前傾の下限 [m]（lean_pre_max と同じ値が既定）
    @note 外力 f に釣り合う CMP のずれは f / (m omega^2)。箱が滑り出して力が抜けたら前傾も戻す
          （固定の前傾だと 30 kg で前に倒れた）。RL に学ばせたい振る舞いの手書き版。
          観測（`build_box_observation`）の手の力（index 29+3, 29+4、/40 で正規化）を使う。
    """

    m_robot: float = 33.34
    omega: float = 3.90
    tau_f: float = 0.2
    lean_min: float = 0.05
    dt_decide: float = 0.1
    needs_obs: bool = True
    _f_lpf: float = 0.0

    def act(self, obs: np.ndarray) -> np.ndarray:
        vals = self.values_at(self.t)
        f_meas = 40.0 * float(obs[HL_OBS_SIZE + 3] + obs[HL_OBS_SIZE + 4])
        a = self.dt_decide / (self.tau_f + self.dt_decide)
        self._f_lpf += a * (max(f_meas, 0.0) - self._f_lpf)
        if vals["f_push"] > 0.0:
            vals["lean_x"] = max(self.lean_min, self._f_lpf / (self.m_robot * self.omega ** 2))
        return self.spec.from_values(vals)


class MlpBoxPolicy:
    """! @brief NumPy だけで推論する tanh MLP。params.npz（W0, b0, ...）と meta.json（obs_mean, obs_std, actions）。
    @note 出力層の後に tanh を掛けて行動を [-1, 1] に収める（\\f$a = \\tanh(\\mu)\\f$）。stable-baselines3 の PPO は
          学習中、方策の平均 \\f$\\mu\\f$ に探索の雑音を足した値を [-1, 1] で clip して環境に渡す（squash しない）。
          したがって学習時の行動の写像（clip）と、この推論（tanh）は同一ではない。\\f$|\\mu|\\f$ が小さい範囲では
          ほぼ一致し、\\f$|\\mu|\\f$ が大きいほど推論側の行動が小さくなる（\\f$\\mu = 1\\f$ で 0.76）。
    @note 観測は (obs - obs_mean) / obs_std で正規化するだけで、学習時の VecNormalize が掛けていた ±10 の clip は掛けない。
    @note 説明資料の評価値は、すべてこの推論（tanh、観測の clip なし）で得たものである。
    """

    def __init__(self, model_dir: str | Path, spec: BoxActionSpec | None = None) -> None:
        d = Path(model_dir)
        z = np.load(d / "params.npz")
        self.Ws = [z[f"W{i}"] for i in range(len([k for k in z.files if k.startswith("W")]))]
        self.bs = [z[f"b{i}"] for i in range(len(self.Ws))]
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8")) if (d / "meta.json").exists() else {}
        ## 基底方策（meta の "base" に相対パス）。学習時に凍結して上位の指令源にしたもの
        self.base = MlpBoxPolicy(d / meta["base"]) if meta.get("base") else None
        self.obs_mean = np.asarray(meta.get("obs_mean", np.zeros(self.Ws[0].shape[0])), dtype=np.float64)
        self.obs_std = np.asarray(meta.get("obs_std", np.ones(self.Ws[0].shape[0])), dtype=np.float64)
        if spec is None:
            spec = BoxActionSpec(actions=tuple(meta.get("actions", BoxActionSpec().actions)),
                                 decimation=int(meta.get("decimation", 50)), slow_period=int(meta.get("slow_period", 50)),
                                 preset_name=str(meta.get("preset", "")))
            if "f_push_max" in meta:
                spec.f_push_max = float(meta["f_push_max"])
            if "scales" in meta:
                spec.scales.update({k: float(v) for k, v in meta["scales"].items()})
        self.spec = spec
        if "actions" in meta and tuple(meta["actions"]) != self.spec.actions:
            raise ValueError(f"学習時の行動 {meta['actions']} と spec.actions {self.spec.actions} が違う")

    def act(self, obs: np.ndarray) -> np.ndarray:
        x = (np.asarray(obs, dtype=np.float64) - self.obs_mean) / np.maximum(self.obs_std, 1e-6)
        for i, (W, b) in enumerate(zip(self.Ws, self.bs)):
            x = x @ W + b
            if i < len(self.Ws) - 1:
                x = np.tanh(x)
        return np.clip(np.tanh(x), -1.0, 1.0)


@dataclass
class BoxPushApplier:
    """! @brief 方策の出力（実値）を計画器と手先タスクに適用する側。毎制御周期 `apply(planner, state, k)` を呼ぶ。
    @param spec      BoxActionSpec [-]
    @param policy    act(obs) -> a [-]
    @param hand_task HandTask（run_experiment にも渡す）[-]
    @param dt        制御周期 [s]
    @param box_half_x 箱の x 半長 [m]
    @param standoff  接触前の手先 site の目標を面から手前に置く距離 [m]（ゴムの手は site より 0.067 m 前）
    @param hand_y_nom 構えの手先 y shape (2,) [m] / hand_z_nom 構えの手先高さ [m]（robot_g1_box.yaml の注記）
    @param ki, ki_limit 押付け力の積分補正 [1/s], 上限（指令比）
    @param reach_max, reach_band, reach_k 手先のリーチ制限: 骨盤から押す向きに測った手先 site の距離が
                      reach_max − reach_band を超えたら押付け力を線形に絞り、reach_max を超えた分は剛性 reach_k [N/m]
                      で引き戻す。押す向きを力制御にしているので、箱が歩行より速く滑ると手が際限なく追い、
                      腕が伸び切って胴体を前に引き倒した（30 kg）
    @param rate_limits 行動の実値に掛ける変化率の上限（名前 -> 単位/s）。方策が 10 Hz で飛び飛びの値を出しても
                      体が追える速さに均す（lean 0.2 m/s、f_push 100 N/s、hand_dz 0.3 m/s。com_dz は計画器側で 0.15 m/s）
    @param vx        押しながら歩く前進速度 [m/s]（0 で立位）。t_walk から set_command
    @param t_walk    歩き始め [s]
    @param controller BipedTorqueController（残差の行動を下層に書くとき必須。None なら残差は捨てる）
    @param lean_mode  "auto"（lean_x が行動にあれば方策、無ければ解析則）/ "policy" / "analytic"
    @param m_robot, omega 前傾の解析則の体重 [kg] と LIPM 固有振動数 [rad/s]（planner があれば planner.omega）
    @param lean_gain  前傾 = lean_gain f / (m omega^2) の係数 [-]（質量と前傾の掃引 `tools/sweep_p2.py` で 1.0 が必要だった）
    @param lean_pre   押す前に先に入れておく前傾 [m] / t_lean_pre その開始時刻 [s]（None で t_walk − t_lean_lead）
    @param f_pre      押付け力の頭打ち「今の前傾で支えられる力 + f_pre」の余裕 [N]
    @param lean_max   前傾の上限 [m]
    @note f_push が 0 のあいだは手を面の手前に構える（位置制御）。0 を超えると x を力制御にする。
          観測・行動のログは `log`（None でなければ (k, obs, a, vals) を追記）。
    """

    spec: BoxActionSpec
    policy: object
    hand_task: HandTask = field(default=None)  # type: ignore[assignment]
    dt: float = 0.002
    box_half_x: float = 0.25
    standoff: float = 0.10
    hand_y_nom: tuple[float, float] = (0.189, -0.189)
    hand_z_nom: float = 0.77
    ki: float = 2.0
    ki_limit: float = 1.0
    vx: float = 0.0
    t_walk: float = 2.0
    rate_limits: dict = field(default_factory=lambda: {"lean_x": 0.2, "cmp_dy": 0.2, "f_push": 100.0, "hand_dz": 0.3})
    reach_max: float = 0.47
    reach_band: float = 0.04
    reach_k: float = 600.0
    target_vals: dict = field(default_factory=dict)   # 方策が最後に出した実値（均す前）
    _smoothed: dict = field(default_factory=dict)
    log: list | None = None
    prev_action: np.ndarray = field(default_factory=lambda: np.zeros(0))
    f_int: np.ndarray = field(default_factory=lambda: np.zeros(2))
    last_vals: dict = field(default_factory=dict)
    last_obs: np.ndarray | None = None
    tau_sat: float = 0.0
    tau_max: np.ndarray | None = None   # 与えると観測のトルク飽和率を state.tau_meas から出す
    controller: object | None = None
    q_nominal: np.ndarray | None = None
    m_total: float = 33.34
    lean_mode: str = "auto"
    m_robot: float = 33.34
    omega: float = 3.90
    lean_gain: float = 1.0
    lean_pre: float = 0.05
    t_lean_pre: float | None = None
    t_lean_lead: float = 1.5
    f_pre: float = 12.0
    lean_max: float = 0.16
    t_now: float = 0.0
    _z_rel_nom: float | None = None

    def __post_init__(self) -> None:
        if self.hand_task is None:
            self.hand_task = HandTask(n_arm=2)
        if self.prev_action.size != self.spec.n_act:
            self.prev_action = np.zeros(self.spec.n_act)
        self.last_vals = dict(self.spec.fixed)
        self.last_vals["f_push"] = 0.0
        self.last_vals["lean_x"] = 0.0
        self.last_vals["cmp_dy"] = 0.0
        self.last_vals["com_dz"] = 0.0
        self.target_vals = dict(self.last_vals)

    @property
    def lean_analytic(self) -> bool:
        """! @brief 前傾を解析則で決めるか。"""
        if self.lean_mode == "auto":
            return not self.spec.uses_lean_action
        return self.lean_mode == "analytic"

    def reset(self) -> None:
        self.prev_action = np.zeros(self.spec.n_act)
        self.t_now = 0.0
        self.f_int[:] = 0.0
        self._z_rel_nom = None
        self._smoothed = {}
        self.last_obs = None
        self.__post_init__()

    def observe(self, planner, state, p_hands: np.ndarray) -> np.ndarray:
        if self._z_rel_nom is None:
            self._z_rel_nom = self.hand_z_nom - float(state.c_W[2])
        if self.tau_max is not None:
            tm = np.asarray(state.tau_meas, dtype=np.float64)
            if tm.size == self.tau_max.size:
                self.tau_sat = float(np.mean(np.abs(tm) >= 0.95 * self.tau_max))
        obs = build_box_observation(planner, state, self.hand_task, self.prev_action, self.spec,
                                    self.box_half_x, self._z_rel_nom, self.tau_sat, p_hands)
        if self.spec.residual_actions:
            info = getattr(self.controller, "last_info", None)
            obs = np.concatenate([obs, build_residual_observation(planner, state, info, self.prev_action, self.spec,
                                                                  self.q_nominal, self.m_total)])
        return obs

    def decide(self, planner, state, k: int, p_hands: np.ndarray) -> dict:
        """! @brief 方策を呼び、行動の実値を返す（decimation ごと）。"""
        t = k * self.dt
        if hasattr(self.policy, "set_time"):
            self.policy.set_time(t)
        # 観測を使わない方策（Zero / Schedule）では組まない（計画器が未初期化でも動く。計算も省く）
        obs = self.observe(planner, state, p_hands) if getattr(self.policy, "needs_obs", True) \
            else np.zeros(box_obs_size(self.spec))
        a = np.clip(np.asarray(self.policy.act(obs), dtype=np.float64).reshape(self.spec.n_act), -1.0, 1.0)
        a = self.spec.hold_slow(a, self.prev_action, k)
        self.prev_action = a
        self.last_obs = obs
        vals = self.spec.to_values(a)
        self.target_vals = vals
        if self.log is not None:
            self.log.append((k, obs, a, vals))
        return vals

    def _rate_limit(self, vals: dict) -> dict:
        """! @brief 変化率の上限で均した実値を返す（内部状態 `_smoothed` を進める）。"""
        out = dict(vals)
        for name, rate in self.rate_limits.items():
            if name not in vals:
                continue
            cur = self._smoothed.get(name, 0.0)
            step = rate * self.dt
            d = float(vals[name]) - cur
            cur += float(np.clip(d, -step, step))
            self._smoothed[name] = cur
            out[name] = cur
        return out

    def _analytic_lean(self, planner, state, vals: dict) -> dict:  # noqa: ARG002
        """! @brief 前傾の解析則（脚側）: 押付け力の指令 f から CMP のずれ lean_gain f / (m omega^2) を作る。
        @note 押す前（t_lean_pre 以降）は lean_pre だけ先に前へ出しておく（重心が後ろに残ると
              手がリーチ制限に掛かって力が伝わらず、箱が滑り出したときに前へ倒れた）。
              変化率は `_rate_limit`、押付け力の頭打ちは `_gate_force`。
        """
        out = dict(vals)
        t_pre = max(self.t_walk - self.t_lean_lead, 0.0) if self.t_lean_pre is None else self.t_lean_pre
        base = self.lean_pre if self.t_now >= t_pre else 0.0
        f_tgt = max(float(vals["f_push"]), 0.0)
        om = float(getattr(planner, "omega", self.omega))
        out["lean_x"] = float(np.clip(max(base, self.lean_gain * f_tgt / (self.m_robot * om ** 2)), 0.0, self.lean_max))
        return out

    def _gate_force(self, planner, vals: dict) -> dict:
        """! @brief 押付け力を「今の前傾で支えられる力 + f_pre」で頭打ちにする（前傾を力に先行させる）。"""
        om = float(getattr(planner, "omega", self.omega))
        f_allow = float(vals["lean_x"]) * self.m_robot * om ** 2 / max(self.lean_gain, 1e-6) + self.f_pre
        out = dict(vals)
        out["f_push"] = min(float(vals["f_push"]), f_allow)
        return out

    def _apply_residual(self, planner, vals: dict) -> None:
        """! @brief 残差の行動（パターン系）を {W} に回して下層に書く。"""
        if self.controller is None or not self.spec.residual_actions or not hasattr(self.controller, "set_external_residual"):
            return
        R2 = planner.rot2() if hasattr(planner, "rot2") else np.eye(2)
        dcop = R2 @ np.array([vals.get("cop_dx", 0.0), vals.get("cop_dy", 0.0)])
        mxy = R2 @ np.array([vals.get("mom_x", 0.0), vals.get("mom_y", 0.0)])
        tau = np.array([vals.get(n, 0.0) for n in TAU_LEG_NAMES]) if self.spec.uses_tau_leg else None
        self.controller.set_external_residual(dcop=dcop, dmom=np.array([mxy[0], mxy[1], 0.0]), tau_leg=tau)

    def apply_values(self, planner, state, vals: dict) -> None:
        """! @brief 行動の実値を計画器と手先タスクに書く（毎制御周期）。lean_x / f_push / hand_dz は変化率を制限する。"""
        if self.lean_analytic:
            vals = self._analytic_lean(planner, state, vals)
        vals = self._rate_limit(vals)
        if self.lean_analytic:
            vals = self._gate_force(planner, vals)
        self.t_now += self.dt
        self.last_vals = vals
        self._apply_residual(planner, vals)
        # 足配置残差（lococtrl HL の口。次の 1 歩に効く）
        if (vals["step_dx"] != 0.0 or vals["step_dy"] != 0.0 or vals["step_dt"] != 0.0) \
                and hasattr(planner, "set_step_residual") and getattr(planner, "initialized", True):
            # 残差が 0 のときは呼ばない（hl_active が立つと計画器の到達判定・足揃えの挙動が変わる）
            dxy_W = planner.rot2() @ np.array([vals["step_dx"], vals["step_dy"]])
            planner.set_step_residual(dxy_W, vals["step_dt"])
        if hasattr(planner, "set_lean"):
            planner.set_lean(vals["lean_x"], vals.get("cmp_dy", 0.0))
        if hasattr(planner, "set_com_height"):
            planner.set_com_height(planner.z_c_nominal + vals["com_dz"])
        # 手先
        bp = np.asarray(state.box_pose, dtype=np.float64)
        if bp.size < 7:
            return
        face_x = float(bp[0]) - self.box_half_x
        # しゃがみに合わせて手も下げる。実測の重心ではなく計画器の重心高さ指令で（初期の沈み込みを拾わない）
        dz_body = float(getattr(planner, "z_c", 0.0) - getattr(planner, "z_c_nominal", 0.0))
        f = float(vals["f_push"])
        wh = np.asarray(state.wrench_hands, dtype=np.float64)
        Rt = planner.rot2().T if hasattr(planner, "rot2") else np.eye(2)
        p_base = np.asarray(getattr(state, "p_WB", np.zeros(3)), dtype=np.float64)
        ph = self._p_hands(state)
        for a in range(2):
            # リーチ制限（骨盤から押す向きに測った手先の距離）
            reach = float((Rt @ (ph[a, 0:2] - p_base[0:2]))[0]) if np.any(ph[a]) else 0.0
            g_reach = float(np.clip((self.reach_max - reach) / max(self.reach_band, 1e-6), 0.0, 1.0))
            f_back = -self.reach_k * max(reach - self.reach_max, 0.0)
            if f > 0.0 and self.ki > 0.0 and wh.shape == (2, 6):
                self.f_int[a] += self.ki * (0.5 * f - (-float(wh[a, 0]))) * self.dt
                lim = self.ki_limit * 0.5 * max(f, 1.0)
                self.f_int[a] = float(np.clip(self.f_int[a], -lim, lim))
            elif f <= 0.0:
                self.f_int[a] = 0.0
            y = float(self.hand_y_nom[a])
            z = self.hand_z_nom + float(vals["hand_dz"]) + dz_body
            if f <= 0.0:
                self.hand_task.set_target(a, np.array([face_x - self.standoff, y, z]), f_ff=None, active=True)
            else:
                fx = g_reach * (0.5 * f + self.f_int[a]) + f_back
                if g_reach < 1.0:
                    self.f_int[a] *= g_reach          # 届かないあいだは積分を溜めない
                self.hand_task.set_target(a, np.array([face_x, y, z]), f_ff=np.array([fx, 0.0, 0.0]),
                                          active=True, kp_axis=(0.0, 1.0, 1.0))

    def __call__(self, planner, state, k: int) -> None:
        """! @brief on_step フック（run_experiment の extra_on_step）。"""
        if self.vx != 0.0 and hasattr(planner, "set_command") and k * self.dt >= self.t_walk:
            planner.set_command(self.vx, 0.0, 0.0)
        p_hands = self._p_hands(state)
        if k % max(1, int(self.spec.decimation)) == 0 and getattr(planner, "initialized", True):
            self.decide(planner, state, k, p_hands)
        self.apply_values(planner, state, self.target_vals)

    @staticmethod
    def _p_hands(state) -> np.ndarray:
        """! @brief 手先位置 {W}。RobotState に無ければ 0（観測の手先高さは 0 になる）。"""
        p = np.asarray(getattr(state, "p_hands", np.zeros((0, 3))), dtype=np.float64)
        return p if p.shape == (2, 3) else np.zeros((2, 3))
