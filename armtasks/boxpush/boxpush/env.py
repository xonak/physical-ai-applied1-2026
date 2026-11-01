"""! @file env.py
@brief 箱押しの学習用環境 `BoxPushEnv`（gym 風）と gymnasium ラッパ `BoxPushGymEnv`。
RL から見た環境の定義

1 step = 方策の 1 判断（`BoxActionSpec.decimation` 制御周期。A / P は 10 Hz、CoP 残差を含む B / P_S は 100 Hz、
トルク残差を含む C / P_S_T は制御周期と同じ 500 Hz。腕・計画層の行動は 100 ms ごとにしか更新しない）。
reset() で箱の質量・摩擦・初期位置を乱択する（ドメインランダム化。質量は観測に入れない）。
制御は lococtrl の dlr（受動性ベース全身制御）+ オンライン計画器（endless、立位から）。
`BipedExperimentConfig`（`configs/experiments/*.yaml` の lococtrl 部分）をそのまま環境の設定にする。

報酬（1 判断あたり、各制御周期で積算 × dt）:
  + w_progress · v_box,x          箱が前に進む
  + w_contact · 1[f_meas ≥ 0.8 f_cmd]   押している（空振り・押し負けでない）
  − w_sat · トルク飽和率          関節トルクの上限に張り付かない
  − w_slip · Σ |v_foot,xy|（接触中） 足が滑らない
  − w_dcm · |e_DCM|²  − w_post · (roll² + pitch²)  − w_act · |a − a_prev|²（10 Hz 換算）
  − w_res · |a_res|²（残差の行動の大きさ。残差を含む条件のみで、RL-M1 / RL-M2 には無い）
  転倒（roll / pitch 40 deg）で −50 して終了。箱が倒れた（roll / pitch 20 deg）ら −20 して終了。
  打ち切り（max_time）で箱が success_dx 以上進んでいれば +success（評価の成功判定と同じ）。
前傾の量（CMP オフセット）や重心の高さそのものは報酬に含めない。胴体の傾きは w_post で罰する
（条件 P 系は 0、`boxpush.policy.PRESET_REWARD`）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from lococtrl.cli.biped_experiment import BipedExperimentConfig
from lococtrl.common.config import load_robot_config
from lococtrl.common.frames import quat_to_euler_zyx
from lococtrl.control.biped_torque_controller import BipedTorqueController
from lococtrl.control.hand_task import HandTask
from lococtrl.control.wbc_passivity import PassivityWbc
from lococtrl.gait.biped_schedule import gait_params_from_config
from lococtrl.model_dynamics.robot_model import create_robot_model
from lococtrl.planner.biped_online import OnlineBipedPlanner, OnlinePlannerParams
from lococtrl.sim.mujoco_env import MujocoEnv

from .policy import BOX_OBS_SIZE, PRESET_REWARD, BoxActionSpec, BoxPushApplier, box_obs_size

__all__ = ["BoxDomain", "RewardWeights", "BoxPushEnv", "make_gym_env"]


@dataclass
class BoxDomain:
    """! @brief ドメインランダム化の範囲（reset ごとに一様乱択）。
    @param mass       箱の質量 [kg]（下限, 上限）
    @param friction   箱-床の滑り摩擦 [-]
    @param x_offset   箱の初期 x の設定値からのずらし [m]
    @param randomize  False なら設定値のまま
    """

    mass: tuple[float, float] = (10.0, 40.0)
    friction: tuple[float, float] = (0.15, 0.35)
    x_offset: tuple[float, float] = (0.0, 0.04)
    randomize: bool = True


@dataclass
class RewardWeights:
    """! @brief 報酬の重み。1 判断あたりの報酬 = Σ w_i r_i dt。
    @param w_progress  箱の前進速度の重み
    @param w_contact   押している（空振り・押し負けでない）の重み
    @param w_sat       関節トルクの上限に張り付かないの重み
    @param w_slip      接触中の足先の水平速度の重み
    @param w_dcm       DCM 位置誤差の重み
    @param w_post      姿勢の傾き（roll/pitch）の重み
    @param w_act       行動の変化量の重み（10 Hz の判断 1 回あたり。判断周期が違うときは比で換算）
    @param w_res       残差の行動（[-1,1]）の大きさの重み（1 s あたり）
    @param fall        転倒ペナルティ（roll/pitch 40 deg）
    @param box_tip     箱が倒れたペナルティ（roll/pitch 20 deg）
    @param success     成功報酬（打ち切り時に箱が success_dx 以上進んでいれば）
    @param success_dx  成功判定の箱の前進距離 [m]
    """
    w_progress: float = 10.0
    w_contact: float = 0.5
    w_sat: float = 2.0
    w_slip: float = 5.0
    w_dcm: float = 10.0
    w_post: float = 1.0
    w_act: float = 0.05
    w_res: float = 0.5
    fall: float = 50.0
    box_tip: float = 20.0
    success: float = 20.0
    success_dx: float = 0.5


class BoxPushEnv:
    """! @brief 箱押しの学習用環境（gym 風。reset() -> obs、step(a) -> (obs, reward, done, info)）。"""

    def __init__(self, cfg: BipedExperimentConfig, spec: BoxActionSpec | None = None,
                 domain: BoxDomain | None = None, weights: RewardWeights | None = None,
                 max_time: float = 10.0, t_walk: float | None = None, vx: float = 0.1,
                 seed: int | None = None, hand_kp: float = 300.0, hand_kd: float = 30.0, hand_f_max: float = 90.0,
                 base_policy=None) -> None:
        """! @brief 構築する。
        @param cfg      lococtrl の実験設定（robot / mapper dlr / planner online endless / stabilizer / scenario.box）[-]
        @param spec     行動仕様 [-]
        @param domain   ランダム化 [-]
        @param weights  報酬重み [-]
        @param max_time 1 エピソードの長さ [s]
        @param vx / t_walk 押しながら歩くときの前進速度 [m/s] と歩き始め [s]（既定 0.1 m/s で押しながら歩く。0 で立位）
        @param seed     乱数の種 [-]
        @param base_policy 基底方策（MlpBoxPolicy など、.spec を持つ）。凍結して上位の指令源にする（None で無し）
        """
        if cfg.mapper.kind != "dlr":
            raise ValueError("BoxPushEnv は mapper.kind = dlr のみ（手先タスクは dlr だけ）")
        if cfg.planner.kind != "online" or not cfg.planner.online.endless:
            raise ValueError("BoxPushEnv は planner.kind = online, endless = true で使う")
        if not cfg.scenario.box.enable:
            raise ValueError("BoxPushEnv には scenario.box.enable = true が要る")
        import dataclasses
        from dataclasses import asdict

        self.cfg = cfg
        self.spec = BoxActionSpec() if spec is None else spec
        self.domain = BoxDomain() if domain is None else domain
        self.w = RewardWeights() if weights is None else weights
        if weights is None:
            for k, v in PRESET_REWARD.get(self.spec.preset_name, {}).items():
                setattr(self.w, k, v)          # 条件ごとの上書き（P 系は姿勢を罰しない）
        self.rng = np.random.default_rng(seed)
        self.base = base_policy
        self.base_prev = np.zeros(self.base.spec.n_act) if self.base is not None else None
        self.base_vals: dict = {}
        self.max_time = float(max_time)
        self.dt_ctrl = float(cfg.sim.dt_ctrl)
        self.rcfg = load_robot_config(cfg.robot)
        self.model = create_robot_model(self.rcfg)
        self.box0 = cfg.scenario.box.to_object()
        self.env = MujocoEnv(self.rcfg, solref=None if cfg.sim.solref is None else tuple(cfg.sim.solref), objects=[self.box0])
        self.par = dataclasses.replace(cfg.stabilizer)
        self.par.extra = {"ihmc": asdict(cfg.mapper.ihmc), "dlr": asdict(cfg.mapper.dlr)}
        d = asdict(cfg.planner.online)
        d.pop("velocity_schedule"); d.pop("footsteps")
        d["lean_xy"] = tuple(float(v) for v in d["lean_xy"])
        d["duration"] = max(float(d["duration"]), self.max_time + 1.0)
        self.planner_params = OnlinePlannerParams(**d)
        self.gait = gait_params_from_config(asdict(cfg.gait))
        self.hand = (hand_kp, hand_kd, hand_f_max)
        self.vx, self.t_walk = float(vx), (2.0 if t_walk is None else float(t_walk))
        self.decim_sim = max(1, int(round(self.dt_ctrl / self.env.dt)))
        # 箱の body / geom
        import mujoco

        self._bid = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_BODY, self.box0.name)
        self._gid = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_GEOM, f"{self.box0.name}_geom")
        self._mass0 = float(self.env.model.body_mass[self._bid])
        self._inertia0 = np.array(self.env.model.body_inertia[self._bid])
        ## 毎制御周期に呼ぶ (env, k) -> None（動画の書き出しなど）
        self.frame_cb = None
        self.box_x0 = 0.0
        self.ctrl: BipedTorqueController | None = None
        self.applier: BoxPushApplier | None = None
        self.state = None
        self.k = 0
        self.n_fall = 0
        self.episode_info: dict = {}

    # ---- gym 風 -----------------------------------------------------------
    @property
    def obs_size(self) -> int:
        return box_obs_size(self.spec)

    @property
    def act_size(self) -> int:
        return self.spec.n_act

    def _randomize_box(self) -> dict:
        m, dm = self.env.model, self.domain
        if dm.randomize:
            mass = float(self.rng.uniform(*dm.mass))
            mu = float(self.rng.uniform(*dm.friction))
            dx = float(self.rng.uniform(*dm.x_offset))
        else:
            mass, mu, dx = self.box0.mass, self.box0.friction[0], 0.0
        m.body_mass[self._bid] = mass
        m.body_inertia[self._bid] = self._inertia0 * (mass / self._mass0)
        m.geom_friction[self._gid, 0] = mu
        a, _ = self.env._box_free[0]
        q0 = self.env._box_init[0].copy()
        q0[0] += dx
        self.env.data.qpos[a : a + 7] = q0
        return {"box_mass": mass, "box_friction": mu, "box_dx": dx, "static_friction_N": mu * mass * 9.81}

    def reset(self, policy_for_schedule=None) -> np.ndarray:
        """! @brief エピソードを始める。箱を乱択し、立位から。
        @return obs shape (40,)
        """
        self.state = self.env.reset(q_joint=self.rcfg.q_nominal, p_WB=np.array([0.0, 0.0, self.rcfg.h_nominal]))
        self.episode_info = self._randomize_box()
        import mujoco

        mujoco.mj_forward(self.env.model, self.env.data)
        self.state = self.env.observe()
        planner = OnlineBipedPlanner(self.model, self.rcfg, self.gait, self.planner_params, self.dt_ctrl)
        planner.set_command(0.0, 0.0, 0.0)
        wbc = PassivityWbc(self.model, self.rcfg, self.par)
        kp, kd, fmax = self.hand
        ht = HandTask(n_arm=2, kp=kp, kd=kd, f_max=fmax)
        wbc.hand_task = ht
        self.ctrl = BipedTorqueController(self.model, self.rcfg, planner, self.par, wbc)
        self.ctrl.reset()
        self.applier = BoxPushApplier(spec=self.spec, policy=None, hand_task=ht, dt=self.dt_ctrl,
                                      box_half_x=float(self.box0.half_size[0]), vx=self.vx, t_walk=self.t_walk,
                                      tau_max=np.asarray(self.rcfg.tau_max), controller=self.ctrl,
                                      q_nominal=np.asarray(self.rcfg.q_nominal), m_total=float(self.model.total_mass),
                                      m_robot=float(self.model.total_mass), omega=float(planner.omega))
        self.k = 0
        if self.base is not None:
            self.base_prev = np.zeros(self.base.spec.n_act)
            self.base_vals = {}
            if self.base.spec.uses_lean_action:
                self.applier.lean_mode = "policy"     # 前傾は基底方策が出す（脚の解析則は使わない）
        self.box_x0 = float(self.env.box_pose(0)[0])
        # 1 周期回して計画器に足位置を入れる
        self._cycle(self.applier.target_vals)
        return self.applier.observe(self.planner, self.state, self.applier._p_hands(self.state))

    @property
    def planner(self):
        return self.ctrl.planner

    def _merge_base(self, vals: dict) -> dict:
        """! @brief 基底方策（凍結）を 100 ms ごとに呼び、その出力を上位の指令に合成する。"""
        ap = self.applier
        bs = self.base.spec
        if not self.base_vals or (self.k % bs.decimation) < max(1, int(self.spec.decimation)):
            from .policy import build_box_observation

            obs_b = build_box_observation(self.planner, self.state, ap.hand_task, self.base_prev, bs,
                                          ap.box_half_x, ap._z_rel_nom or 0.0, ap.tau_sat, ap._p_hands(self.state))
            a_b = np.clip(np.asarray(self.base.act(obs_b), dtype=np.float64).reshape(bs.n_act), -1.0, 1.0)
            self.base_prev = a_b
            self.base_vals = bs.to_values(a_b)
        return self.spec.merge_base(vals, self.base_vals, bs.actions)

    def _cycle(self, vals: dict) -> dict:
        """! @brief 制御 1 周期（行動の実値を適用 → 制御 → 物理）。"""
        ap = self.applier
        if ap.vx != 0.0 and self.k * self.dt_ctrl >= ap.t_walk:
            self.planner.set_command(ap.vx, 0.0, 0.0)
        ap.apply_values(self.planner, self.state, vals)
        u = self.ctrl.compute(self.state, self.k)
        self.env.apply(u)
        self.env.step(self.decim_sim)
        self.state = self.env.observe()
        self.k += 1
        if self.frame_cb is not None:
            self.frame_cb(self, self.k)
        return {"u": u, "info": self.ctrl.last_info}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        """! @brief 1 判断ぶん進める。
        @param action shape (n_act,) [-1,1]
        @return (obs, reward, done, info)
        """
        ap, w = self.applier, self.w
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(self.spec.n_act), -1.0, 1.0)
        a = self.spec.hold_slow(a, ap.prev_action, self.k)      # 腕・計画層は 100 ms ごとにだけ更新
        da = a - ap.prev_action
        ap.prev_action = a
        vals = self.spec.to_values(a)
        if self.base is not None:
            vals = self._merge_base(vals)
        ap.target_vals = vals
        reward = 0.0
        done = False
        info: dict = dict(self.episode_info)
        Rt = self.planner.rot2().T
        for _ in range(max(1, int(self.spec.decimation))):
            out = self._cycle(vals)
            st, ci = self.state, out["info"]
            bp = np.asarray(st.box_pose)
            v_box = float((Rt @ bp[7:9])[0]) if bp.size >= 10 else 0.0
            wh = np.asarray(st.wrench_hands)
            f_meas = float(-wh[:, 0].sum()) if wh.shape == (2, 6) else 0.0
            f_cmd = float(ap.last_vals["f_push"])   # 変化率制限後の指令
            contact_ok = 1.0 if (f_cmd > 1.0 and f_meas >= 0.8 * f_cmd) else 0.0
            tau = np.asarray(out["u"].tau)
            sat = float(np.mean(np.abs(tau) >= 0.95 * np.asarray(self.rcfg.tau_max)))
            # 足の滑り（接触中の足先の水平速度）
            slip = 0.0
            q = self.model.make_q(st.p_WB, st.quat_WB, st.q)
            v = self.model.make_v(np.asarray(st.v_B), st.w_B, st.dq)
            for i in range(self.rcfg.n_leg):
                if st.contact[i]:
                    vf = self.model.jacobian_foot(q, i) @ v
                    slip += float(np.linalg.norm(vf[0:2]))
            e_xi = ci.xi_hat - ci.ref.xi_ref
            rpy = quat_to_euler_zyx(st.quat_WB)[::-1]
            reward += (w.w_progress * v_box + w.w_contact * contact_ok - w.w_sat * sat - w.w_slip * slip
                       - w.w_dcm * float(e_xi @ e_xi) - w.w_post * float(rpy[0] ** 2 + rpy[1] ** 2)) * self.dt_ctrl
            fell = abs(rpy[0]) > np.radians(40.0) or abs(rpy[1]) > np.radians(40.0)
            box_rp = quat_to_euler_zyx(bp[3:7])[::-1] if bp.size >= 7 else np.zeros(3)
            tipped = abs(box_rp[0]) > np.radians(20.0) or abs(box_rp[1]) > np.radians(20.0)
            if fell:
                reward -= w.fall; done = True; info["fell"] = True; self.n_fall += 1
                break
            if tipped:
                reward -= w.box_tip; done = True; info["box_tipped"] = True
                break
            if self.k * self.dt_ctrl >= self.max_time:
                done = True
                if bp.size and float(bp[0]) - self.box_x0 >= w.success_dx:
                    reward += w.success
                    info["success"] = True
                break
        dec = max(1, int(self.spec.decimation))
        reward -= w.w_act * (dec / 50.0) * float(da @ da)
        ridx = self.spec.residual_index
        if ridx.size:
            reward -= w.w_res * float(a[ridx] @ a[ridx]) * dec * self.dt_ctrl
        info.update(k=self.k, t=self.k * self.dt_ctrl, f_meas=f_meas, f_cmd=f_cmd, v_box=v_box,
                    box_x=float(bp[0]) if bp.size else 0.0, box_dx=float(bp[0]) - self.box_x0 if bp.size else 0.0,
                    com_z=float(self.state.c_W[2]),
                    lean_x=float(self.planner.lean_xy[0]), lean_y=float(self.planner.lean_xy[1]),
                    base_vals=dict(self.base_vals) if self.base is not None else None,
                    z_c=float(self.planner.z_c), vals=vals)
        obs = ap.observe(self.planner, self.state, ap._p_hands(self.state))
        return obs, float(reward), done, info


def make_gym_env(cfg: BipedExperimentConfig, **kw):
    """! @brief gymnasium の Env にして返す（stable-baselines3 用）。gymnasium が無ければ ImportError。
    @param cfg BipedExperimentConfig [-]
    @param kw  BoxPushEnv の引数 [-]
    @return gymnasium.Env
    """
    import gymnasium as gym

    class BoxPushGymEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self):
            super().__init__()
            self.inner = BoxPushEnv(cfg, **kw)
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(self.inner.obs_size,), dtype=np.float32)
            self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(self.inner.act_size,), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            if seed is not None:
                self.inner.rng = np.random.default_rng(seed)
            obs = self.inner.reset()
            return obs.astype(np.float32), dict(self.inner.episode_info)

        def step(self, action):
            obs, r, done, info = self.inner.step(np.asarray(action, dtype=np.float64))
            truncated = bool(done and not info.get("fell") and not info.get("box_tipped"))
            terminated = bool(done and not truncated)
            return obs.astype(np.float32), float(r), terminated, truncated, info

    return BoxPushGymEnv()
