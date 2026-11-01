"""! @file biped_torque_run.py
@brief トルク制御の全身制御（ihmc / dlr / rl）で G1 を歩かせる実行関数。

`run_biped.py --mapper ihmc|dlr|rl` から呼ばれる。`run_biped_walk`（IK + 関節 PD）と
同じ歩容・同じ評価指標で比較できるように、戻り値は `BipedWalkResult` に揃える。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..bench.metrics import foot_support_polygon, point_in_polygon
from ..common.config import load_robot_config
from ..common.frames import quat_to_euler_zyx
from ..control.biped_stack import BipedPlannerLayer, StabilizerParams
from ..control.biped_torque_controller import BipedTorqueController
from ..control.contact_wrench import FootContactSpec, wrench_to_cop
from ..gait.biped_schedule import BipedGaitParams, SegKind
from ..model_dynamics.robot_model import create_robot_model
from ..sim.mujoco_env import MujocoEnv

__all__ = ["TorqueWalkLog", "make_mapper", "run_biped_torque"]


@dataclass
class TorqueWalkLog:
    """! @brief トルク制御固有のログ（`BipedWalkResult.extra` に入れる）。
    @param xi_hat     推定 DCM shape (T,2) [m]
    @param xi_ref     参照 DCM shape (T,2) [m]
    @param cop_cmd    CoP 指令 shape (T,2) [m]
    @param cop_ft     足裏 F/T から求めた CoP shape (T,2) [m]
    @param tau        指令トルク shape (T,n) [N*m]
    @param tau_meas   関節トルクセンサ shape (T,n) [N*m]
    @param wrench_cmd 全身制御器が決めた接触レンチ shape (T,nf,6) [N, N*m]
    @param wrench_meas 足裏 F/T の床反力レンチ shape (T,nf,6) [N, N*m]
    @param load       荷重係数 shape (T,nf) [-]
    @param solve_time QP 解時間 shape (T,) [s]
    @param status     QP が解けたか shape (T,) [bool]
    @param wrench_hands 手首 F/T の手の接触レンチ shape (T,n_arm,6) [N, N*m]（腕のある機体のみ）
    @param f_hand_cmd 手先力の指令 shape (T,n_arm,3) [N]（手先タスクがあるとき）
    @param p_hands    手先位置 {W} shape (T,n_arm,3) [m]（腕のある機体のみ）
    @param box_pose   箱の位置・姿勢・速度 shape (T,10)（箱があるときのみ）
    @param q_arm_err  腕の関節角 - 姿勢参照 shape (T,n_arm_joint) [rad]（腕のある機体のみ）
    """

    xi_hat: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    xi_ref: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    cop_cmd: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    cop_ft: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    tau: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    tau_meas: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    wrench_cmd: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 6)))
    wrench_meas: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 6)))
    load: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    solve_time: np.ndarray = field(default_factory=lambda: np.zeros(0))
    status: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    wrench_hands: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 6)))
    f_hand_cmd: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 3)))
    p_hands: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 3)))
    box_pose: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    q_arm_err: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))


def make_mapper(name: str, model, rcfg, par: StabilizerParams):
    """! @brief 名前から全身制御器を作る。
    @param name  "ihmc" | "dlr" [-]
    @param model RobotModelBase [-]
    @param rcfg  RobotConfig [-]
    @param par   StabilizerParams [-]
    @return 全身制御器
    """
    if name == "ihmc":
        from ..control.wbc_momentum import MomentumWbc

        return MomentumWbc(model, rcfg, par)
    if name == "dlr":
        from ..control.wbc_passivity import PassivityWbc

        return PassivityWbc(model, rcfg, par)
    raise ValueError(f"未知の全身制御器: {name}（ihmc | dlr）")


def run_biped_torque(
    robot: str = "robot_g1.yaml",
    gait_overrides: dict | None = None,
    dt_ctrl: float = 0.002,
    mapper: str = "ihmc",
    k_dcm: float | None = None,
    policy_dir: str | None = None,
    rl_base: str = "ihmc",
    verbose: bool = True,
    summarize_fn=None,
    result_cls=None,
    params_fn=None,
    par: StabilizerParams | None = None,
    scenario=None,
    solref: tuple[float, float] | None = None,
    tau_scale: float | None = None,
    planner_kind: str = "schedule",
    planner_params=None,
    on_step=None,
    viewer: bool = False,
    realtime: bool = False,
    command_source=None,
    view_every: int = 5,
    hand_task=None,
):
    """! @brief トルク制御の全身制御で二足歩行を実行し、評価指標を測る。
    @param robot          ロボット設定ファイル名 [-]
    @param gait_overrides 歩容パラメータの上書き [-]
    @param dt_ctrl        制御周期 [s]
    @param mapper         "ihmc" | "dlr" | "rl" [-]
    @param k_dcm          DCM フィードバックゲイン（None で biped.yaml の値）[-]
    @param policy_dir     rl のときの学習済み方策（None で残差ゼロ = ベースそのもの）[-]
    @param rl_base        rl のベース全身制御器 "ihmc" | "dlr" [-]
    @param verbose        進捗を表示するか [-]
    @param summarize_fn   `run_biped.summarize` [-]
    @param result_cls     `run_biped.BipedWalkResult` [-]
    @param params_fn      `run_biped._biped_params` [-]
    @param par            StabilizerParams（None なら biped.yaml から。実験設定ファイル経由で渡す）[-]
    @param scenario       外乱（`biped_experiment.ScenarioSection`。None で無し）[-]
    @param solref         接触の solref（None で MJCF のまま）[-]
    @param tau_scale      残差方策の上限 [N*m]（None で方策の既定）
    @param planner_kind   "schedule"（オフライン固定列）| "online"（オンライン計画）[-]
    @param planner_params online のとき `planner.biped_online.OnlinePlannerParams`（None で既定）[-]
    @param on_step        毎周期呼ぶフック (planner, state, k) -> None（速度変更・足配置ストリームの注入に使う）[-]
    @param viewer         MuJoCo ビューアを開く [-]（ヘッドレスなら自動で無効）
    @param realtime       実時間で回す [-]
    @param command_source テレオペ入力（`teleop.command_source` の poll() を持つもの）。与えると毎周期 poll して
                          planner.set_command に流し、push / quit も処理する（planner は online を想定）[-]
    @param view_every     ビューアを更新する周期の間引き [-]（描画コストを制御周期に乗せない）
    @param hand_task      手先タスク（`control.hand_task.HandTask`）。dlr の全身制御器に渡す（None で無し）[-]
    @return BipedWalkResult（`extra` に TorqueWalkLog）
    """
    from .run_biped import BipedWalkResult, _biped_params, summarize

    summarize_fn = summarize if summarize_fn is None else summarize_fn
    result_cls = BipedWalkResult if result_cls is None else result_cls
    params_fn = _biped_params if params_fn is None else params_fn

    cfg = load_robot_config(robot)
    model = create_robot_model(cfg)
    box_obj = None
    if scenario is not None and getattr(scenario, "box", None) is not None:
        box_obj = scenario.box.to_object()
    env = MujocoEnv(cfg, solref=None if solref is None else tuple(solref),
                    objects=None if box_obj is None else [box_obj])
    params: BipedGaitParams = params_fn(gait_overrides)
    if par is None:
        par = StabilizerParams.from_config()
    if k_dcm is not None:
        par.k_dcm = float(k_dcm)
    spec = FootContactSpec.from_config(cfg)
    if planner_kind == "online":
        from ..planner.biped_online import OnlineBipedPlanner, OnlinePlannerParams

        planner = OnlineBipedPlanner(
            model, cfg, params, OnlinePlannerParams() if planner_params is None else planner_params, dt_ctrl
        )
    elif planner_kind == "schedule":
        planner = BipedPlannerLayer(model, cfg, params, dt_ctrl)
    else:
        raise ValueError(f"未知の planner_kind: {planner_kind}（schedule | online）")

    residual = None
    obs_fn = None
    base_name = mapper
    if mapper == "rl":
        from ..policy.biped_residual_policy import MlpResidualPolicy, ZeroResidualPolicy, build_observation

        base_name = rl_base
        residual = (
            ZeroResidualPolicy(cfg.n_joint) if policy_dir is None
            else MlpResidualPolicy(policy_dir, cfg.n_joint)
        )
        if tau_scale is not None and hasattr(residual, "tau_scale"):
            residual.tau_scale = float(tau_scale)
        obs_fn = build_observation
    wbc = make_mapper(base_name, model, cfg, par)
    if hand_task is not None:
        if not hasattr(wbc, "hand_task"):
            raise ValueError(f"全身制御器 {base_name} は手先タスクに未対応（dlr を使う）")
        wbc.hand_task = hand_task
    ctrl = BipedTorqueController(model, cfg, planner, par, wbc, residual=residual, residual_obs_fn=obs_fn)

    if verbose:
        print(f"[torque {mapper} / planner {planner_kind}] z_c = {planner.z_c:.4f} m, omega = {planner.omega:.4f} rad/s, k_dcm = {par.k_dcm}")
        print(f"歩数 {params.n_step}, 周期 {params.period:.3f} s, "
              f"平均速度 {params.mean_speed:.3f} m/s, 全体 {planner.total_time:.2f} s")
        print(f"関節トルクセンサ: {env.has_joint_torque_sensors}, 足裏 F/T: {env.has_foot_ft_sensors}")

    st = env.reset(q_joint=cfg.q_nominal, p_WB=np.array([0.0, 0.0, cfg.h_nominal]))
    ctrl.reset()
    nf = cfg.n_leg
    decim = max(1, int(round(dt_ctrl / env.dt)))
    rec: dict[str, list] = {k: [] for k in (
        "t", "c_ref", "c_meas", "cop", "zmp_ref", "p_WF", "con_sched", "con_meas", "rpy",
        "inside", "xi_hat", "xi_ref", "cop_cmd", "cop_ft", "tau", "tau_meas", "w_cmd", "w_meas",
        "load", "solve", "status", "w_hands", "box", "q_arm_err", "f_hand_cmd", "p_hands")}
    n_lj = cfg.n_leg_joint
    fell = False
    sch = getattr(planner, "schedule", None)

    push_done = False
    from ..common.timing import LoopPacer
    from ..sim.viewer import Viewer

    view = Viewer(env, enable=viewer, key_callback=getattr(command_source, "key_callback", None))
    pacer = LoopPacer(dt_ctrl, realtime=realtime)
    cmd_filter = None
    if command_source is not None:
        from ..teleop.command_source import CommandFilter, load_limits

        cmd_filter = CommandFilter(load_limits(), dt_ctrl)
        if hasattr(planner, "set_command"):
            planner.set_command(0.0, 0.0, 0.0)
    quit_req = False
    for k in range(planner.n_t):
        if command_source is not None:
            raw = command_source.poll(k * dt_ctrl)
            if raw.quit:
                quit_req = True
                break
            v = cmd_filter(raw)
            if hasattr(planner, "set_command"):
                planner.set_command(*v)
            if raw.push is not None:
                env.apply_impulse_base(np.asarray(raw.push, dtype=np.float64))
                if verbose:
                    print(f"外乱（操作）: 力積 {np.asarray(raw.push)} N*s を t = {k * dt_ctrl:.2f} s に印加")
        if on_step is not None:
            on_step(planner, st, k)
        u = ctrl.compute(st, k)
        ci = ctrl.last_info
        env.apply(u)
        if scenario is not None and getattr(scenario, "kind", "none") == "push" and not push_done \
                and float(ci.ref.t) >= float(scenario.t_push):
            env.apply_impulse_base(np.asarray(scenario.impulse, dtype=np.float64))
            push_done = True
            if verbose:
                print(f"外乱: 力積 {np.asarray(scenario.impulse)} N*s を t = {ci.ref.t:.3f} s に印加")
        env.step(decim)
        st = env.observe()
        t = float(ci.ref.t)

        cop = env.contact_cop()
        q_now = model.make_q(st.p_WB, st.quat_WB, st.q)
        R_WF = np.array([model.foot_rotation(q_now, i) for i in range(nf)])
        poly = foot_support_polygon(st.p_WF, R_WF, spec.half_length, spec.half_width, st.contact)
        inside = bool(poly.shape[0] >= 3 and np.all(np.isfinite(cop)) and point_in_polygon(cop[0:2], poly))
        rpy = quat_to_euler_zyx(st.quat_WB)[::-1]
        # 足裏 F/T から全体 CoP（法線力で重み付け）
        wf = np.asarray(st.wrench_feet)
        cops = [wrench_to_cop(wf[i], st.p_WF[i], R_WF[i]) for i in range(nf)]
        fz = np.maximum(wf[:, 2], 0.0)
        if fz.sum() > 1.0:
            cop_ft = sum(fz[i] * np.nan_to_num(cops[i][0:2]) for i in range(nf)) / fz.sum()
        else:
            cop_ft = np.full(2, np.nan)

        rec["t"].append(t)
        rec["c_ref"].append(ci.ref.c_ref[0:2].copy())
        rec["c_meas"].append(st.c_W.copy())
        rec["cop"].append(cop)
        rec["zmp_ref"].append(ci.ref.cop_sched.copy())
        rec["p_WF"].append(st.p_WF.copy())
        rec["con_sched"].append(ci.ref.contact.copy())
        rec["con_meas"].append(st.contact.copy())
        rec["rpy"].append(rpy)
        rec["inside"].append(inside)
        rec["xi_hat"].append(ci.xi_hat.copy())
        rec["xi_ref"].append(ci.ref.xi_ref.copy())
        rec["cop_cmd"].append(ci.cop_cmd.copy())
        rec["cop_ft"].append(cop_ft)
        rec["tau"].append(u.tau.copy())
        rec["tau_meas"].append(np.asarray(st.tau_meas).copy())
        rec["w_cmd"].append(np.asarray(wbc.last_wrench).copy())
        rec["w_meas"].append(wf.copy())
        rec["load"].append(ci.load.copy())
        rec["solve"].append(float(wbc.last_solve_time))
        rec["status"].append(wbc.last_status in ("solved", "solved inaccurate"))
        if cfg.n_arm > 0:
            rec["w_hands"].append(np.asarray(st.wrench_hands).copy())
            rec["q_arm_err"].append(np.asarray(st.q)[n_lj:] - np.asarray(ci.ref.q_posture)[n_lj:])
            rec["p_hands"].append(np.array([model.hand_position(q_now, a) for a in range(cfg.n_arm)]))
            if hand_task is not None:
                rec["f_hand_cmd"].append(np.asarray(hand_task.last_F).copy())
        if box_obj is not None:
            rec["box"].append(np.asarray(st.box_pose).copy())

        if abs(rpy[0]) > np.radians(40.0) or abs(rpy[1]) > np.radians(40.0):
            fell = True
            if verbose:
                print(f"転倒 t = {t:.3f} s (roll {np.degrees(rpy[0]):+.1f} deg, pitch {np.degrees(rpy[1]):+.1f} deg)")
            break
        if view.enabled and k % max(1, int(view_every)) == 0:
            view.sync()
            if not view.is_running():
                quit_req = True
                break
        if realtime:
            pacer.sleep()
    view.close()
    if command_source is not None:
        command_source.close()
    if verbose and quit_req:
        print(f"終了要求（ビューアを閉じた / quit）t = {k * dt_ctrl:.2f} s")

    if sch is not None:
        n_step_done = int(sum(1 for i in range(sch.segment_at(float(rec["t"][-1])) + 1)
                              if sch.seg_kind[i] is SegKind.SS))
    else:
        n_step_done = int(planner.steps_done)
    res = result_cls(
        t=np.asarray(rec["t"]),
        c_ref=np.asarray(rec["c_ref"]),
        c_meas=np.asarray(rec["c_meas"]),
        cop_meas=np.asarray(rec["cop"]),
        zmp_ref=np.asarray(rec["zmp_ref"]),
        p_WF=np.asarray(rec["p_WF"]),
        contact_sched=np.asarray(rec["con_sched"]),
        contact_meas=np.asarray(rec["con_meas"]),
        rpy=np.asarray(rec["rpy"]),
        ik_res=np.zeros(len(rec["t"])),
        cop_inside=np.asarray(rec["inside"]),
        fell=fell,
        n_step_done=n_step_done,
    )
    res.summary = summarize_fn(res, params)
    if realtime:
        stt = pacer.stats()
        if stt is not None and stt.n > 0:
            res.summary["周期 平均 / 最大 [ms]"] = [round(float(stt.dt_mean) * 1e3, 3), round(float(stt.dt_max) * 1e3, 3)]
            res.summary["周期超過率 [%]"] = round(100.0 * float(stt.overrun) / max(int(stt.n), 1), 2)
    log = TorqueWalkLog(
        xi_hat=np.asarray(rec["xi_hat"]), xi_ref=np.asarray(rec["xi_ref"]),
        cop_cmd=np.asarray(rec["cop_cmd"]), cop_ft=np.asarray(rec["cop_ft"]),
        tau=np.asarray(rec["tau"]), tau_meas=np.asarray(rec["tau_meas"]),
        wrench_cmd=np.asarray(rec["w_cmd"]), wrench_meas=np.asarray(rec["w_meas"]),
        load=np.asarray(rec["load"]), solve_time=np.asarray(rec["solve"]),
        status=np.asarray(rec["status"], dtype=bool),
    )
    if rec["w_hands"]:
        log.wrench_hands = np.asarray(rec["w_hands"])
        log.q_arm_err = np.asarray(rec["q_arm_err"])
        log.p_hands = np.asarray(rec["p_hands"])
    if rec["f_hand_cmd"]:
        log.f_hand_cmd = np.asarray(rec["f_hand_cmd"])
    if rec["box"]:
        log.box_pose = np.asarray(rec["box"])
    e_xi = log.xi_hat - log.xi_ref
    res.summary["DCM 誤差 RMS [mm]"] = round(1000.0 * float(np.sqrt(np.mean(np.sum(e_xi ** 2, axis=1)))), 2)
    res.summary["DCM 誤差 max [mm]"] = round(1000.0 * float(np.max(np.linalg.norm(e_xi, axis=1))), 2)
    d_cop = np.linalg.norm(np.nan_to_num(log.cop_ft) - log.cop_cmd, axis=1)
    res.summary["CoP 指令-実測(F/T) RMS [mm]"] = round(1000.0 * float(np.sqrt(np.mean(d_cop ** 2))), 2)
    res.summary["QP 解時間 p95 [ms]"] = round(1000.0 * float(np.percentile(log.solve_time, 95)), 3)
    res.summary["QP 失敗率 [%]"] = round(100.0 * float(1.0 - np.mean(log.status)), 2)
    res.summary["トルク飽和率 [%]"] = round(100.0 * float(np.mean(np.abs(log.tau) >= 0.999 * cfg.tau_max)), 2)
    res.summary["トルク計測-指令 RMS [N*m]"] = round(float(np.sqrt(np.mean((log.tau_meas - log.tau) ** 2))), 3)
    if hasattr(planner, "n_timing_adapt"):
        res.summary["時間調整の回数 [-]"] = int(planner.n_timing_adapt)
    # 滑らかさ（方策の推論周期・残差の比較に使う）
    if len(res.t) > 2:
        dts = float(np.mean(np.diff(res.t)))
        dtau = np.diff(log.tau, axis=0) / dts
        res.summary["トルク変化率 RMS [N*m/s]"] = round(float(np.sqrt(np.mean(dtau ** 2))), 1)
        ddc = np.diff(res.c_meas, n=2, axis=0) / dts ** 2
        res.summary["重心加速度 RMS [m/s^2]"] = round(float(np.sqrt(np.mean(np.sum(ddc[:, 0:2] ** 2, axis=1)))), 3)
        fz = np.asarray(log.wrench_meas)[:, :, 2] if log.wrench_meas.ndim == 3 else np.zeros((1, 1))
        res.summary["法線力 max [N]"] = round(float(np.max(fz)), 1)
    # 箱押し用（腕・箱があるときだけ足す。既存の要約は変えない）
    if log.q_arm_err.size:
        res.summary["腕姿勢誤差 RMS / max [deg]"] = [
            round(float(np.degrees(np.sqrt(np.mean(log.q_arm_err ** 2)))), 2),
            round(float(np.degrees(np.max(np.abs(log.q_arm_err)))), 2)]
        f_h = np.linalg.norm(log.wrench_hands[:, :, 0:3], axis=2)
        res.summary["手の接触力 max 左/右 [N]"] = [round(float(v), 1) for v in f_h.max(axis=0)]
        res.summary["腕トルク飽和率 [%]"] = round(
            100.0 * float(np.mean(np.abs(log.tau[:, n_lj:]) >= 0.999 * cfg.tau_max[n_lj:])), 2)
    if log.box_pose.size:
        res.summary["箱の移動 [m]"] = round(float(np.linalg.norm(log.box_pose[-1, 0:2] - log.box_pose[0, 0:2])), 4)
    res.extra = log
    if verbose:
        for k_, v_ in res.summary.items():
            print(f"  {k_}: {v_}")
    return res
