"""! @file run_biped.py
@brief 二足（G1）を MuJoCo 上で歩かせる実行エントリ。

2D で作った重心軌道を、3D の機体に載せる。mapper = pd の構成は次のとおり。

```
BipedGaitParams -> make_biped_schedule   足配置と区分一定 CoP 参照
                -> DcmPlanner.plan       x, y 独立に重心軌道（閉形式）
                -> solve_whole_body_ik   重心・足先・胴体姿勢 -> 関節角
                -> ControlOutput         関節 PD で追従（kp 大）
```

**状態推定はシミュレータの真値（Cheater）である。**
pd では安定化器（CoM-ZMP フィードバックと足首アドミッタンス）を入れない。
LIPM の仮定（重心高さ一定・角運動量ゼロ）が機体で破れることによる ZMP のずれを
**そのまま観測して記録する**ための構成で、ずれを吸収するのはトルク制御（ihmc / dlr）側である。

使い方:

```bash
python -m lococtrl.cli.run_biped --mode walk --n-step 10
python -m lococtrl.cli.run_biped --mode stand --duration 5
```
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np

from ..bench.metrics import foot_support_polygon, point_in_polygon, rms
from ..common.config import load_robot_config
from ..common.frames import quat_to_euler_zyx
from ..common.types import ControlOutput
from ..control.dcm_feedback import clamp_cop, dcm_feedback_cop
from ..control.whole_body_ik import WbikTarget, WbikWeights, solve_whole_body_ik
from ..gait.biped_schedule import BipedGaitParams, SegKind, make_biped_schedule
from ..model_dynamics.robot_model import create_robot_model
from ..planner.dcm import DcmPlanner
from ..sim.mujoco_env import MujocoEnv

__all__ = ["BipedWalkResult", "gravity_feedforward", "run_biped_walk", "main"]

## 足裏の半長 [m]。configs/robot_g1.yaml の MJCF 生成値と一致させること
## （MJCF 生成時の foot_box_size）。
_FOOT_HALF_LENGTH = 0.085
_FOOT_HALF_WIDTH = 0.0275


def gravity_feedforward(
    model,
    q: np.ndarray,
    c_ddot_xy: np.ndarray,
    cop_ref: np.ndarray,
    contact: np.ndarray,
    p_WF: np.ndarray,
    gravity: float = 9.81,
) -> np.ndarray:
    """! @brief 重力と目標重心加速度を支える関節トルクを計算する（フィードフォワード）。

    @param model     RobotModelBase [-]
    @param q         一般化座標 shape (nq,) [m, -, rad]
    @param c_ddot_xy 目標重心加速度の水平成分 shape (2,) [m/s^2]
    @param cop_ref   参照 CoP shape (2,) [m]
    @param contact   立脚フラグ shape (nf,) [bool]
    @param p_WF      足先位置 {W} shape (nf,3) [m]
    @param gravity   重力加速度 [m/s^2]
    @return tau_ff shape (n,) [N*m]

    @note 浮遊ベース動力学 `M ddq + h = S^T tau + J_c^T f` で `ddq = 0` と置くと
          `tau = h[6:] - (J_c^T f)[6:]` である。**`h[6:]` だけを出すのは誤り**で、
          接触力が支えているぶんを二重に出してしまう。

          必要な合力は `f_sum = m (c_ddot + g)`。これを立脚足へ分配するとき、
          参照 CoP に近い足へ大きく配る（両脚支持では CoP の位置で線形補間、
          単脚支持ではその足に全部）。こうすると合力とモーメントの両方が
          近似的に満たされる。

          **これはトルク制御の安定化器ではない。** 実測の ZMP を見て帰還するのではなく、
          参照値だけから前向きに計算する開ループ項である。片足支持中の重力
          モーメントは G1 でおよそ 33 N*m あり、kp = 300 N*m/rad の関節 PD だけでは
          6 度ほどの追従誤差が残って横に倒れる。それを消すために要る。
    """
    m = model.total_mass
    idx = np.flatnonzero(np.asarray(contact, dtype=bool))
    n = model.n_joint
    if idx.size == 0:
        return np.zeros(n, dtype=np.float64)

    f_sum = m * np.array(
        [float(c_ddot_xy[0]), float(c_ddot_xy[1]), gravity], dtype=np.float64
    )

    # 参照 CoP から各立脚足への配分比を決める
    if idx.size == 1:
        wts = np.array([1.0])
    else:
        d = np.linalg.norm(
            np.asarray(p_WF)[idx, 0:2] - np.asarray(cop_ref).reshape(1, 2), axis=1
        )
        # CoP に近い足に大きく配る。両足が等距離なら半分ずつ。
        inv = 1.0 / np.maximum(d, 1.0e-6)
        wts = inv / inv.sum()

    v0 = model.make_v(np.zeros(3), np.zeros(3), np.zeros(n))
    tau = model.nle(q, v0)[6:].copy()
    for a, leg in enumerate(idx):
        J = model.jacobian_foot(q, int(leg))  # (3, nv)
        tau -= (J.T @ (wts[a] * f_sum))[6:]
    return tau


@dataclass
class BipedWalkResult:
    """! @brief 歩行試験の結果と評価指標の実測値。

    @param t            時刻 shape (T,) [s]
    @param c_ref        重心参照（水平）shape (T,2) [m]
    @param c_meas       重心実測 shape (T,3) [m]
    @param cop_meas     実測 CoP shape (T,3) [m]
    @param zmp_ref      参照 CoP（スケジュールの区分一定値）shape (T,2) [m]
    @param p_WF         足先位置 shape (T,nf,3) [m]
    @param contact_sched 予定接触 shape (T,nf) [bool]
    @param contact_meas 実測接触 shape (T,nf) [bool]
    @param rpy          ロール・ピッチ・ヨー shape (T,3) [rad]
    @param ik_res       IK の足先位置残差 shape (T,) [m]
    @param cop_inside   CoP が支持多角形内か shape (T,) [bool]
    @param fell         転倒したか [-]
    @param n_step_done  歩き切った歩数 [-]
    """

    t: np.ndarray
    c_ref: np.ndarray
    c_meas: np.ndarray
    cop_meas: np.ndarray
    zmp_ref: np.ndarray
    p_WF: np.ndarray
    contact_sched: np.ndarray
    contact_meas: np.ndarray
    rpy: np.ndarray
    ik_res: np.ndarray
    cop_inside: np.ndarray
    fell: bool
    n_step_done: int
    summary: dict = field(default_factory=dict)
    ## トルク制御固有のログ（`cli.biped_torque_run.TorqueWalkLog`）。pd では None
    extra: object = None


def _biped_params(overrides: dict | None = None) -> BipedGaitParams:
    """! @brief configs/biped.yaml の walk_3d 節から歩容パラメータを作る（`gait.biped_schedule.gait_params_from_config` の別名）。
    @param overrides 上書きする値 [-]
    @return BipedGaitParams
    """
    from ..gait.biped_schedule import gait_params_from_config

    return gait_params_from_config(overrides)


def run_biped_walk(
    robot: str = "robot_g1.yaml",
    gait_overrides: dict | None = None,
    dt_ctrl: float = 0.002,
    kp_scale: float = 1.0,
    ik_iter: int = 8,
    use_ff: bool = True,
    k_dcm: float = 0.0,
    verbose: bool = True,
    viewer: bool = False,
    realtime: bool = False,
    view_every: int = 5,
) -> BipedWalkResult:
    """! @brief 二足歩行を実行し、評価指標を測る。

    @param robot          ロボット設定ファイル名 [-]
    @param gait_overrides 歩容パラメータの上書き [-]
    @param dt_ctrl        制御周期 [s]
    @param kp_scale       関節 P ゲインの倍率 [-]
    @param ik_iter        1 制御周期あたりの IK 反復回数 [-]
    @param use_ff         重力・慣性のフィードフォワードを使うか [-]
    @param k_dcm          DCM フィードバックのゲイン [-]。0 で開ループ。
                          正にすると `dcm_feedback_cop` で CoP 指令を修正する
    @param verbose        進捗を表示するか [-]
    @param viewer         MuJoCo ビューアを開く [-]
    @param realtime       実時間で回す [-]
    @param view_every     描画の間引き [-]
    @return BipedWalkResult
    @note IK は前回解を初期値にするので、1 周期あたり数回の反復で足りる。
          毎周期ゼロから解くと計算量が増えるだけでなく、解が不連続に飛ぶ。
    """
    cfg = load_robot_config(robot)
    model = create_robot_model(cfg)
    env = MujocoEnv(cfg)
    params = _biped_params(gait_overrides)
    # 公称姿勢での足先位置をスケジュールの初期足配置にする。
    # ここを原点にすると IK が足を引き戻して重心が振られ、立位でも転倒する。
    _q_nom0 = model.make_q(
        np.array([0.0, 0.0, cfg.h_nominal]), np.array([1.0, 0.0, 0.0, 0.0]), cfg.q_nominal
    )
    foot_init = model.fk(_q_nom0).copy()
    foot_init[:, 2] = params.foot_z
    sch = make_biped_schedule(params, foot_init=foot_init)

    # --- 重心軌道（x, y 独立に閉形式）---
    # z_c は公称姿勢の実測値を使う（h_nominal は pelvis 高さなので別物）。
    q_nom = model.make_q(
        np.array([0.0, 0.0, cfg.h_nominal]), np.array([1.0, 0.0, 0.0, 0.0]), cfg.q_nominal
    )
    z_c = float(model.com(q_nom)[2])
    planner = DcmPlanner(z_c=z_c)
    plan_x = planner.plan(sch.seg_cop[:, 0], sch.seg_dur, dt_ctrl, c0=float(model.com(q_nom)[0]))
    plan_y = planner.plan(sch.seg_cop[:, 1], sch.seg_dur, dt_ctrl, c0=0.0)
    n_t = min(plan_x.t.size, plan_y.t.size)

    if verbose:
        print(f"z_c = {z_c:.4f} m, omega = {planner.omega:.4f} rad/s")
        print(f"歩数 {params.n_step}, 周期 {params.period:.3f} s, "
              f"平均速度 {params.mean_speed:.3f} m/s, 全体 {sch.total_time:.2f} s")

    # --- 初期化 ---
    st = env.reset(q_joint=cfg.q_nominal, p_WB=np.array([0.0, 0.0, cfg.h_nominal]))
    q_ik = model.make_q(st.p_WB, st.quat_WB, st.q)
    R_flat = np.array([np.eye(3), np.eye(3)])  # 足裏は床に平ら
    w_ik = WbikWeights(max_iter=int(ik_iter), tol=1.0e-8)
    kp = np.array([300.0, 300.0, 200.0, 300.0, 200.0, 200.0] * 2) * kp_scale
    kd = np.array([10.0, 10.0, 5.0, 10.0, 5.0, 5.0] * 2) * np.sqrt(kp_scale)

    nf = cfg.n_leg
    rec: dict[str, list] = {k: [] for k in (
        "t", "c_ref", "c_meas", "cop", "zmp_ref", "p_WF",
        "con_sched", "con_meas", "rpy", "ik_res", "inside")}
    fell = False
    decim = max(1, int(round(dt_ctrl / env.dt)))

    from ..common.timing import LoopPacer
    from ..sim.viewer import Viewer

    view = Viewer(env, enable=viewer)
    pacer = LoopPacer(dt_ctrl, realtime=realtime)
    for k in range(n_t):
        t = float(plan_x.t[k])
        gs, p_foot_des, v_foot_des, contact_sched = sch.state_at(t)
        c_xy_des = np.array([plan_x.c[k], plan_y.c[k]])

        tgt = WbikTarget(
            p_WF_des=p_foot_des,
            R_WF_des=R_flat,
            c_xy_des=c_xy_des,
            quat_WB_des=np.array([1.0, 0.0, 0.0, 0.0]),
            q_posture=cfg.q_nominal,
        )
        r = solve_whole_body_ik(model, tgt, q_ik, w_ik)
        q_ik = r.q

        # CoP 指令。k_dcm = 0 なら参照そのまま（開ループ）。
        # k_dcm > 0 のときは DCM フィードバックで LIPM の不安定極を移す。
        cop_sched = sch.seg_cop[sch.segment_at(t)]
        if k_dcm > 0.0:
            xi_meas = st.c_W[0:2] + st.dc_W[0:2] / planner.omega
            xi_ref = np.array([plan_x.xi[k], plan_y.xi[k]])
            xi_ref_dot = np.array([plan_x.xi_dot[k], plan_y.xi_dot[k]])
            cop_cmd = dcm_feedback_cop(
                xi_meas, xi_ref, xi_ref_dot, planner.omega, k_dcm
            )
            # 立脚足の足裏矩形（和集合の外接矩形）で clamp する。
            act = np.flatnonzero(contact_sched)
            lo = p_foot_des[act, 0:2].min(axis=0) - np.array(
                [_FOOT_HALF_LENGTH, _FOOT_HALF_WIDTH])
            hi = p_foot_des[act, 0:2].max(axis=0) + np.array(
                [_FOOT_HALF_LENGTH, _FOOT_HALF_WIDTH])
            cop_cmd, _ = clamp_cop(cop_cmd, 0.5 * (lo + hi), 0.5 * (hi - lo))
            c_ddot_xy = (planner.omega ** 2) * (st.c_W[0:2] - cop_cmd)
        else:
            cop_cmd = cop_sched
            c_ddot_xy = (planner.omega ** 2) * (c_xy_des - cop_sched)

        # 重力と目標重心加速度のフィードフォワード。
        tau_ff = gravity_feedforward(
            model, q_ik, c_ddot_xy, cop_cmd, contact_sched, p_foot_des,
        ) if use_ff else np.zeros(cfg.n_joint)

        u = ControlOutput(
            tau=tau_ff,
            q_des=q_ik[7:].copy(),
            dq_des=np.zeros(cfg.n_joint),
            kp=kp,
            kd=kd,
        )
        env.apply(u)
        env.step(decim)
        st = env.observe()

        cop = env.contact_cop()
        R_WF = np.array([model.foot_rotation(
            model.make_q(st.p_WB, st.quat_WB, st.q), i) for i in range(nf)])
        poly = foot_support_polygon(
            st.p_WF, R_WF, _FOOT_HALF_LENGTH, _FOOT_HALF_WIDTH, st.contact
        )
        inside = bool(
            poly.shape[0] >= 3 and np.all(np.isfinite(cop))
            and point_in_polygon(cop[0:2], poly)
        )
        rpy = quat_to_euler_zyx(st.quat_WB)[::-1]  # (roll, pitch, yaw)

        rec["t"].append(t)
        rec["c_ref"].append(c_xy_des)
        rec["c_meas"].append(st.c_W.copy())
        rec["cop"].append(cop)
        rec["zmp_ref"].append(sch.seg_cop[sch.segment_at(t)].copy())
        rec["p_WF"].append(st.p_WF.copy())
        rec["con_sched"].append(contact_sched.copy())
        rec["con_meas"].append(st.contact.copy())
        rec["rpy"].append(rpy)
        rec["ik_res"].append(r.res_foot_pos)
        rec["inside"].append(inside)

        if abs(rpy[0]) > np.radians(40.0) or abs(rpy[1]) > np.radians(40.0):
            fell = True
            if verbose:
                print(f"転倒 t = {t:.3f} s "
                      f"(roll {np.degrees(rpy[0]):+.1f} deg, pitch {np.degrees(rpy[1]):+.1f} deg)")
            break
        if view.enabled and k % max(1, int(view_every)) == 0:
            view.sync()
            if not view.is_running():
                break
        if realtime:
            pacer.sleep()
    view.close()

    res = BipedWalkResult(
        t=np.asarray(rec["t"]),
        c_ref=np.asarray(rec["c_ref"]),
        c_meas=np.asarray(rec["c_meas"]),
        cop_meas=np.asarray(rec["cop"]),
        zmp_ref=np.asarray(rec["zmp_ref"]),
        p_WF=np.asarray(rec["p_WF"]),
        contact_sched=np.asarray(rec["con_sched"]),
        contact_meas=np.asarray(rec["con_meas"]),
        rpy=np.asarray(rec["rpy"]),
        ik_res=np.asarray(rec["ik_res"]),
        cop_inside=np.asarray(rec["inside"]),
        fell=fell,
        n_step_done=int(
            sum(1 for i in range(sch.segment_at(float(rec["t"][-1])) + 1)
                if sch.seg_kind[i] is SegKind.SS)
        ),
    )
    res.summary = summarize(res, params)
    if verbose:
        for k, v in res.summary.items():
            print(f"  {k}: {v}")
    return res


def summarize(res: BipedWalkResult, params: BipedGaitParams) -> dict:
    """! @brief 評価指標の実測値をまとめる。
    @param res    実行結果 [-]
    @param params 歩容パラメータ [-]
    @return 指標の辞書
    @note 静止区間（t_start 以前と歩き終わり以降）を除いた歩行中だけで測る。
          静止中は CoP が両足中央に落ち着くので、混ぜると内包率が甘くなる。
    """
    t = res.t
    # 転倒で打ち切られた場合は「歩行開始から打ち切りまで」を見る。
    # 固定で t_end を引くと評価区間が消えて指標が意味を失う。
    t_hi = t[-1] if res.fell else (t[-1] - params.t_end + 1.0e-9)
    walk = (t >= params.t_start) & (t <= t_hi)
    if not np.any(walk):
        walk = np.ones_like(t, dtype=bool)

    e_com = res.c_meas[:, 0:2] - res.c_ref
    zmp_err = res.cop_meas[:, 0:2] - res.zmp_ref
    fin = np.all(np.isfinite(zmp_err), axis=1)

    return {
        "転倒": res.fell,
        "歩数": res.n_step_done,
        "重心追従誤差 RMS [mm]": round(float(rms(e_com[walk])) * 1000, 2),
        "重心追従誤差 max [mm]": round(float(np.abs(e_com[walk]).max()) * 1000, 2),
        "CoP 内包率 [%]": round(float(res.cop_inside[walk].mean()) * 100, 2),
        "IK 残差 max [m]": float(f"{res.ik_res.max():.3e}"),
        "ロール最大 [deg]": round(float(np.degrees(np.abs(res.rpy[:, 0]).max())), 2),
        "ピッチ最大 [deg]": round(float(np.degrees(np.abs(res.rpy[:, 1]).max())), 2),
        "前進距離 [m]": round(float(res.c_meas[-1, 0] - res.c_meas[0, 0]), 4),
        "ZMP ずれ RMS [mm]": round(float(rms(zmp_err[walk & fin])) * 1000, 2),
        "ZMP ずれ max [mm]": round(float(np.abs(zmp_err[walk & fin]).max()) * 1000, 2),
        "接触一致率 [%]": round(
            float((res.contact_sched[walk] == res.contact_meas[walk]).mean()) * 100, 2
        ),
    }


def main(argv: list[str] | None = None) -> int:
    """! @brief CLI 入口。
    @param argv コマンドライン引数 [-]
    @return 終了コード [-]
    """
    ap = argparse.ArgumentParser(description="二足（G1）の歩行実行")
    ap.add_argument("--mode", choices=("walk", "stand"), default="walk")
    ap.add_argument("--robot", default="robot_g1.yaml")
    ap.add_argument("--n-step", type=int, default=None, help="歩数 [-]")
    ap.add_argument("--step-length", type=float, default=None, help="歩幅 [m]")
    ap.add_argument("--t-ss", type=float, default=None, help="単脚支持期 [s]")
    ap.add_argument("--t-ds", type=float, default=None, help="両脚支持期 [s]")
    ap.add_argument("--kp-scale", type=float, default=1.0)
    ap.add_argument("--ik-iter", type=int, default=8)
    ap.add_argument("--no-ff", action="store_true",
                    help="重力・慣性のフィードフォワードを切る（比較用）")
    ap.add_argument("--k-dcm", type=float, default=None,
                    help="DCM フィードバックのゲイン [-]。pd では既定 0（開ループ）、"
                         "ihmc/dlr/rl では既定 biped.yaml の stabilizer.k_dcm")
    # 下層の制御を選ぶ。pd = IK + 関節 PD、ihmc = 運動量ベース全身 QP、
    # dlr = 受動性ベース全身制御、rl = ベース（--rl-base）+ 学習残差トルク方策
    ap.add_argument("--mapper", choices=("pd", "ihmc", "dlr", "rl"), default="pd",
                    help="下層の制御: pd | ihmc | dlr | rl")
    ap.add_argument("--rl-base", choices=("ihmc", "dlr"), default="ihmc",
                    help="rl のベース全身制御器")
    ap.add_argument("--policy", default=None,
                    help="rl の学習済み方策ディレクトリ（params.npz + meta.json）。無ければ残差ゼロ")
    ap.add_argument("--log", default=None, help="結果を npz に保存するパス")
    # 表示・操作（条件ではないので --config と併用できる）
    ap.add_argument("--viewer", action="store_true", help="MuJoCo ビューアを開く")
    ap.add_argument("--realtime", action="store_true", help="実時間で回す（--viewer と併用して見る）")
    ap.add_argument("--source", choices=("keyboard", "gamepad", "udp", "scripted"), default=None,
                    help="テレオペ入力。与えると計画器を online + endless にし、速度指令で操作する（mapper は ihmc | dlr）")
    ap.add_argument("--duration", type=float, default=None, help="--source のときの実行時間 [s]（既定 600）")
    ap.add_argument("--udp-port", type=int, default=None, help="--source udp のポート")
    ap.add_argument("--dt-ctrl", type=float, default=None, help="制御周期 [s]（実時間で周期を超えるとき 0.004 など）")
    # 実験条件は設定ファイルに書き、CLI の条件フラグと併用しない。
    ap.add_argument("--config", default=None,
                    help="実験設定ファイル（.yaml / .json）。configs/experiments/ を参照。条件フラグとは併用不可")
    ap.add_argument("--dump-config", default=None, metavar="PATH",
                    help="今の条件（CLI フラグ込み）を設定ファイルの雛形として書き出して終了する")
    ap.add_argument("--config-print", choices=("full", "changed", "compact", "none"), default="changed",
                    help="実行前に設定を表示する: full = 全項目 / changed = 既定値から変えた項目 / compact = 1 行 / none")
    a = ap.parse_args(argv)  # None なら sys.argv（従来は引数を無視していた）

    if a.config is not None or a.dump_config is not None:
        return _main_with_config(ap, a)

    if a.source is not None and a.mapper == "pd":
        if a.mapper == ap.get_default("mapper"):
            a.mapper = "dlr"  # テレオペの既定は dlr（QP 0.3 ms で実時間に収まる）
            print("[run_biped] --source: mapper を dlr にします（pd は固定列専用。ihmc は --mapper ihmc）")
        else:
            ap.error("--source（テレオペ）は --mapper ihmc | dlr と組み合わせる")
    ov: dict = {"n_step": a.n_step, "step_length": a.step_length,
                "t_ss": a.t_ss, "t_ds": a.t_ds}
    if a.mode == "stand":
        ov["n_step"] = 0
    if a.mapper == "pd":
        res = run_biped_walk(robot=a.robot, gait_overrides=ov, kp_scale=a.kp_scale,
                             ik_iter=a.ik_iter, use_ff=not a.no_ff,
                             k_dcm=0.0 if a.k_dcm is None else a.k_dcm)
    else:
        from .biped_torque_run import run_biped_torque

        pk, pp, src = _teleop_setup(a)
        res = run_biped_torque(robot=a.robot, gait_overrides=ov, mapper=a.mapper,
                               k_dcm=a.k_dcm, policy_dir=a.policy, rl_base=a.rl_base,
                               planner_kind=pk, planner_params=pp, command_source=src,
                               viewer=a.viewer, realtime=a.realtime or src is not None,
                               **({} if a.dt_ctrl is None else {"dt_ctrl": a.dt_ctrl}))
    if a.log:
        save_result(res, a.log)
    return 0


def _teleop_setup(a: argparse.Namespace, online_params=None):
    """! @brief --source から計画器の種類・パラメータ・入力源を作る。
    @param a             解析済み引数 [-]
    @param online_params 設定ファイル由来の OnlinePlannerParams（None なら既定）[-]
    @return (planner_kind, planner_params, command_source)。--source 無しなら ("schedule" | 設定どおり, online_params, None)
    """
    if a.source is None:
        return ("schedule" if online_params is None else "online"), online_params, None
    if a.mapper == "pd":
        raise SystemExit("--source（テレオペ）は --mapper ihmc | dlr と組み合わせる（pd は固定列専用）")
    import dataclasses

    from ..planner.biped_online import OnlinePlannerParams
    from ..teleop.command_source import load_limits, load_teleop_yaml, make_source

    pp = OnlinePlannerParams() if online_params is None else online_params
    pp = dataclasses.replace(pp, source="velocity", endless=True,
                             duration=600.0 if a.duration is None else float(a.duration))
    tcfg = load_teleop_yaml()
    scfg = dict(tcfg.get(a.source) or {})
    if a.udp_port is not None:
        scfg["port"] = a.udp_port
    src = make_source(a.source, load_limits(), scfg)
    return "online", pp, src


def _main_with_config(ap: argparse.ArgumentParser, a: argparse.Namespace) -> int:
    """! @brief 設定ファイル経由の実行（--config / --dump-config）。
    @param ap パーサ（既定値の照合に使う）[-]
    @param a  解析済み引数 [-]
    @return 終了コード [-]
    @note --config のときは条件を決めるフラグ（--mapper など）を受け付けない。条件は 1 か所（ファイル）に置く。
          --dump-config のときは逆に、CLI フラグで作った条件をファイルに書く（雛形作り）。
    """
    from .biped_experiment import (
        BipedExperimentConfig,
        describe_config,
        dump_experiment_config,
        load_experiment_config,
        run_experiment,
    )

    cond_flags = ("mode", "robot", "n_step", "step_length", "t_ss", "t_ds", "kp_scale", "ik_iter",
                  "no_ff", "k_dcm", "mapper", "rl_base", "policy", "log")
    changed = [f for f in cond_flags if getattr(a, f) != ap.get_default(f)]

    if a.dump_config is not None:
        cfg = BipedExperimentConfig()
        if a.config is not None:
            cfg = load_experiment_config(a.config)
        # 明示的に渡した CLI フラグだけを雛形に反映する（既定のままの項目はファイルの既定を保つ）
        if "robot" in changed:
            cfg.robot = a.robot
        if a.mode == "stand":
            cfg.gait.n_step = 0
        for k in ("n_step", "step_length", "t_ss", "t_ds"):
            if getattr(a, k) is not None:
                setattr(cfg.gait, k, getattr(a, k))
        if "mapper" in changed:
            cfg.mapper.kind = a.rl_base if a.mapper == "rl" else a.mapper
            cfg.policy.kind = "residual" if a.mapper == "rl" else "none"
            cfg.policy.model_dir = a.policy if a.mapper == "rl" else None
        if cfg.mapper.kind == "pd":
            if "kp_scale" in changed:
                cfg.mapper.pd.kp_scale = a.kp_scale
            if "ik_iter" in changed:
                cfg.mapper.pd.ik_iter = a.ik_iter
            if "no_ff" in changed:
                cfg.mapper.pd.use_ff = not a.no_ff
            if a.k_dcm is not None:
                cfg.mapper.pd.k_dcm = a.k_dcm
        elif a.k_dcm is not None:
            cfg.stabilizer.k_dcm = a.k_dcm
        if "log" in changed:
            cfg.output.log = a.log
        out = dump_experiment_config(cfg, a.dump_config)
        print(f"設定の雛形を書きました: {out}")
        return 0

    if changed:
        ap.error(f"--config と条件フラグ {changed} は併用できない（条件はファイルに書く）")
    cfg = load_experiment_config(a.config)
    if a.config_print != "none":
        print(describe_config(cfg, changed_only=(a.config_print == "changed"), compact=(a.config_print == "compact")))
    src = None
    if a.source is not None:
        if cfg.mapper.kind == "pd":
            ap.error("--source（テレオペ）は mapper.kind が ihmc | dlr の設定と組み合わせる")
        a.mapper = cfg.mapper.kind
        _, _, src = _teleop_setup(a)
    if a.dt_ctrl is not None:
        cfg.sim.dt_ctrl = float(a.dt_ctrl)
    run_experiment(cfg, viewer=a.viewer, realtime=a.realtime or src is not None, command_source=src,
                   duration=a.duration)
    return 0


def save_result(res: BipedWalkResult, path: str) -> None:
    """! @brief 結果を npz に保存する（トルク制御の追加ログも含む）。
    @param res  結果 [-]
    @param path 保存先 [-]
    @return なし
    """
    from dataclasses import asdict, is_dataclass

    d = {k: v for k, v in vars(res).items() if isinstance(v, np.ndarray)}
    d["fell"] = np.array(res.fell)
    d["n_step_done"] = np.array(res.n_step_done)
    if res.extra is not None and is_dataclass(res.extra):
        for k, v in asdict(res.extra).items():
            d[f"b5_{k}"] = np.asarray(v)
    np.savez(path, **d)


if __name__ == "__main__":
    raise SystemExit(main())
