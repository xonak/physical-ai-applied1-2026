"""! @file trace_policy.py
@brief 方策を 1 エピソード再生し、出力（行動）と体・箱の時系列を記録する。学習方策と手書きの比較に使う。

方策の指定は `eval_policy` と同じ（upright / schedule / reactive / analytic / 学習済みディレクトリ）。
質量・摩擦・箱の初期位置は全方策で同じにする（比べるのは方策だけ）。

記録（1 本 = 1 条件、`<out>/<label>_<mass>kg.npz`）:
- 20 ms ごと（制御 10 周期ごと）
  - 入力（方策の出力）: f_tgt（方策の出した押付け力）[N]、f_cmd（変化率制限後、手先タスクに入った値）[N]、
    lean_tgt / lean / lean_y（方策の出した CMP オフセット x / 計画器に入った x / y）[m]、z_c_cmd（重心高さ指令）[m]、
    step_res（足配置残差 dx, dy, dt）
  - 状態: f_meas / f_L / f_R（手首 F/T の実測、両手合計 / 左 / 右）[N]、box_dx [m]、v_box [m/s]、
    c_x / c_z（重心）[m]、face_c（箱の面 − 重心、押す向き）[m]、reach（骨盤から手先）[m]、pitch / roll [deg]、
    e_dcm（DCM 誤差）[m]、foot_x（足 L/R の x）[m]、contact（接触 L/R）、tau_sat（関節トルク飽和率）
- 判断ごと: k_dec（制御周期番号）、a（生の行動 [-1,1]）
- 動画用: t_frame、qpos（fps ごと。`render_trace` が描画に使う）
- meta（JSON 文字列）: 方策、質量、摩擦、成功 / 転倒、箱の移動

質量・摩擦・箱の初期位置は全条件で同じ（シード 0、x_offset 0）。比べるのは方策だけ。

```bash
cd armtasks/boxpush
python -m boxpush.cli.trace_policy box_p2_walk_push \\
       --policies ../../models/rl_m2,schedule,reactive --labels rl,schedule,reactive \\
       --masses 10,20,30,40 --mu 0.2 --workers 8 --out ../../outputs/trace
python tools/plot_trace.py ../../outputs/trace                  # 図（条件ごと + 質量ごとの比較）
python -m boxpush.cli.render_trace ../../outputs/trace --workers 8   # 動画（条件ごと）
python tools/make_trace_viewer.py ../../outputs/trace           # 閲覧ページ index.html
```
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def run_one(job: dict) -> dict:
    """! @brief 1 エピソード再生して npz に書く。"""
    import boxpush  # noqa: F401

    from lococtrl.common.frames import quat_to_euler_zyx

    from ..cli.eval_policy import _make_policy
    from ..cli.run_box import load_box_experiment
    from ..env import BoxDomain, BoxPushEnv

    cfg, push = load_box_experiment(Path(job["config"]), robot=job.get("robot"))
    cfg.output.verbose = False
    mass, mu = float(job["mass"]), float(job["mu"])
    pol, spec = _make_policy(job["policy"], None, push, mass, mu)
    env = BoxPushEnv(cfg, spec=spec, domain=BoxDomain(mass=(mass, mass), friction=(mu, mu), x_offset=(0.0, 0.0)),
                     max_time=job["max_time"], vx=job["vx"], t_walk=push.t_start, seed=0, base_policy=getattr(pol, "base", None))
    keys = ("t", "f_tgt", "f_cmd", "lean_tgt", "lean", "lean_y", "z_c_cmd", "step_res", "f_base", "lean_base", "f_meas", "f_L", "f_R", "box_dx", "v_box", "c_x", "c_z",
            "face_c", "reach", "pitch", "roll", "e_dcm", "foot_x", "contact", "tau_sat")
    rec: dict = {k: [] for k in keys}
    dec_k, dec_a = [], []
    fps = float(job.get("fps", 30))
    frames_t, frames_q = [], []
    tau_max = np.asarray(env.rcfg.tau_max)

    def frame(e, k):
        t = k * e.dt_ctrl
        if t + 1e-9 >= len(frames_t) / fps:
            frames_t.append(t)
            frames_q.append(np.array(e.env.data.qpos, dtype=np.float32))
        if k % 10:
            return
        st, ap, pl = e.state, e.applier, e.planner
        bp = np.asarray(st.box_pose)
        wh = np.asarray(st.wrench_hands)
        ph = ap._p_hands(st)
        Rt = pl.rot2().T
        face = float(bp[0]) - ap.box_half_x
        info = e.ctrl.last_info
        rpy = quat_to_euler_zyx(st.quat_WB)[::-1]
        fh = -wh[:, 0] if wh.shape == (2, 6) else np.zeros(2)
        rec["t"].append(t)
        rec["f_tgt"].append(float(ap.target_vals.get("f_push", 0.0)))
        rec["f_cmd"].append(float(ap.last_vals.get("f_push", 0.0)))
        # 前傾を方策が出さない条件（A / B / C、脚の解析則）では「方策の出力」は無いので NaN
        rec["lean_tgt"].append(float(ap.target_vals.get("lean_x", 0.0)) if spec.uses_lean_action else float("nan"))
        rec["lean"].append(float(pl.lean_xy[0]))
        rec["lean_y"].append(float(pl.lean_xy[1]))
        rec["z_c_cmd"].append(float(pl.z_c))
        bv = getattr(e, "base_vals", None) or {}
        rec["f_base"].append(float(bv.get("f_push", np.nan)))
        rec["lean_base"].append(float(bv.get("lean_x", np.nan)))
        rec["step_res"].append(np.array([ap.last_vals.get(k, 0.0) for k in ("step_dx", "step_dy", "step_dt")]))
        rec["f_meas"].append(float(fh.sum()))
        rec["f_L"].append(float(fh[0])); rec["f_R"].append(float(fh[1]))
        rec["box_dx"].append(float(bp[0]) - e.box_x0)
        rec["v_box"].append(float((Rt @ bp[7:9])[0]))
        rec["c_x"].append(float(st.c_W[0])); rec["c_z"].append(float(st.c_W[2]))
        rec["face_c"].append(face - float(st.c_W[0]))
        rec["reach"].append(float(np.mean((Rt @ (ph[:, 0:2] - np.asarray(st.p_WB)[0:2]).T)[0])))
        rec["pitch"].append(float(np.degrees(rpy[1]))); rec["roll"].append(float(np.degrees(rpy[0])))
        rec["e_dcm"].append(float(np.linalg.norm(info.xi_hat - info.ref.xi_ref)) if info is not None else 0.0)
        rec["foot_x"].append(np.asarray(st.p_WF, dtype=np.float64)[:, 0].copy())
        rec["contact"].append(np.asarray(st.contact, dtype=np.float64).copy())
        tm = np.asarray(st.tau_meas, dtype=np.float64)
        rec["tau_sat"].append(float(np.mean(np.abs(tm) >= 0.95 * tau_max)) if tm.size == tau_max.size else 0.0)

    env.frame_cb = frame
    t0 = time.time()
    obs = env.reset()
    done = False
    while not done:
        if hasattr(pol, "set_time"):
            pol.set_time(env.k * env.dt_ctrl)
        a = np.asarray(pol.act(obs), dtype=np.float64)
        dec_k.append(env.k); dec_a.append(a)
        obs, r, done, info = env.step(a)
    out = Path(job["out"]) / f"{job['label']}_{mass:.0f}kg.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {"label": job["label"], "policy": job["policy"], "mass": mass, "mu": mu, "actions": list(spec.actions),
            "decimation": int(spec.decimation), "fps": fps, "max_time": float(job["max_time"]), "vx": job["vx"], "config": Path(job["config"]).stem,
            "robot": (Path(job["robot"]).stem if job.get("robot") else None),
            "box_tipped": bool(info.get("box_tipped", False)),
            "success": bool(info.get("success", False)), "fell": bool(info.get("fell", False)),
            "box_dx": round(float(info["box_dx"]), 3), "t_end": round(float(info["t"]), 2),
            "static_friction_N": round(mu * mass * 9.81, 1), "wall_s": round(time.time() - t0, 1)}
    np.savez_compressed(out, **{k: np.asarray(v) for k, v in rec.items()}, k_dec=np.asarray(dec_k), a=np.asarray(dec_a),
                        t_frame=np.asarray(frames_t), qpos=np.asarray(frames_q), meta=json.dumps(meta, ensure_ascii=False))
    return meta


def main(argv: list[str] | None = None) -> int:
    from ..cli.run_box import resolve_experiment, resolve_robot

    ap = argparse.ArgumentParser(description="方策の再生と時系列の記録（学習方策と手書きの比較）")
    ap.add_argument("config")
    ap.add_argument("--policies", default="../../models/rl_m2,schedule,reactive",
                    help="upright | schedule | reactive | analytic | 学習済みディレクトリ（カンマ区切り）")
    ap.add_argument("--labels", default="rl,schedule,reactive", help="ファイル名・図の名前（policies と同順）")
    ap.add_argument("--masses", default="10,20,30,40")
    ap.add_argument("--fps", type=int, default=30, help="動画用に qpos を保存する間隔 [1/s]")
    ap.add_argument("--mu", type=float, default=0.2)
    ap.add_argument("--vx", type=float, default=0.1)
    ap.add_argument("--max-time", type=float, default=10.0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--only", default=None, help="label_mass の部分だけ回す（例 learned_40,schedule_40）")
    ap.add_argument("--skip-existing", action="store_true", help="npz がある条件は回さない")
    ap.add_argument("--out", default="../../outputs/trace")
    ap.add_argument("--robot", default=None,
                    help="ロボット設定の差し替え（例 robot_g1_box_sphere: 手の衝突を球にした別モデル）。npz の meta に残り render_trace も同じモデルで描く")
    a = ap.parse_args(sys.argv[1:] if argv is None else argv)

    pols = a.policies.split(",")
    labels = a.labels.split(",") if a.labels else pols
    robot = str(resolve_robot(a.robot)) if a.robot else None
    jobs = [{"config": str(resolve_experiment(a.config)), "policy": p, "label": lab, "mass": float(m), "mu": a.mu,
             "vx": a.vx, "max_time": a.max_time, "out": a.out, "fps": a.fps, "robot": robot}
            for p, lab in zip(pols, labels) for m in a.masses.split(",")]
    if a.only:
        keep = set(a.only.split(","))
        jobs = [j for j in jobs if f"{j['label']}_{j['mass']:.0f}" in keep]
    if a.skip_existing:
        jobs = [j for j in jobs if not (Path(a.out) / f"{j['label']}_{j['mass']:.0f}kg.npz").exists()]
    if a.workers > 1:
        import multiprocessing as mp

        with mp.get_context("spawn").Pool(a.workers) as pool:
            for r in pool.imap_unordered(run_one, jobs):
                print(json.dumps(r, ensure_ascii=False), flush=True)
    else:
        for j in jobs:
            print(json.dumps(run_one(j), ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
