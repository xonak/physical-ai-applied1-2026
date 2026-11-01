"""質量 × 前傾 × 重心高さの掃引（押しながら歩く、box_p2_walk_push + 手書きスケジュール）。
1 行 1 条件で ../../outputs/p2_sweep.jsonl に追記。引数は 質量:前傾の倍率:重心高さずらし（例 30:1:0）。
前傾 = 倍率 × F / (m ω²)、F = 1.15 μ m_box g。

    cd armtasks/boxpush && python tools/sweep_p2.py 10:0:0 10:1:0 20:0:0 20:0.5:0 20:1:0
"""
import sys, json, time, numpy as np
sys.path.insert(0, "."); import boxpush
from pathlib import Path
from boxpush.cli.run_box import load_box_experiment, resolve_experiment
from boxpush.push import BoxPushHook
from lococtrl.cli.biped_experiment import run_experiment
OUT = Path(__file__).resolve().parents[3] / "outputs" / "p2_sweep.jsonl"
OUT.parent.mkdir(parents=True, exist_ok=True)
for spec in sys.argv[1:]:
    mass, lean_mult, dz = (float(x) for x in spec.split(":"))
    cfg, push = load_box_experiment(resolve_experiment("box_p2_walk_push"))
    cfg.output.verbose = False
    mu = cfg.scenario.box.friction[0]
    cfg.scenario.box.mass = mass
    f = 1.15 * mu * mass * 9.81
    push.f_push = f
    push.f_max = max(60.0, 0.6 * f)
    m_robot, omega = 33.34, 3.90
    push.lean_x = lean_mult * f / (m_robot * omega ** 2)
    push.dz_com = dz
    t0 = time.time()
    hook = BoxPushHook(push, dt=cfg.sim.dt_ctrl)
    res = run_experiment(cfg, hand_task=hook.hand_task, extra_on_step=hook)
    log = res.extra
    box_dx = float(log.box_pose[-1, 0] - log.box_pose[0, 0])
    i0 = int(np.searchsorted(res.t, 3.5))
    fe = -log.wrench_hands[i0:, :, 0].sum(axis=1)
    tau_max = np.array([88,139,88,139,50,50]*2 + [25,25,25,25,25,5,5]*2, dtype=float)
    sat = float(np.mean(np.abs(log.tau) >= 0.99 * tau_max))
    r = {"mass": mass, "mu": mu, "f_push": round(f, 1), "lean_mult": lean_mult, "lean_x": round(push.lean_x, 3), "dz_com": dz,
         "fell": bool(res.fell), "t_end": round(float(res.t[-1]), 2), "box_dx": round(box_dx, 3),
         "f_env_mean": round(float(fe.mean()), 1) if fe.size else 0.0, "com_z_end": round(float(res.c_meas[-1, 2]), 3),
         "dcm_rms_mm": res.summary["DCM 誤差 RMS [mm]"], "tau_sat_pct": round(100 * sat, 2),
         "pitch_max": res.summary["ピッチ最大 [deg]"], "wall_s": round(time.time() - t0, 1)}
    r["success"] = (not r["fell"]) and r["box_dx"] >= 0.5
    with OUT.open("a") as fp:
        fp.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(r, ensure_ascii=False), flush=True)
