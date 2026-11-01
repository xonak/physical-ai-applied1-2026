"""! @file eval_policy.py
@brief 方策の評価: 箱の質量ごとに複数シードで BoxPushEnv を回し、成功率・前傾・押付け力などを JSONL に書く。
どれだけの質量まで押せるようになったかを数値で比較する。

方策（`--policies` にカンマ区切り）:
- `upright`  : 手書き。押付け力 = 静止摩擦 × 1.15（質量を知っている）、前傾 0
- `schedule` : 手書き。押付け力は同上、前傾 = F / (m ω²)（質量を知っている）
- `reactive` : 手書き。押付け力は同上、前傾は手首 F/T の実測から（ReactiveBoxPolicy）
- `analytic` : 説明資料の Analytic。押付け力は同上（腕のスケジュール）、前傾は脚側の解析則（BoxPushApplier）
- ディレクトリ: 学習済み（params.npz + meta.json、MlpBoxPolicy）。質量は知らない。行動と判断周期は meta.json から
  （説明資料の RL-M1 は ../../models/rl_m1、RL-M2 は ../../models/rl_m2）

手書き 4 つは質量を知っている比較基準である（押付け力と前傾を質量から計算する）。
学習方策は観測（手の力、箱の動き）だけから決める。

```bash
cd armtasks/boxpush
python -m boxpush.cli.eval_policy box_p2_walk_push --robot robot_g1_box \\
       --policies analytic,../../models/rl_m1,../../models/rl_m2 --labels Analytic,RL-M1,RL-M2 \\
       --masses 10,20,30,40 --seeds 3 --friction 0.15,0.35 --x-offset 0,0 --workers 4 --out ../../outputs/eval_main.jsonl
python tools/plot_eval.py ../../outputs/eval_main.jsonl ../../outputs/figs/eval_main.png
```
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

M_ROBOT = 33.34      # G1（腕付き）の質量 [kg]
OMEGA = 3.90         # LIPM 固有振動数（z_c ≈ 0.645 m）[rad/s]
G = 9.81


def _make_policy(name: str, spec, push, mass: float, mu: float):
    """! @brief 方策と、それに合う BoxActionSpec を作る。"""
    from ..policy import BoxActionSpec, MlpBoxPolicy, ReactiveBoxPolicy, ScheduleBoxPolicy

    if name == "analytic":
        f = 1.15 * mu * mass * G
        sp = BoxActionSpec.preset("A", f_push_max=max(120.0, 1.2 * f))
        pol = ScheduleBoxPolicy(sp, targets={"f_push": f}, t_start=push.t_start, t_ramp=push.t_ramp)
        return pol, sp
    if name in ("upright", "schedule", "reactive"):
        f = 1.15 * mu * mass * G
        sp = BoxActionSpec(actions=("f_push", "lean_x"), f_push_max=max(120.0, 1.2 * f))
        sp.fixed.update({"com_dz": 0.0, "hand_dz": 0.0})
        lean = 0.0 if name == "upright" else f / (M_ROBOT * OMEGA ** 2)
        cls = ReactiveBoxPolicy if name == "reactive" else ScheduleBoxPolicy
        pol = cls(sp, targets={"f_push": f, "lean_x": lean}, t_start=push.t_start, t_ramp=push.t_ramp,
                  t_lean_lead=push.t_lean_lead, lean_pre_max=push.lean_pre_max)
        if name == "reactive":
            pol.lean_min = push.lean_pre_max
            pol.dt_decide = sp.decimation * 0.002
        return pol, sp
    pol = MlpBoxPolicy(name)
    pol.spec.fixed.update({"com_dz": 0.0, "hand_dz": 0.0, "step_dx": 0.0, "step_dy": 0.0, "step_dt": 0.0})
    return pol, pol.spec


def run_one(job: dict) -> dict:
    """! @brief 1 エピソード（方策 × 質量 × シード）。プロセス並列のために dict で受け渡す。"""
    import boxpush  # noqa: F401  lococtrl を sys.path に

    from ..cli.run_box import load_box_experiment
    from ..env import BoxDomain, BoxPushEnv

    cfg, push = load_box_experiment(Path(job["config"]), robot=job.get("robot"))
    cfg.output.verbose = False
    mass = float(job["mass"])
    mu_lo, mu_hi = job["friction"]
    rng = np.random.default_rng(job["seed"])
    mu = float(rng.uniform(mu_lo, mu_hi))
    pol, spec = _make_policy(job["policy"], None, push, mass, mu)
    env = BoxPushEnv(cfg, spec=spec, domain=BoxDomain(mass=(mass, mass), friction=(mu, mu), x_offset=tuple(job["x_offset"])),
                     max_time=job["max_time"], vx=job["vx"], t_walk=push.t_start, seed=job["seed"], base_policy=getattr(pol, "base", None))
    t0 = time.time()
    obs = env.reset()
    done, ret = False, 0.0
    leans, fcmd, fmeas, comz, res_abs = [], [], [], [], []
    ridx = spec.residual_index
    while not done:
        if hasattr(pol, "set_time"):
            pol.set_time(env.k * env.dt_ctrl)
        act = pol.act(obs)
        obs, r, done, info = env.step(act)
        ret += r
        if ridx.size:
            res_abs.append(float(np.mean(np.abs(np.asarray(act)[ridx]))))
        if info["t"] >= push.t_start + push.t_ramp:
            leans.append(env.applier.last_vals["lean_x"]); fcmd.append(env.applier.last_vals["f_push"])
            fmeas.append(info["f_meas"]); comz.append(info["com_z"])
    res = {
        "policy": job["policy_label"], "mass": mass, "mu": round(mu, 3), "seed": job["seed"],
        "success": bool(info.get("success", False)), "fell": bool(info.get("fell", False)),
        "box_tipped": bool(info.get("box_tipped", False)), "t_end": round(info["t"], 2),
        "box_dx": round(info["box_dx"], 3), "return": round(ret, 2),
        "lean_mean": round(float(np.mean(leans)), 4) if leans else None,
        "f_cmd_mean": round(float(np.mean(fcmd)), 1) if fcmd else None,
        "f_meas_mean": round(float(np.mean(fmeas)), 1) if fmeas else None,
        "com_z_mean": round(float(np.mean(comz)), 4) if comz else None,
        "preset": spec.preset_name, "res_abs_mean": round(float(np.mean(res_abs)), 4) if res_abs else None,
        "static_friction_N": round(mu * mass * G, 1), "wall_s": round(time.time() - t0, 1),
    }
    return res


def main(argv: list[str] | None = None) -> int:
    from ..cli.run_box import resolve_experiment, resolve_robot

    ap = argparse.ArgumentParser(description="方策の評価（成功率 × 箱の質量）")
    ap.add_argument("config", help="実験設定（例 box_p2_walk_push）")
    ap.add_argument("--policies", default="upright,schedule", help="upright,schedule,reactive,analytic,<学習済みディレクトリ>")
    ap.add_argument("--labels", default=None, help="図・表での名前（カンマ区切り、policies と同順）")
    ap.add_argument("--masses", default="10,20,30,40")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--friction", default="0.2,0.2", help="箱-床の摩擦の範囲（シードごとに乱択）")
    ap.add_argument("--x-offset", default="0.0,0.02")
    ap.add_argument("--max-time", type=float, default=10.0)
    ap.add_argument("--vx", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default="../../outputs/eval.jsonl")
    ap.add_argument("--robot", default=None, help="ロボット設定の差し替え（例 robot_g1_box_sphere）")
    a = ap.parse_args(sys.argv[1:] if argv is None else argv)

    cfg_path = str(resolve_experiment(a.config))
    pols = a.policies.split(",")
    labels = a.labels.split(",") if a.labels else [Path(p).parent.name if "/" in p or "\\" in p else p for p in pols]
    jobs = [{"config": cfg_path, "policy": p, "policy_label": lab, "mass": float(m), "seed": s,
             "friction": [float(v) for v in a.friction.split(",")], "x_offset": [float(v) for v in a.x_offset.split(",")],
             "max_time": a.max_time, "vx": a.vx, "robot": (str(resolve_robot(a.robot)) if a.robot else None)}
            for p, lab in zip(pols, labels) for m in a.masses.split(",") for s in range(a.seeds)]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if a.workers > 1:
        import multiprocessing as mp

        with mp.get_context("spawn").Pool(a.workers) as pool:
            it = pool.imap_unordered(run_one, jobs)
            for r in it:
                with out.open("a", encoding="utf-8") as fp:
                    fp.write(json.dumps(r, ensure_ascii=False) + "\n")
                print(json.dumps(r, ensure_ascii=False), flush=True)
    else:
        for j in jobs:
            r = run_one(j)
            with out.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(json.dumps(r, ensure_ascii=False), flush=True)
    # 集計
    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"\n{len(jobs)} エピソード / {time.time() - t0:.0f} s。集計（{out} 全体）:")
    for lab in dict.fromkeys(r["policy"] for r in rows):
        line = []
        for m in sorted({r["mass"] for r in rows}):
            rr = [r for r in rows if r["policy"] == lab and r["mass"] == m]
            if rr:
                line.append(f"{m:.0f} kg {sum(r['success'] for r in rr)}/{len(rr)}")
        print(f"  {lab:>12}: " + "  ".join(line))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
