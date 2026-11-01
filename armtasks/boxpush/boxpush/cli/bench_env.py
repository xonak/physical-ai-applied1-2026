"""! @file bench_env.py
@brief 学習スループットの計測: BoxPushEnv をダミー方策（Zero / Schedule / 乱数）で回し、1 時間あたりの判断数を出す。
スループット：単位時間当たりの方策の判断回数（環境の 1 step 方策の計算コストを含む）。
学習に何時間かかるかの目安になる。
```bash
python -m boxpush.cli.bench_env box_p1_stand_push --policy schedule --episodes 2
python -m boxpush.cli.bench_env box_p1_stand_push --policy random --steps 200
```
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from ..cli.run_box import load_box_experiment, resolve_experiment
from ..env import BoxDomain, BoxPushEnv
from ..policy import BoxActionSpec, ScheduleBoxPolicy, ZeroBoxPolicy


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="BoxPushEnv のスループット計測")
    ap.add_argument("config")
    ap.add_argument("--policy", choices=["zero", "schedule", "random"], default="schedule")
    ap.add_argument("--preset", default=None, help="学習の条件 A | B | B_mom | C | legacy（--actions より優先）")
    ap.add_argument("--actions", default="f_push,lean_x", help="方策が出す行動（カンマ区切り）")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--steps", type=int, default=0, help="0 でエピソード終端まで")
    ap.add_argument("--max-time", type=float, default=10.0)
    ap.add_argument("--vx", type=float, default=None, help="押しながら歩く速度（既定は実験設定の push.vx）")
    ap.add_argument("--no-randomize", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(sys.argv[1:] if argv is None else argv)

    cfg, push = load_box_experiment(resolve_experiment(a.config))
    cfg.output.verbose = False
    spec = BoxActionSpec.preset(a.preset) if a.preset else BoxActionSpec(actions=tuple(a.actions.split(",")))
    if push is not None:
        spec.fixed.update({"f_push": push.f_push, "lean_x": 0.0 if push.lean_x is None else push.lean_x,
                           "com_dz": push.dz_com, "hand_dz": push.dz})
    vx = a.vx if a.vx is not None else (push.vx if push is not None else 0.0)
    env = BoxPushEnv(cfg, spec=spec, domain=BoxDomain(randomize=not a.no_randomize), max_time=a.max_time, seed=a.seed,
                     vx=vx, t_walk=push.t_start if push is not None else 2.0)
    rng = np.random.default_rng(a.seed)
    if a.policy == "zero":
        pol = ZeroBoxPolicy(spec)
    elif a.policy == "schedule":
        pol = ScheduleBoxPolicy(spec, t_start=push.t_start if push else 2.0, t_ramp=push.t_ramp if push else 1.0,
                                t_lean_lead=push.t_lean_lead if push else 1.5,
                                lean_pre_max=push.lean_pre_max if push else 0.05)
    else:
        pol = None
    n_steps = 0
    t0 = time.perf_counter()
    for ep in range(a.episodes):
        obs = env.reset()
        ret, done, steps = 0.0, False, 0
        while not done:
            if pol is None:
                act = rng.uniform(-1, 1, spec.n_act)
                act[spec.residual_index] *= 0.2          # 残差は学習初期の探索幅（std 0.2）程度に
            else:
                if hasattr(pol, "set_time"):
                    pol.set_time(env.k * env.dt_ctrl)
                act = pol.act(obs)
            obs, r, done, info = env.step(act)
            ret += r; steps += 1
            if a.steps and steps >= a.steps:
                break
        n_steps += steps
        print(f"ep {ep}: steps {steps} return {ret:7.2f} t_sim {info['t']:.1f} s box_dx {info['box_dx']:.3f} success {info.get('success', False)} "
              f"mass {info['box_mass']:.1f} kg mu {info['box_friction']:.2f} f_meas {info['f_meas']:.1f} "
              f"fell {info.get('fell', False)} tipped {info.get('box_tipped', False)}", flush=True)
    wall = time.perf_counter() - t0
    sim_s = n_steps * spec.decimation * env.dt_ctrl
    print(f"判断 {n_steps} 回 / 実時間 {wall:.1f} s = {n_steps / wall:.1f} 判断/s → 1 時間 {3600 * n_steps / wall:,.0f} 判断"
          f"（シミュレーション {sim_s:.1f} s、実時間比 {sim_s / wall:.2f}）、1 プロセスあたり")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
