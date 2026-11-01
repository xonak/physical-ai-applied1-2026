"""! @file train_ppo.py
@brief 箱押し上位方策の学習（stable-baselines3 PPO、プロセス並列）と、学習済みを NumPy 推論用（params.npz + meta.json）に書き出すスクリプト。

依存: リポジトリ直下で `pip install -r requirements-train.txt`（CPU 版の PyTorch でよい。制御ループが CPU 律速で、GPU は効かない）。
`--n-envs` は PC のスペックに合わせて決める（目安は物理コア数 − 1）。1 プロセスは物理時間のおよそ 0.5 倍速で進む
（10 Hz 判断で約 5 判断/s）。

学習の条件は `--preset`（`boxpush.policy.PRESETS`）で選ぶ。説明資料の方策は次の 2 つである。
- P      : RL-M1。押付け力、手先高さと計画層の残差（CMP オフセット 2、重心高さ、足配置 3）の 8 行動。報酬の姿勢項 w_post = 0。10 Hz
- legacy : RL-M2。押付け力と前傾の 2 行動。w_post = 1。10 Hz
ほかに A（腕だけ）、B / B_mom（+ 安定化層の残差、100 Hz）、C（+ 全身制御の残差トルク、500 Hz）、P_S / P_S_T、S / S_T を
選べる（説明資料では評価していない）。腕・計画層の行動はどの条件でも 100 ms ごとにしか更新しない（残差だけ速く回す）。
学習曲線は tensorboard の `time/sim_seconds`（シミュレーション時間）を横軸に比べる（判断周期が違うと判断数では比べられない）。
判断周期が違っても同じ物理時間の割引になるよう、gamma = 0.99^(decimation/50)、gae_lambda = 0.95^(decimation/50)、
1 環境のロールアウト長 = n_steps × 50 / decimation にする。--total-steps は判断数なので、同じシミュレーション時間なら
100 Hz の条件は 10 Hz の 10 倍、500 Hz の条件は 50 倍の判断数になる（30 万 → 300 万 → 1500 万）。
初期の探索幅は腕と残差で分ける（--log-std-arm / --log-std-res）。残差を大きく揺らすと学習の初めに転んでばかりになる。

```bash
cd armtasks/boxpush
python -m boxpush.train.train_ppo box_p2_walk_push --preset P      --n-envs 8 --total-steps 300000 --out ../../outputs/train_rl_m1
python -m boxpush.train.train_ppo box_p2_walk_push --preset legacy --n-envs 8 --total-steps 300000 --out ../../outputs/train_rl_m2
python -m boxpush.train.train_ppo box_p2_walk_push --preset legacy --n-envs 8 --total-steps 100000 --out ../../outputs/train_rl_m2 \\
       --resume ../../outputs/train_rl_m2/ppo.zip
python -m boxpush.cli.eval_policy box_p2_walk_push --policies analytic,../../outputs/train_rl_m1/export,../../outputs/train_rl_m2/export
```

書き出し（export/）: params.npz（W0, b0, W1, b1, ...: 方策ネットの平均の経路。tanh、最後に tanh で [-1,1]）、
meta.json（obs_mean, obs_std, actions, obs_size, act_size, decimation, preset）。`boxpush.policy.MlpBoxPolicy` が読む。

@note 書き出した方策は出力層の後に tanh を掛ける（\\f$a = \\tanh(\\mu)\\f$）。学習中の PPO は平均 \\f$\\mu\\f$ に
      探索の雑音を足した値を [-1, 1] で clip して環境に渡すので、学習時と推論時で行動の写像は同一ではない。
      また、学習時の VecNormalize が掛ける観測の ±10 の clip は、推論側では掛けない。
      説明資料の評価値は、書き出した方策（tanh、観測の clip なし）で得たものである。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def export_sb3_policy(model, out_dir: Path, actions: tuple[str, ...], obs_rms=None, spec=None) -> Path:
    """! @brief SB3 の MlpPolicy（tanh 活性）を params.npz + meta.json に書く。
    @note 推論側（`boxpush.policy.MlpBoxPolicy`）は出力層の後に tanh を掛ける。学習中の PPO は行動を clip で
          [-1, 1] に収めるので、両者の行動の写像は同一ではない（モジュール冒頭の @note を参照）。
    @param model   PPO [-]
    @param out_dir 書き出し先 [-]
    @param actions 行動の名前 [-]
    @param obs_rms VecNormalize の obs_rms（None なら正規化なし）[-]
    @return out_dir
    """
    import torch

    out_dir.mkdir(parents=True, exist_ok=True)
    pol = model.policy
    layers = []
    # 共有/方策側の MLP 抽出器 → 行動の平均
    net = pol.mlp_extractor.policy_net
    for m in net:
        if isinstance(m, torch.nn.Linear):
            layers.append((m.weight.detach().cpu().numpy().T, m.bias.detach().cpu().numpy()))
    lin = pol.action_net
    layers.append((lin.weight.detach().cpu().numpy().T, lin.bias.detach().cpu().numpy()))
    params = {}
    for i, (W, b) in enumerate(layers):
        params[f"W{i}"] = W.astype(np.float64)
        params[f"b{i}"] = b.astype(np.float64)
    np.savez(out_dir / "params.npz", **params)
    obs_size = layers[0][0].shape[0]
    meta = {"actions": list(actions), "obs_size": int(obs_size), "act_size": int(layers[-1][0].shape[1]),
            "obs_mean": (obs_rms.mean.tolist() if obs_rms is not None else [0.0] * obs_size),
            "obs_std": (np.sqrt(obs_rms.var + 1e-8).tolist() if obs_rms is not None else [1.0] * obs_size),
            "note": "tanh MLP。最後に tanh で [-1,1]。boxpush.policy.MlpBoxPolicy が読む", "exported": time.strftime("%Y-%m-%d %H:%M")}
    if spec is not None:
        meta["decimation"] = int(spec.decimation)
        meta["slow_period"] = int(spec.slow_period)
        meta["preset"] = str(spec.preset_name)
        meta["f_push_max"] = float(spec.f_push_max)
        meta["scales"] = {k: float(v) for k, v in spec.scales.items()}
        meta["fixed"] = {k: float(v) for k, v in spec.fixed.items()}
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return out_dir


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="箱押し上位方策の PPO 学習")
    ap.add_argument("config", help="実験設定（configs/experiments/ の名前）。lococtrl 部分を環境の設定にする")
    ap.add_argument("--preset", default="A", help="学習の条件（boxpush.policy.PRESETS のキー。RL-M1 は P、RL-M2 は legacy）")
    ap.add_argument("--actions", default=None, help="行動を直接指定（カンマ区切り。--preset より優先）")
    ap.add_argument("--decimation", type=int, default=None, help="判断周期（制御周期の倍数。既定は条件の値）")
    ap.add_argument("--base", default=None,
                    help="基底方策（学習済みディレクトリ、例 ../../models/rl_m2）。凍結して上位の指令源にし、RL は残差を学ぶ")
    ap.add_argument("--slow-period", type=int, default=None,
                    help="腕・計画層の行動を更新する周期（制御周期の倍数。既定 50 = 100 ms。decimation と同じ値で保持なし）")
    ap.add_argument("--log-std-arm", type=float, default=-0.5, help="腕などの行動の初期 log 標準偏差")
    ap.add_argument("--log-std-res", type=float, default=-1.6, help="下層残差の行動の初期 log 標準偏差（std 0.2）")
    ap.add_argument("--log-std-plan", type=float, default=-1.0, help="計画層残差の行動の初期 log 標準偏差（std 0.37）")
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--total-steps", type=int, default=200_000)
    ap.add_argument("--max-time", type=float, default=10.0)
    ap.add_argument("--vx", type=float, default=0.1, help="押しながら歩く速度（0 で立位）")
    ap.add_argument("--out", default="../../outputs/train_ppo")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-randomize", action="store_true")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-steps", type=int, default=256, help="1 環境あたりのロールアウト長（10 Hz 換算の判断数）")
    ap.add_argument("--batch-size", type=int, default=256, help="ミニバッチ（10 Hz 換算。判断周期の比で増やす）")
    a = ap.parse_args(sys.argv[1:] if argv is None else argv)

    try:
        import gymnasium  # noqa: F401
        import torch
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
    except ImportError as e:  # pragma: no cover
        print("stable-baselines3 / gymnasium / torch が要る: pip install stable-baselines3 gymnasium torch", file=sys.stderr)
        raise SystemExit(2) from e

    from ..cli.run_box import load_box_experiment, resolve_experiment
    from ..env import BoxDomain, make_gym_env
    from ..policy import PLAN_NAMES, BoxActionSpec

    cfg_path = resolve_experiment(a.config)

    def make_spec():
        if a.actions:
            sp = BoxActionSpec(actions=tuple(a.actions.split(",")), preset_name="custom")
        else:
            sp = BoxActionSpec.preset(a.preset)
        if a.decimation is not None:
            sp.decimation = int(a.decimation)
        if a.slow_period is not None:
            sp.slow_period = int(a.slow_period)
        return sp

    spec0 = make_spec()
    actions = spec0.actions
    ratio = 50.0 / max(1, spec0.decimation)          # 10 Hz に対する判断の多さ
    gamma = 0.99 ** (1.0 / ratio)
    lam = 0.95 ** (1.0 / ratio)
    n_steps = int(round(a.n_steps * ratio))
    batch = int(round(a.batch_size * ratio))
    print(f"条件 {spec0.preset_name}: 行動 {len(actions)} 次元 {actions[:6]}{' ...' if len(actions) > 6 else ''}、"
          f"判断 {500.0 / spec0.decimation:.0f} Hz、gamma {gamma:.4f}、lambda {lam:.4f}、n_steps {n_steps}、batch {batch}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    def make(rank: int):
        def _f():
            cfg, push = load_box_experiment(cfg_path)
            cfg.output.verbose = False
            spec = make_spec()
            if push is not None:
                spec.fixed.update({"f_push": push.f_push, "lean_x": 0.0 if push.lean_x is None else push.lean_x,
                                   "com_dz": push.dz_com, "hand_dz": push.dz})
            from ..policy import MlpBoxPolicy

            base = MlpBoxPolicy(a.base) if a.base else None
            return make_gym_env(cfg, spec=spec, domain=BoxDomain(randomize=not a.no_randomize),
                                max_time=a.max_time, vx=a.vx, seed=a.seed + rank, base_policy=base)
        return _f

    venv = SubprocVecEnv([make(i) for i in range(a.n_envs)]) if a.n_envs > 1 else DummyVecEnv([make(0)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)
    torch.set_num_threads(1)
    policy_kwargs = dict(net_arch=dict(pi=[64, 64], vf=[64, 64]), activation_fn=torch.nn.Tanh)
    if a.resume:
        model = PPO.load(a.resume, env=venv, device="cpu")
    else:
        model = PPO("MlpPolicy", venv, learning_rate=a.lr, n_steps=n_steps, batch_size=batch, n_epochs=10,
                    gamma=gamma, gae_lambda=lam, clip_range=0.2, ent_coef=0.0, policy_kwargs=policy_kwargs,
                    seed=a.seed, device="cpu", verbose=1, tensorboard_log=str(out / "tb"))
        with torch.no_grad():
            ls = model.policy.log_std
            ls[:] = float(a.log_std_arm)
            ridx = spec0.residual_index
            if ridx.size:
                ls[torch.as_tensor(ridx)] = float(a.log_std_res)
            pidx = [i for i, n in enumerate(spec0.actions) if n in PLAN_NAMES]
            if pidx:
                ls[torch.as_tensor(pidx)] = float(a.log_std_plan)
    from stable_baselines3.common.callbacks import BaseCallback

    class SimTime(BaseCallback):
        """! @brief 判断数をシミュレーション時間 [s] に直して記録する（条件間の比較用）。"""

        def _on_rollout_end(self) -> None:
            self.logger.record("time/sim_seconds", self.model.num_timesteps * spec0.decimation * 0.002)

        def _on_step(self) -> bool:
            return True

    t0 = time.time()
    model.learn(total_timesteps=a.total_steps, progress_bar=False, callback=SimTime(), reset_num_timesteps=not a.resume)
    wall = time.time() - t0
    model.save(out / "ppo.zip")
    venv.save(str(out / "vecnormalize.pkl"))
    export_sb3_policy(model, out / "export", actions, obs_rms=venv.obs_rms, spec=spec0)
    if a.base:   # 基底方策を同梱して meta から相対参照する（推論側が自動で読む）
        import shutil

        shutil.copytree(a.base, out / "export" / "base", dirs_exist_ok=True)
        mp = out / "export" / "meta.json"
        meta = json.loads(mp.read_text(encoding="utf-8")); meta["base"] = "base"
        mp.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "train_info.json").write_text(json.dumps({
        "config": str(cfg_path), "preset": spec0.preset_name, "actions": list(actions), "decimation": spec0.decimation,
        "gamma": gamma, "gae_lambda": lam, "n_steps": n_steps, "batch_size": batch, "base": a.base,
        "n_envs": a.n_envs, "total_steps": a.total_steps,
        "wall_s": round(wall, 1), "steps_per_s": round(a.total_steps / max(wall, 1e-9), 1)}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"学習 {a.total_steps} 判断 / {wall:.0f} s = {a.total_steps / wall:.1f} 判断/s。書き出し: {out / 'export'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
