"""! @file run_box.py
@brief 箱押しの実験設定（armtasks/boxpush/configs/experiments/*.yaml）を lococtrl の実行系で回す。
実験を設定しているyamlを指定して実行するコマンド

`lococtrl.cli.run_biped --config` と同じ約束（1 ファイル = 1 条件、未知キーはエラー）に、
箱押し専用の `push` 節を足したもの。`push` 節があれば手先タスク（`boxpush.push.BoxPushHook`）を
lococtrl の全身制御器に差し込む。設定の `robot:` は実験設定ファイル基準の相対パスで書ける。

```bash
python -m boxpush.cli.run_box box_p0                 # configs/experiments/box_p0.yaml
python -m boxpush.cli.run_box box_p1_stand_push --viewer
python -m boxpush.cli.run_box configs/experiments/box_p0.yaml --config-print changed
```
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import yaml

from .. import PKG_ROOT
from ..push import BoxPushHook, PushParams

EXPERIMENTS = PKG_ROOT / "configs" / "experiments"
ROBOTS = PKG_ROOT / "configs"

__all__ = ["resolve_experiment", "resolve_robot", "load_box_experiment", "run_box_experiment", "main"]


def resolve_experiment(name: str) -> Path:
    """! @brief 実験設定名をファイルパスにする。
    @param name "box_p0" / "box_p0.yaml" / パス [-]
    @return Path
    """
    p = Path(name)
    if p.is_file():
        return p.resolve()
    cand = EXPERIMENTS / (name if name.endswith((".yaml", ".yml", ".json")) else f"{name}.yaml")
    if cand.is_file():
        return cand.resolve()
    raise FileNotFoundError(f"実験設定が見つからない: {name}（{EXPERIMENTS} 直下、またはパス）")


def resolve_robot(name: str | None) -> Path | None:
    """! @brief ロボット設定名をファイルパスにする（`--robot` の差し替え用）。
    @param name "robot_g1_box_sphere" / "robot_g1_box_sphere.yaml" / パス / None [-]
    @return Path（None なら None: 実験設定の robot: をそのまま使う）
    """
    if not name:
        return None
    p = Path(name)
    if p.is_file():
        return p.resolve()
    cand = ROBOTS / (name if name.endswith((".yaml", ".yml")) else f"{name}.yaml")
    if cand.is_file():
        return cand.resolve()
    raise FileNotFoundError(f"ロボット設定が見つからない: {name}（{ROBOTS} 直下、またはパス）")


def load_box_experiment(path: str | Path, robot: str | Path | None = None):
    """! @brief 実験設定を読む。`push` 節を取り出し、残りを lococtrl に渡す。
    @param path 実験設定ファイル [-]
    @param robot ロボット設定の差し替え [-]（例 robot_g1_box_sphere）。None なら実験設定の robot: を使う
    @return (BipedExperimentConfig, PushParams | None)
    """
    from lococtrl.cli.biped_experiment import experiment_config_from_dict

    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    push_d = data.pop("push", None)
    cfg = experiment_config_from_dict(data)
    cfg.config_path = p
    r = str(cfg.robot)
    if ("/" in r or "\\" in r) and not Path(r).is_absolute():
        cand = (p.parent / r).resolve()
        if cand.is_file():
            cfg.robot = str(cand)
    rp = resolve_robot(robot)
    if rp is not None:
        cfg.robot = str(rp)
    push = None
    if push_d is not None:
        push_d = {k: v for k, v in push_d.items() if not str(k).startswith("_comment")}
        names = {f.name for f in fields(PushParams)}
        unknown = set(push_d) - names
        if unknown:
            raise KeyError(f"[push] 未知のキー {sorted(unknown)}（使えるキー: {sorted(names)}）")
        if cfg.scenario.box.enable:
            push_d.setdefault("box_half_x", float(cfg.scenario.box.half_size[0]))
        push = PushParams(**push_d)
    return cfg, push


def make_hook(cfg, push: PushParams, policy: str | None = None, actions: str | None = None):
    """! @brief 毎周期フックを作る。policy が None なら手書きスケジュール、"zero" なら固定値だけ、
    ディレクトリなら学習済み MLP（`boxpush.policy.MlpBoxPolicy`）。
    @param cfg     BipedExperimentConfig [-]
    @param push    PushParams（固定値・スケジュールの元）[-]
    @param policy  None | "schedule" | "zero" | 学習済みディレクトリ [-]
    @param actions 方策が出す行動（カンマ区切り。zero / 学習済みのとき）[-]
    @return BoxPushApplier
    """
    from lococtrl.common.config import load_robot_config
    from lococtrl.control.hand_task import HandTask

    from ..policy import BoxActionSpec, BoxPushApplier, MlpBoxPolicy, ZeroBoxPolicy

    tau_max = np.asarray(load_robot_config(cfg.robot).tau_max)
    if policy in (None, "schedule"):
        return BoxPushHook(push, dt=cfg.sim.dt_ctrl, tau_max=tau_max)
    if policy == "zero":
        spec = BoxActionSpec(actions=tuple((actions or "f_push,lean_x").split(",")))
        pol = ZeroBoxPolicy(spec)
    else:
        pol = MlpBoxPolicy(policy, None if actions is None else BoxActionSpec(actions=tuple(actions.split(","))))
        spec = pol.spec
    if spec.residual_actions:
        raise ValueError("残差層（条件 B / C）の方策は run_box では回せない（制御器に残差を書く口が無い）。"
                         "boxpush.cli.eval_policy か boxpush.cli.record（BoxPushEnv 経由）を使う")
    spec.fixed.update({"f_push": push.f_push, "lean_x": 0.0 if push.lean_x is None else push.lean_x,
                       "com_dz": push.dz_com, "hand_dz": push.dz})
    return BoxPushApplier(spec=spec, policy=pol, hand_task=HandTask(n_arm=2, kp=push.kp, kd=push.kd, f_max=push.f_max),
                          dt=cfg.sim.dt_ctrl, box_half_x=push.box_half_x, standoff=push.standoff, ki=push.ki,
                          ki_limit=push.ki_limit, vx=push.vx, t_walk=push.t_start, tau_max=tau_max)


def run_box_experiment(cfg, push: PushParams | None, viewer: bool = False, realtime: bool = False,
                       policy: str | None = None, actions: str | None = None):
    """! @brief 1 条件を実行し、押しの指標を要約に足す。
    @param cfg  BipedExperimentConfig [-]
    @param push PushParams（None なら手先タスク無し = lococtrl そのまま）[-]
    @param policy / actions  `make_hook` を見よ
    @return BipedWalkResult
    """
    from lococtrl.cli.biped_experiment import run_experiment

    if push is None:
        return run_experiment(cfg, viewer=viewer, realtime=realtime)
    hook = make_hook(cfg, push, policy, actions)
    res = run_experiment(cfg, viewer=viewer, realtime=realtime, hand_task=hook.hand_task, extra_on_step=hook)
    log = res.extra
    if getattr(log, "f_hand_cmd", np.zeros(0)).size and log.wrench_hands.size:
        t = np.asarray(res.t)
        if hasattr(hook, "f_push_at"):
            f_cmd = np.array([hook.f_push_at(ti) for ti in t])       # 両手合計の指令 [N]（スケジュール）
        else:
            f_cmd = np.full(len(t), float(push.f_push))              # 学習済み: 固定値を基準に見る
        f_env_x = -log.wrench_hands[:, :, 0].sum(axis=1)             # 箱が手を押し返す力の −x → 押付け力 [N]
        on = f_cmd >= 0.999 * push.f_push
        if on.any():
            err = f_env_x[on] - f_cmd[on]
            res.summary["押付け力 指令/実測平均 [N]"] = [round(float(push.f_push), 1), round(float(f_env_x[on].mean()), 1)]
            res.summary["押付け力 追従誤差 RMS [N] / 指令比 [%]"] = [round(float(np.sqrt(np.mean(err ** 2))), 2),
                                                             round(100.0 * float(np.sqrt(np.mean(err ** 2))) / push.f_push, 1)]
        res.summary["手先力指令 max [N]"] = round(float(np.linalg.norm(log.f_hand_cmd, axis=2).max()), 1)
        if log.box_pose.size:
            v = log.box_pose[:, 7]
            res.summary["箱の速度 max [m/s]"] = round(float(v.max()), 3)
        res.summary["ピッチ 終端 [deg]"] = round(float(np.degrees(res.rpy[-1, 1])), 2)
        if cfg.output.verbose:
            for k in ("押付け力 指令/実測平均 [N]", "押付け力 追従誤差 RMS [N] / 指令比 [%]", "手先力指令 max [N]",
                      "箱の速度 max [m/s]", "ピッチ 終端 [deg]"):
                if k in res.summary:
                    print(f"  {k}: {res.summary[k]}")
    return res


def main(argv: list[str] | None = None) -> int:
    """! @brief CLI 入口。
    @param argv コマンドライン引数 [-]
    @return 終了コード [-]
    """
    from lococtrl.cli.biped_experiment import describe_config

    ap = argparse.ArgumentParser(description="箱押し実験（lococtrl の run_biped --config + push 節）")
    ap.add_argument("config", help="実験設定（configs/experiments/ 直下の名前、またはパス）")
    ap.add_argument("--viewer", action="store_true", help="MuJoCo ビューアを開く")
    ap.add_argument("--realtime", action="store_true", help="実時間で回す")
    ap.add_argument("--config-print", choices=["full", "changed", "compact", "none"], default="changed")
    ap.add_argument("--policy", default=None, help="schedule（既定）| zero | 学習済みディレクトリ（params.npz + meta.json）")
    ap.add_argument("--actions", default=None, help="方策が出す行動（カンマ区切り。zero / 学習済みのとき）")
    a = ap.parse_args(sys.argv[1:] if argv is None else argv)

    cfg, push = load_box_experiment(resolve_experiment(a.config))
    if a.config_print != "none":
        print(describe_config(cfg, changed_only=(a.config_print == "changed"),
                              compact=(a.config_print == "compact")))
        if push is not None:
            print("  push:", {f.name: getattr(push, f.name) for f in fields(PushParams)})
    res = run_box_experiment(cfg, push, viewer=a.viewer, realtime=a.realtime, policy=a.policy, actions=a.actions)
    return 1 if res.fell else 0


if __name__ == "__main__":
    raise SystemExit(main())
