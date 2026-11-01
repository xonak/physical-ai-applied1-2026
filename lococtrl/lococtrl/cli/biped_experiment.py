"""! @file biped_experiment.py
@brief 二足の実験設定（機能の組み合わせとパラメータ）を 1 つの YAML / JSON で決める。

設定ファイルは次の規則で読み込む:

- キーは dataclass のフィールド名。セクション（robot / gait / planner / stabilizer / mapper /
  policy / hl_policy / sim / scenario / output）は `BipedExperimentConfig` のフィールド名
- 書かなかった項目は既定値（`configs/biped.yaml` の値 = dataclass の既定）
- `_comment` で始まるキーは無視する。**未知のキーはエラー**（typo を黙って無視しない）
- 拡張子で YAML（.yaml / .yml）と JSON（.json）を切り替える
- `dump_experiment_config()` で今の設定を雛形として書き出せる
- 実行前に `describe_config()` で設定をターミナルに出す（* = 既定値から変えた項目）

使い方:

```bash
python -m lococtrl.cli.run_biped --dump-config configs/experiments/my_case.yaml   # 雛形を書く
python -m lococtrl.cli.run_biped --config configs/experiments/b5_ihmc.yaml        # 実行
python -m lococtrl.cli.run_biped --config ... --config-print changed              # 変えた項目だけ表示
```

設定ファイルで実行するとき、条件を決める CLI フラグ（--mapper, --n-step など）は併用できない。
条件は必ずファイルに書く（再現性のため）。
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..common.config import _REPO_ROOT, load_yaml
from ..control.biped_stack import StabilizerParams
from ..gait.biped_schedule import BipedGaitParams

__all__ = [
    "GaitSection", "PlannerSection", "IhmcSection", "DlrSection", "PdSection", "MapperSection",
    "OnlineSection", "PolicySection", "HlPolicySection", "SimSection", "BoxSection", "ScenarioSection", "OutputSection", "BipedExperimentConfig",
    "experiment_config_from_dict", "load_experiment_config", "experiment_config_to_dict",
    "dump_experiment_config", "describe_config", "run_experiment",
]


# ======================================================================
# セクション
# ======================================================================
@dataclass
class GaitSection:
    """! @brief 歩容（`gait.biped_schedule.BipedGaitParams` と同じ項目）。既定は biped.yaml の walk_3d。
    @param n_step 歩数 [-] / step_length 歩幅 [m] / step_width 足の y 間隔 [m] / t_ds 両脚支持期 [s] /
           t_ss 単脚支持期 [s] / swing_height 遊脚高さ [m] / t_start, t_end 静止時間 [s] /
           first_swing 最初に踏み出す足（0 = 左）[-] / ds_subdiv DS の分割数 [-]
    """

    n_step: int = 10
    step_length: float = 0.20
    step_width: float = 0.237
    t_ds: float = 0.15
    t_ss: float = 0.35
    swing_height: float = 0.05
    t_start: float = 1.0
    t_end: float = 2.0
    first_swing: int = 0
    ds_subdiv: int = 10

    def to_params(self) -> BipedGaitParams:
        """! @brief BipedGaitParams に変換する。
        @return BipedGaitParams
        """
        return BipedGaitParams(**{f.name: getattr(self, f.name) for f in fields(self)})


@dataclass
class OnlineSection:
    """! @brief オンライン足配置計画（`planner.biped_online.OnlinePlannerParams` と同じ項目）。

    @param vx / vy / wz     速度指令 [m/s, m/s, rad/s]（source = velocity、パターン座標系）
    @param turn_max_step    1 歩で回してよい向きの上限 [rad]
    @param source           "velocity"（速度指令から足配置を生成）| "stream"（`footsteps` を順に踏む）
    @param k_step           DCM 誤差の着地位置への反映率 [-]（0 で着地位置の調整なし）
    @param adapt_timing     時間調整を使うか [-]
    @param timing_margin    時間調整を始める DCM 誤差 [m]
    @param t_ss_min         単脚支持の下限 [s]
    @param v_swing_max      遊脚の最大水平速度 [m/s]（届く範囲の制限）
    @param t_ss_extra       未接地のとき単脚支持を延長できる上限 [s]
    @param touchdown_depth  遊脚終端を地面より下に置く量 [m]
    @param step_max_length  前後の最大歩幅 [m]
    @param step_min_width   左右の最小間隔 [m]
    @param step_max_width   左右の最大間隔 [m]
    @param latch_ratio      着地目標を固定する遊脚進捗 [-]
    @param preview_steps    先読みする歩数 [-]
    @param duration         stream のときの実行時間 [s]
    @param closing_step     最後に足を揃える 1 歩を入れるか [-]
    @param z_c_offset       重心高さのずらし [m]
    @param stop_margin      歩き終わりに移ってよい DCM 誤差 [m]
    @param max_extra_steps  回復の歩数の上限 [-]
    @param endless          歩数で終わらず指令が続く限り歩く（テレオペ。実行時間は duration）
    @param idle_speed / idle_turn  これ未満の指令で足を揃えて立つ [m/s, rad/s]
    @param lean_xy          DCM 参照の CoP 列（CMP）を足から前へずらす量（前傾、パターン座標系）[m]
    @param z_c_rate         重心高さの切替速度の上限 [m/s]（set_com_height、両脚支持・立位のみ）
    @param velocity_schedule 歩行中の速度変更 [[t, vx, vy] または [t, vx, vy, wz], ...]（t [s] 以降にその指令にする）
    @param footsteps        source = stream のときの足配置列 [[foot, x, y], ...]（foot 0 = 左, 1 = 右、{W} [m]）
    """

    vx: float = 0.2
    vy: float = 0.0
    wz: float = 0.0
    turn_max_step: float = 0.3
    source: str = "velocity"
    k_step: float = 1.0
    adapt_timing: bool = True
    timing_margin: float = 0.10
    t_ss_min: float = 0.25
    v_swing_max: float = 1.2
    t_ss_extra: float = 0.0
    touchdown_depth: float = 0.005
    step_max_length: float = 0.35
    step_min_width: float = 0.15
    step_max_width: float = 0.40
    latch_ratio: float = 0.7
    preview_steps: int = 3
    duration: float = 10.0
    closing_step: bool = True
    z_c_offset: float = -0.04
    stop_margin: float = 0.04
    max_extra_steps: int = 4
    endless: bool = False
    idle_speed: float = 0.02
    idle_turn: float = 0.05
    lean_xy: list = field(default_factory=lambda: [0.0, 0.0])
    z_c_rate: float = 0.15
    velocity_schedule: list = field(default_factory=list)
    footsteps: list = field(default_factory=list)


@dataclass
class PlannerSection:
    """! @brief 足配置・重心軌道の計画器。
    @param kind   "schedule" = オフライン固定列 | "online" = オンライン計画
    @param online kind = online のときの設定
    """

    kind: str = "schedule"
    online: OnlineSection = field(default_factory=OnlineSection)


@dataclass
class IhmcSection:
    """! @brief 運動量ベース全身 QP の重み（`wbc_momentum.MomentumWbcWeights`）。"""

    w_contact: float = 1.0e2
    w_momentum: float = 1.0
    w_swing: float = 1.0
    w_base: float = 1.0e-1
    w_posture: float = 1.0e-4
    w_wrench: float = 1.0e-4


@dataclass
class DlrSection:
    """! @brief 受動性ベース全身制御の固有パラメータ（`wbc_passivity.PassivityWbcParams`）。"""

    w_com: float = 1.0e2
    w_reg: float = 1.0e-2
    m_foot_eq: float = 2.0
    i_foot_eq: float = 0.05
    posture_scale: float = 1.0
    null_damping: float = 1.0e-3
    i_base_eq: float = 0.3
    arm_gain_scale: float = 10.0


@dataclass
class PdSection:
    """! @brief IK + 関節 PD（位置制御、mapper = pd）の設定。
    @param kp_scale 関節 P ゲインの倍率 [-] / ik_iter IK 反復 [-] / use_ff 重力 FF [-] /
           k_dcm DCM フィードバック [-]（pd だけ別に持つ。0 で開ループ）
    """

    kp_scale: float = 1.0
    ik_iter: int = 8
    use_ff: bool = True
    k_dcm: float = 0.0


@dataclass
class MapperSection:
    """! @brief 下層の制御（全身制御器）。
    @param kind "pd" | "ihmc" | "dlr"
    """

    kind: str = "ihmc"
    ihmc: IhmcSection = field(default_factory=IhmcSection)
    dlr: DlrSection = field(default_factory=DlrSection)
    pd: PdSection = field(default_factory=PdSection)


@dataclass
class PolicySection:
    """! @brief 学習方策。
    @param kind "none" | "residual"（mapper の上に残差トルク方策を載せる。ihmc / dlr のみ）
    @param model_dir 学習済み方策（params.npz + meta.json）。None なら残差ゼロ
    @param tau_scale 残差トルクの上限 [N*m]
    """

    kind: str = "none"
    model_dir: str | None = None
    tau_scale: float = 10.0


@dataclass
class HlPolicySection:
    """! @brief HL（上位）方策: オンライン計画への着地位置・時間の残差（`policy.biped_hl_policy`、planner.kind = online のみ）。
    @param kind       "none" | "zero"（残差ゼロ、口の確認）| "oracle"（解析則を方策の形で与える）| "mlp"（学習済み）
    @param model_dir  mlp のとき params.npz（+ meta.json）のあるディレクトリ
    @param k_oracle   oracle の DCM 誤差ゲイン [-]（planner.online.k_step = 0 と組み合わせると解析則の着地位置調整と同等）
    @param scale_xy   着地位置残差の上限 [m]
    @param scale_t    単脚支持時間残差の上限 [s]
    @param decimation 方策を呼ぶ周期（制御周期の整数倍）[-]
    """

    kind: str = "none"
    model_dir: str | None = None
    k_oracle: float = 1.0
    scale_xy: float = 0.10
    scale_t: float = 0.10
    decimation: int = 25


@dataclass
class SimSection:
    """! @brief シミュレーション。
    @param dt_ctrl 制御周期 [s] / solref 接触の solref [time_const, damping_ratio]（None で MJCF のまま）
    """

    dt_ctrl: float = 0.002
    solref: list | None = None


@dataclass
class BoxSection:
    """! @brief シーンに置く箱（箱押し、`sim.mujoco_env.BoxObject`）。
    @param enable    箱を置くか [-]（false なら MJCF のまま。既存の数値を変えない）
    @param half_size 半長 (x, y, z) [m]
    @param mass      質量 [kg]
    @param friction  摩擦 (slide, spin, roll) [-]
    @param pos       初期位置 {W} [x, y]（z は床に置く）[m]
    @param yaw       初期ヨー [rad]
    @note 手は箱とだけ当たる（手の衝突ビット: armtasks/boxpush/boxpush/model.py）。腕のある robot と組み合わせる。
    """

    enable: bool = False
    half_size: list = field(default_factory=lambda: [0.20, 0.30, 0.25])
    mass: float = 20.0
    friction: list = field(default_factory=lambda: [0.4, 0.005, 0.0001])
    pos: list = field(default_factory=lambda: [0.8, 0.0])
    yaw: float = 0.0

    def to_object(self):
        """! @brief `BoxObject` にする（enable = false なら None）。
        @return BoxObject | None
        """
        if not self.enable:
            return None
        from ..sim.mujoco_env import BoxObject

        return BoxObject(half_size=tuple(float(v) for v in self.half_size), mass=float(self.mass),
                         friction=tuple(float(v) for v in self.friction),
                         pos=(float(self.pos[0]), float(self.pos[1]), None), yaw=float(self.yaw))


@dataclass
class ScenarioSection:
    """! @brief 外乱シナリオ。
    @param kind "none" | "push"（胴体に力積）
    @param impulse 力積 {W} [N*s] shape (3,)
    @param t_push 印加時刻 [s]
    @param box   シーンに置く箱 [-]
    """

    kind: str = "none"
    impulse: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    t_push: float = 1.85
    box: BoxSection = field(default_factory=BoxSection)


@dataclass
class OutputSection:
    """! @brief 出力。
    @param log 結果 npz の保存先（None で保存しない）/ verbose 進捗表示 [-]
    """

    log: str | None = None
    verbose: bool = True


def _default_gait() -> GaitSection:
    """! @brief biped.yaml の walk_3d を既定にした GaitSection。
    @return GaitSection
    """
    d = load_yaml("biped.yaml").get("walk_3d", {}) or {}
    names = {f.name for f in fields(GaitSection)}
    return GaitSection(**{k: v for k, v in d.items() if k in names})


def _default_stabilizer() -> StabilizerParams:
    """! @brief biped.yaml の stabilizer 節を既定にした StabilizerParams（extra は空にする）。
    @return StabilizerParams
    """
    p = StabilizerParams.from_config()
    p.extra = {}
    return p


@dataclass
class BipedExperimentConfig:
    """! @brief 二足の実験設定（1 ファイル = 1 条件）。
    @param robot      ロボット設定ファイル名（configs/robot_*.yaml）、または YAML へのパス（実験設定ファイル基準の相対可）[-]
    @param gait       歩容 [-]
    @param planner    計画器 [-]
    @param stabilizer DCM 安定化と全身制御の共通ゲイン（`biped_stack.StabilizerParams`）[-]
    @param mapper     下層の制御 [-]
    @param policy     学習方策（下層、残差トルク）[-]
    @param hl_policy  HL 方策（上位、足配置・時間の残差）[-]
    @param sim        シミュレーション [-]
    @param scenario   外乱 [-]
    @param output     出力 [-]
    """

    robot: str = "robot_g1.yaml"
    gait: GaitSection = field(default_factory=_default_gait)
    planner: PlannerSection = field(default_factory=PlannerSection)
    stabilizer: StabilizerParams = field(default_factory=_default_stabilizer)
    mapper: MapperSection = field(default_factory=MapperSection)
    policy: PolicySection = field(default_factory=PolicySection)
    hl_policy: HlPolicySection = field(default_factory=HlPolicySection)
    sim: SimSection = field(default_factory=SimSection)
    scenario: ScenarioSection = field(default_factory=ScenarioSection)
    output: OutputSection = field(default_factory=OutputSection)
    ## 読み込んだファイル（ログ用。ファイルには書かない）
    config_path: Path | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        """! @brief 組み合わせの整合を検査する。
        @return なし
        """
        if self.mapper.kind not in ("pd", "ihmc", "dlr"):
            raise ValueError(f"mapper.kind は pd | ihmc | dlr: {self.mapper.kind}")
        if self.policy.kind not in ("none", "residual"):
            raise ValueError(f"policy.kind は none | residual: {self.policy.kind}")
        if self.policy.kind == "residual" and self.mapper.kind == "pd":
            raise ValueError("policy.kind = residual は mapper.kind が ihmc | dlr のときだけ使える")
        if self.planner.kind not in ("schedule", "online"):
            raise ValueError(f"planner.kind は schedule | online: {self.planner.kind}")
        if self.planner.kind == "online":
            if self.mapper.kind == "pd":
                raise ValueError("planner.kind = online は mapper.kind が ihmc | dlr のときだけ使える（pd は固定列専用）")
            if self.planner.online.source not in ("velocity", "stream"):
                raise ValueError(f"planner.online.source は velocity | stream: {self.planner.online.source}")
            if self.planner.online.source == "stream" and not self.planner.online.footsteps:
                raise ValueError("planner.online.source = stream には footsteps が必要")
        if self.hl_policy.kind not in ("none", "zero", "oracle", "mlp"):
            raise ValueError(f"hl_policy.kind は none | zero | oracle | mlp: {self.hl_policy.kind}")
        if self.hl_policy.kind != "none" and self.planner.kind != "online":
            raise ValueError("hl_policy は planner.kind = online のときだけ使える")
        if self.hl_policy.kind == "mlp" and not self.hl_policy.model_dir:
            raise ValueError("hl_policy.kind = mlp には model_dir が必要")
        if self.scenario.kind not in ("none", "push"):
            raise ValueError(f"scenario.kind は none | push: {self.scenario.kind}")
        if self.scenario.box.enable and self.mapper.kind == "pd":
            raise ValueError("scenario.box は mapper.kind が ihmc | dlr のときだけ使える")

    @property
    def label(self) -> str:
        """! @brief 1 行の要約 [-]"""
        pol = "" if self.policy.kind == "none" else f"+residual({'zero' if self.policy.model_dir is None else self.policy.model_dir})"
        if self.hl_policy.kind != "none":
            pol += f" hl={self.hl_policy.kind}"
        if self.planner.kind == "online":
            o = self.planner.online
            plan = (f"planner=online/{o.source} k_step={o.k_step:g} " +
                    (f"v=({o.vx:g},{o.vy:g}) " if o.source == "velocity" else f"n_fs={len(o.footsteps)} "))
        else:
            plan = f"planner=schedule n_step={self.gait.n_step} step={self.gait.step_length:g} m "
        return (f"mapper={self.mapper.kind}{pol} k_dcm={self.stabilizer.k_dcm:g} " + plan +
                f"scenario={self.scenario.kind} dt_ctrl={self.sim.dt_ctrl:g}")


# ======================================================================
# 読み込み
# ======================================================================
_SKIP_FIELDS = {("stabilizer", "extra"), (None, "config_path")}


def _strip_comments(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip_comments(v) for k, v in obj.items() if not str(k).startswith("_comment")}
    return obj


def _section_from_dict(dc_type, data: dict, section: str, default=None):
    """! @brief 1 セクションの dict を dataclass にする（ネスト、未知キー検査）。
    @param dc_type dataclass の型 [-]
    @param data    dict [-]
    @param section エラー表示用の名前 [-]
    @param default 既定インスタンス（None なら dc_type()）[-]
    @return dataclass インスタンス
    """
    default = dc_type() if default is None else default
    field_map = {f.name: f for f in fields(dc_type) if (section, f.name) not in _SKIP_FIELDS}
    kw = {f.name: getattr(default, f.name) for f in fields(dc_type)}
    for key, val in data.items():
        if key not in field_map:
            raise KeyError(f"[{section}] 未知のキー '{key}'（使えるキー: {sorted(field_map)}）")
        cur = getattr(default, key)
        if is_dataclass(cur) and isinstance(val, dict):
            kw[key] = _section_from_dict(type(cur), val, f"{section}.{key}", default=cur)
        elif isinstance(cur, bool):
            kw[key] = bool(val)
        elif isinstance(cur, int) and not isinstance(cur, bool) and val is not None:
            kw[key] = int(val)
        elif isinstance(cur, float) and val is not None:
            kw[key] = float(val)
        else:
            kw[key] = val
    return dc_type(**kw)


def experiment_config_from_dict(data: dict) -> BipedExperimentConfig:
    """! @brief dict → BipedExperimentConfig。
    @param data 読み込んだ dict [-]
    @return BipedExperimentConfig
    """
    data = _strip_comments(data or {})
    base = BipedExperimentConfig()
    top = {f.name: f for f in fields(BipedExperimentConfig) if (None, f.name) not in _SKIP_FIELDS}
    kw = {f.name: getattr(base, f.name) for f in fields(BipedExperimentConfig)}
    for key, val in data.items():
        if key not in top:
            raise KeyError(f"未知のキー '{key}'（使えるキー: {sorted(top)}）")
        cur = getattr(base, key)
        if is_dataclass(cur):
            if not isinstance(val, dict):
                raise TypeError(f"'{key}' はセクション（dict）で書く")
            kw[key] = _section_from_dict(type(cur), val, key, default=cur)
        else:
            kw[key] = val
    return BipedExperimentConfig(**kw)


def load_experiment_config(path: str | Path) -> BipedExperimentConfig:
    """! @brief YAML / JSON ファイル → BipedExperimentConfig。
    @param path ファイルパス（絶対、またはリポジトリ直下からの相対）[-]
    @return BipedExperimentConfig
    """
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = _REPO_ROOT / p
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        data = json.loads(text)
    elif p.suffix.lower() in (".yaml", ".yml"):
        data = yaml.safe_load(text)
    else:
        raise ValueError(f"拡張子は .yaml / .yml / .json: {p}")
    cfg = experiment_config_from_dict(data)
    cfg.config_path = p
    # robot がパス（区切りを含む）で相対なら、実験設定ファイルのあるディレクトリ基準で絶対化する
    # （別パッケージが自前の configs/ に robot_*.yaml と experiments/*.yaml を持つための口）
    r = str(cfg.robot)
    if ("/" in r or "\\" in r) and not Path(r).is_absolute():
        cand = (p.parent / r).resolve()
        if cand.is_file():
            cfg.robot = str(cand)
    return cfg


# ======================================================================
# 書き出し
# ======================================================================
def _to_jsonable(v: Any) -> Any:
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v


def _section_to_dict(obj, section: str | None) -> dict:
    out = {}
    for f in fields(obj):
        if (section, f.name) in _SKIP_FIELDS:
            continue
        v = getattr(obj, f.name)
        out[f.name] = _section_to_dict(v, f.name) if is_dataclass(v) else _to_jsonable(v)
    return out


def experiment_config_to_dict(cfg: BipedExperimentConfig, comment: str | None = None) -> dict:
    """! @brief BipedExperimentConfig → dict（ファイルに書ける形）。
    @param cfg     設定 [-]
    @param comment 先頭の _comment（None で既定の説明）[-]
    @return dict
    """
    d = {"_comment": comment or (
        "二足の実験設定の雛形（run_biped.py --config で読む）。編集して別名で保存する。"
        "単位: m, rad, s, N, N*m。書かなかった項目は既定（configs/biped.yaml）。"
        "_comment は無視、未知キーはエラー。mapper.kind = pd | ihmc | dlr、"
        "policy.kind = none | residual、scenario.kind = none | push")}
    d.update(_section_to_dict(cfg, None))
    return d


def dump_experiment_config(cfg: BipedExperimentConfig, path: str | Path, comment: str | None = None) -> Path:
    """! @brief 設定を YAML / JSON に書く（拡張子で切替）。
    @param cfg     設定 [-]
    @param path    書き出し先 [-]
    @param comment 先頭の説明 [-]
    @return 書いたパス
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    d = experiment_config_to_dict(cfg, comment)
    if p.suffix.lower() == ".json":
        p.write_text(json.dumps(d, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        p.write_text(yaml.safe_dump(d, allow_unicode=True, sort_keys=False, default_flow_style=False, width=100), encoding="utf-8")
    return p


# ======================================================================
# 表示
# ======================================================================
def _same(a: Any, b: Any) -> bool:
    try:
        return bool(np.all(np.asarray(a) == np.asarray(b)))
    except Exception:  # noqa: BLE001
        return a == b


def _walk(obj, default, indent: str, changed_only: bool, lines: list[str], section: str | None) -> None:
    for f in fields(obj):
        if (section, f.name) in _SKIP_FIELDS:
            continue
        v, d = getattr(obj, f.name), getattr(default, f.name)
        if is_dataclass(v):
            sub: list[str] = []
            _walk(v, d, indent + "  ", changed_only, sub, f.name)
            if sub or not changed_only:
                lines.append(f"{indent}{f.name}:")
                lines.extend(sub)
            continue
        changed = not _same(v, d)
        if changed_only and not changed:
            continue
        lines.append(f"{indent}{'*' if changed else ' '} {f.name} = {_to_jsonable(v)}")


def describe_config(cfg: BipedExperimentConfig, changed_only: bool = False, compact: bool = False) -> str:
    """! @brief 設定を人が読む文字列にする。
    @param cfg          設定 [-]
    @param changed_only 既定値から変えた項目だけ出す [-]
    @param compact      1 行の要約だけ [-]
    @return 文字列
    """
    if compact:
        return cfg.label
    default = BipedExperimentConfig()
    src = f"、設定ファイル {cfg.config_path}" if cfg.config_path else ""
    lines = [f"---- 実行条件（* = 既定値から変更{src}）  {cfg.label} ----"]
    _walk(cfg, default, "  ", changed_only, lines, None)
    lines.append("----")
    return "\n".join(lines)


# ======================================================================
# 実行
# ======================================================================
def _online_planner_setup(cfg: BipedExperimentConfig):
    """! @brief planner セクションから OnlinePlannerParams と毎周期フックを作る。
    @param cfg BipedExperimentConfig [-]
    @return (planner_params, on_step)。schedule のときは (None, None)
    @note on_step は velocity_schedule の時刻になったら set_command、stream なら最初の周期に
          footsteps を全部 push する（テレオペの逐次注入はこのフックを差し替えれば済む）。
    """
    if cfg.planner.kind != "online":
        return None, None
    from dataclasses import asdict

    from ..planner.biped_online import OnlinePlannerParams

    o = cfg.planner.online
    d = asdict(o)
    sched = sorted((float(r[0]), float(r[1]), float(r[2]), float(r[3]) if len(r) > 3 else 0.0) for r in d.pop("velocity_schedule"))
    footsteps = [(int(f), float(x), float(y)) for f, x, y in d.pop("footsteps")]
    d["lean_xy"] = tuple(float(v) for v in d["lean_xy"])
    pp = OnlinePlannerParams(**d)
    state = {"next": 0, "pushed": False}

    def on_step(planner, st, k):
        if not state["pushed"]:
            for f, x, y in footsteps:
                planner.push_footstep(f, np.array([x, y]))
            state["pushed"] = True
        t = k * cfg.sim.dt_ctrl
        while state["next"] < len(sched) and t >= sched[state["next"]][0]:
            _, vx, vy, wz = sched[state["next"]]
            planner.set_command(vx, vy, wz)
            state["next"] += 1

    hl_step = _hl_policy_on_step(cfg)
    if not sched and not footsteps:
        return pp, hl_step
    if hl_step is None:
        return pp, on_step

    def both(planner, st, k):
        on_step(planner, st, k)
        hl_step(planner, st, k)

    return pp, both


def _hl_policy_on_step(cfg: BipedExperimentConfig):
    """! @brief hl_policy セクションから毎周期フックを作る（none なら None）。
    @param cfg BipedExperimentConfig [-]
    @return on_step(planner, state, k) | None
    """
    h = cfg.hl_policy
    if h.kind == "none":
        return None
    from ..policy.biped_hl_policy import (
        HlPolicySpec, MlpHlPolicy, OracleHlPolicy, ZeroHlPolicy, make_hl_on_step,
    )

    spec = HlPolicySpec(scale_xy=h.scale_xy, scale_t=h.scale_t, decimation=h.decimation)
    if h.kind == "zero":
        pol = ZeroHlPolicy()
    elif h.kind == "oracle":
        pol = OracleHlPolicy(h.k_oracle, spec)
    else:
        pol = MlpHlPolicy(h.model_dir)
    return make_hl_on_step(pol, spec)


def run_experiment(cfg: BipedExperimentConfig, viewer: bool = False, realtime: bool = False,
                   command_source=None, duration: float | None = None,
                   hand_task=None, extra_on_step=None):
    """! @brief 設定どおりに 1 条件を実行する。
    @param cfg 設定 [-]
    @param viewer / realtime / command_source / duration  表示・実時間・テレオペ入力（CLI の --viewer --realtime --source）[-]
    @param hand_task     手先タスク（`control.hand_task.HandTask`、腕のある機体 × dlr）。全身制御器に渡す [-]
    @param extra_on_step 設定由来のフックの後に毎周期呼ぶ追加フック (planner, state, k) -> None
                         （別パッケージが手先目標などを書くための口）[-]
    @return BipedWalkResult
    @note mapper.kind = pd は `run_biped_walk`、ihmc / dlr（+ residual）は
          `run_biped_torque` を呼ぶ。評価指標は両者で同じ `summarize` を使う。
    """
    from dataclasses import asdict

    from .run_biped import run_biped_walk, save_result

    gait_ov = asdict(cfg.gait)
    if cfg.mapper.kind == "pd":
        res = run_biped_walk(
            robot=cfg.robot, gait_overrides=gait_ov, dt_ctrl=cfg.sim.dt_ctrl,
            kp_scale=cfg.mapper.pd.kp_scale, ik_iter=cfg.mapper.pd.ik_iter,
            use_ff=cfg.mapper.pd.use_ff, k_dcm=cfg.mapper.pd.k_dcm, verbose=cfg.output.verbose,
            viewer=viewer, realtime=realtime,
        )
    else:
        from .biped_torque_run import run_biped_torque

        par = dataclasses.replace(cfg.stabilizer)
        par.extra = {"ihmc": asdict(cfg.mapper.ihmc), "dlr": asdict(cfg.mapper.dlr)}
        planner_params, on_step = _online_planner_setup(cfg)
        planner_kind = cfg.planner.kind
        if command_source is not None:
            # テレオペ: 計画器を online + endless にし、速度指令は入力源から入る（velocity_schedule は無視）
            from ..planner.biped_online import OnlinePlannerParams

            base = planner_params if planner_params is not None else OnlinePlannerParams()
            planner_params = dataclasses.replace(base, source="velocity", endless=True,
                                                 duration=600.0 if duration is None else float(duration))
            planner_kind = "online"
            on_step = _hl_policy_on_step(cfg)
        if extra_on_step is not None:
            base_step = on_step

            def on_step(planner, st, k, _b=base_step, _e=extra_on_step):  # noqa: E306
                if _b is not None:
                    _b(planner, st, k)
                _e(planner, st, k)
        res = run_biped_torque(
            robot=cfg.robot, gait_overrides=gait_ov, dt_ctrl=cfg.sim.dt_ctrl,
            mapper="rl" if cfg.policy.kind == "residual" else cfg.mapper.kind,
            policy_dir=cfg.policy.model_dir, rl_base=cfg.mapper.kind,
            verbose=cfg.output.verbose, par=par, scenario=cfg.scenario, solref=cfg.sim.solref,
            tau_scale=cfg.policy.tau_scale,
            planner_kind=planner_kind, planner_params=planner_params, on_step=on_step,
            viewer=viewer, realtime=realtime, command_source=command_source,
            hand_task=hand_task,
        )
    if cfg.output.log:
        save_result(res, cfg.output.log)
    return res
