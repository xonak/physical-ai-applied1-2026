"""! @file config.py
@brief configs/*.yaml を dataclass として読み込む。

方針:
- **コード中にマジックナンバーを書かない。** 物理定数・ゲイン・歩容パラメータは
  すべて configs/*.yaml に置き、ここで型付きに読む。
- 既定の探索パスはリポジトリ直下の configs/。環境変数 LOCOCTRL_CONFIG_DIR で上書き可。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

__all__ = [
    "config_dir",
    "asset_dir",
    "load_yaml",
    "RobotConfig",
    "resolve_robot_name",
    "load_robot_config",
]

_REPO_ROOT = Path(__file__).resolve().parents[2]


def config_dir() -> Path:
    """! @brief 設定ディレクトリを返す。
    @return configs/ の絶対パス
    """
    env = os.environ.get("LOCOCTRL_CONFIG_DIR")
    return Path(env).resolve() if env else _REPO_ROOT / "configs"


def asset_dir() -> Path:
    """! @brief モデル（MJCF/URDF）ディレクトリを返す。
    @return assets/ の絶対パス
    """
    env = os.environ.get("LOCOCTRL_ASSET_DIR")
    return Path(env).resolve() if env else _REPO_ROOT / "assets"


def load_yaml(name: str) -> dict[str, Any]:
    """! @brief configs/ 直下の YAML を読み込む。
    @param name ファイル名（例 "robot_g1.yaml"）[-]
    @return 辞書
    """
    path = config_dir() / name
    if not path.is_file():
        raise FileNotFoundError(f"設定ファイルが見つからない: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class RobotConfig:
    """! @brief ロボット固有の定数。

    @param name          機体名 [-]
    @param mjcf_path     MJCF ファイルパス [-]
    @param backend       動力学バックエンド "mujoco" | "pinocchio" [-]
    @param leg_names     脚名（インデックス順） [-]
    @param joint_names   関節名 shape (n_leg, 3) [-]（脚ごと hip/thigh/calf 順）
    @param foot_sites    足先 site 名（脚順） [-]
    @param foot_geoms    足先 geom 名（脚順） [-]
    @param base_body     ベースボディ名 [-]
    @param q_nominal     公称関節角 shape (n,) [rad]
    @param h_nominal     公称胴体高さ [m]
    @param foot_radius   足先球半径 [m]
    @param mu            床面摩擦係数 [-]
    @param f_min         接触力 z 下限 [N]
    @param f_max         接触力 z 上限 [N]
    @param tau_max       関節トルク上限 shape (n,) [N*m]
    @param extra         その他（機体固有の追加項目。二足の足裏寸法など）[-]
    @param arm_joint_names 腕の関節名 shape (n_arm, k) [-]（箱押し用 G1 など。無ければ空）。
                         自前の関節順は「脚（joint_names）→ 腕」。q_nominal / tau_max もこの順で並べる
    @param hand_sites    手先 site 名（腕順）[-]
    @param config_dir_of 設定ファイルのあったディレクトリ [-]（パスで読んだときだけ。相対 mjcf_path の基準）
    """

    name: str
    mjcf_path: str
    backend: str
    leg_names: list[str]
    joint_names: list[list[str]]
    foot_sites: list[str]
    foot_geoms: list[str]
    base_body: str
    q_nominal: np.ndarray
    h_nominal: float
    foot_radius: float
    mu: float
    f_min: float
    f_max: float
    tau_max: np.ndarray
    extra: dict[str, Any] = field(default_factory=dict)
    arm_joint_names: list[list[str]] = field(default_factory=list)
    hand_sites: list[str] = field(default_factory=list)
    config_dir_of: str | None = None

    @property
    def n_leg(self) -> int:
        """! @brief 脚数 nf [-]"""
        return len(self.leg_names)

    @property
    def n_joint(self) -> int:
        """! @brief 駆動関節数 n [-]"""
        return sum(len(j) for j in self.joint_names) + self.n_arm_joint

    @property
    def n_arm(self) -> int:
        """! @brief 腕の本数 [-]"""
        return len(self.arm_joint_names)

    @property
    def n_arm_joint(self) -> int:
        """! @brief 腕の関節数（全腕の合計）[-]"""
        return sum(len(j) for j in self.arm_joint_names)

    @property
    def n_leg_joint(self) -> int:
        """! @brief 脚の関節数（全脚の合計）[-]。自前順で 0..n_leg_joint-1 が脚、残りが腕"""
        return sum(len(j) for j in self.joint_names)

    @property
    def all_joint_names(self) -> list[str]:
        """! @brief 自前順の全駆動関節名（脚 → 腕）[-]"""
        return [j for leg in self.joint_names for j in leg] + [j for arm in self.arm_joint_names for j in arm]

    def mjcf_abspath(self) -> Path:
        """! @brief MJCF の絶対パスを返す。
        @return Path
        """
        p = Path(self.mjcf_path)
        if p.is_absolute():
            return p
        if self.config_dir_of is not None:
            cand = (Path(self.config_dir_of) / p).resolve()
            if cand.is_file():
                return cand
        cand = _REPO_ROOT / p
        return cand if cand.is_file() else asset_dir() / p.name


def resolve_robot_name(name: str) -> str:
    """! @brief ロボット設定の短縮名を実ファイル名へ正規化する。
    @param name "g1" / "robot_g1" / "robot_g1.yaml" のいずれか [-]
    @return configs/ 直下のファイル名
    @note CLI で `--robot g1` と書けるようにするための糖衣である。
          ファイル名（"robot_g1.yaml"）を渡してもそのまま通る。
    """
    n = str(name)
    if not n.endswith(".yaml"):
        n += ".yaml"
    if not n.startswith("robot_"):
        n = "robot_" + n
    return n


def _is_path_like(name: str) -> bool:
    """! @brief 設定名がファイルパスか（ディレクトリ区切りを含むか）。
    @param name 設定名 [-]
    @return bool
    """
    return "/" in str(name) or "\\" in str(name)


def load_robot_config(name: str = "robot_g1.yaml") -> RobotConfig:
    """! @brief ロボット設定を読む。
    @param name ファイル名（configs/ 直下）、または YAML ファイルへのパス [-]
    @return RobotConfig
    @note パス（区切りを含む）を与えたときは configs/ を見ずにそのファイルを読む。相対パスは
          カレントディレクトリ基準。別パッケージ（armtasks/boxpush など）が自前の設定を持つための口で、
          その設定の `mjcf_path` は設定ファイルのあるディレクトリ基準で解決する（`config_dir_of`）。
    """
    if _is_path_like(name):
        path = Path(name).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"設定ファイルが見つからない: {path}")
        with path.open("r", encoding="utf-8") as f:
            d = yaml.safe_load(f) or {}
        d.setdefault("config_dir_of", str(path.parent))
    else:
        d = load_yaml(resolve_robot_name(name))
    return RobotConfig(
        name=d["name"],
        mjcf_path=d["mjcf_path"],
        backend=d.get("backend", "mujoco"),
        leg_names=list(d["leg_names"]),
        joint_names=[list(j) for j in d["joint_names"]],
        foot_sites=list(d["foot_sites"]),
        foot_geoms=list(d["foot_geoms"]),
        base_body=d["base_body"],
        q_nominal=np.asarray(d["q_nominal"], dtype=np.float64),
        h_nominal=float(d["h_nominal"]),
        foot_radius=float(d["foot_radius"]),
        mu=float(d["mu"]),
        f_min=float(d["f_min"]),
        f_max=float(d["f_max"]),
        tau_max=np.asarray(d["tau_max"], dtype=np.float64),
        arm_joint_names=[list(j) for j in d.get("arm_joint_names", []) or []],
        hand_sites=list(d.get("hand_sites", []) or []),
        config_dir_of=d.get("config_dir_of"),
        extra={
            k: v
            for k, v in d.items()
            if k
            not in {
                "name", "mjcf_path", "backend", "leg_names", "joint_names", "foot_sites",
                "foot_geoms", "base_body", "q_nominal", "h_nominal", "foot_radius",
                "mu", "f_min", "f_max", "tau_max", "arm_joint_names", "hand_sites", "config_dir_of",
            }
        },
    )
