"""! @file terrain.py
@brief 地形の生成と、地面高さ関数。

## 何を解いているか

着地目標の高さや接触脚の高さを定数で置くと「地面は z=0 の平面」を暗黙に仮定することになる。
本モジュールはその定数を **地面高さ関数** `GroundHeightFn = Callable[[float, float], float]`
に置き換えるための地形定義と、対応する MJCF を生成する。

    h_ground(x, y) [m]   … 地面「表面」の高さ（{W}）。平地は恒等的に 0

足先中心の目標高さは `ground_z + h_ground(x, y)` とする。既定の
`flat_ground()`（常に 0）を渡せば、**平地での数値は定数の地面高さと完全に一致する**。

## 地形の種類

| kind | 実現方法 | 高さ関数 |
|---|---|---|
| `flat` | 追加なし | 0 |
| `step` | box 1 個 | x >= x0 で height |
| `stairs` | box n 個 | 段ごとに height ずつ上がる |
| `slope` | 回転した box 1 個 | tan(slope) * (x - x0) |
| `rough` | hfield（実行時に `model.hfield_data` を書く） | 疑似乱数の格子補間 |

**MJCF の形状と高さ関数が食い違うと、足が地面にめり込むか宙に浮く。**
両者の一致は `mujoco.mj_ray` による実測で確かめられる。

記号: {W} ワールド系、x 前方 / y 左 / z 上。
"""

from __future__ import annotations

import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

__all__ = [
    "GroundHeightFn",
    "TerrainSpec",
    "flat_ground",
    "height_function",
    "make_terrain_xml",
    "write_heightfield",
    "TERRAIN_KINDS",
]

## 地面高さ関数の型。(x, y) [m] -> 地面表面の高さ [m]
GroundHeightFn = Callable[[float, float], float]

TERRAIN_KINDS = ("flat", "step", "stairs", "slope", "rough")

_HFIELD_NAME = "q6_rough"


def flat_ground(x: float, y: float) -> float:
    """! @brief 平地の地面高さ関数（常に 0）。
    @param x 位置 x {W} [m]
    @param y 位置 y {W} [m]
    @return 地面高さ [m]（常に 0.0）
    @note **既定値。これを渡したときは平地（z = 0）の数値が完全に再現される。**
    """
    del x, y
    return 0.0


@dataclass
class TerrainSpec:
    """! @brief 地形の定義。

    @param kind         種類 "flat" | "step" | "stairs" | "slope" | "rough" [-]
    @param height       段差 1 段の高さ [m]（step / stairs）
    @param start_x      地形が始まる x 位置 {W} [m]
    @param n_steps      段数 [-]（stairs）
    @param step_length  1 段の奥行き [m]（stairs）
    @param length       地形の全長 [m]（step / slope / rough）
    @param width        地形の幅 [m]
    @param slope_deg    斜面の角度 [deg]（slope）
    @param roughness    粗さの振幅（山と谷の差）[m]（rough）
    @param cell         粗面の格子間隔 [m]（rough）
    @param seed         粗面の乱数種 [-]
    @note 単位はすべて m / deg。`kind="flat"` のとき他の項目は無視される。
    """

    kind: str = "flat"
    height: float = 0.15
    start_x: float = 1.0
    n_steps: int = 4
    step_length: float = 0.5
    length: float = 8.0
    width: float = 4.0
    slope_deg: float = 10.0
    roughness: float = 0.04
    cell: float = 0.25
    seed: int = 0
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """! @brief 種類名を検査する。
        @return なし
        """
        if self.kind not in TERRAIN_KINDS:
            raise ValueError(f"未知の地形種別: {self.kind}（{TERRAIN_KINDS} のいずれか）")

    @staticmethod
    def from_dict(d: dict | None) -> "TerrainSpec":
        """! @brief 辞書（YAML 由来）から作る。
        @param d 辞書（None なら平地）[-]
        @return TerrainSpec
        """
        if not d:
            return TerrainSpec()
        known = {f for f in TerrainSpec.__dataclass_fields__ if f != "extra"}
        kw = {k: v for k, v in d.items() if k in known}
        return TerrainSpec(**kw, extra={k: v for k, v in d.items() if k not in known})

    @property
    def is_flat(self) -> bool:
        """! @brief 平地かどうか [-]"""
        return self.kind == "flat"

    def label(self) -> str:
        """! @brief シナリオ表に出す短い名前。
        @return 文字列 [-]
        """
        if self.kind == "flat":
            return "flat"
        if self.kind == "step":
            return f"step{self.height * 100:.0f}cm"
        if self.kind == "stairs":
            return f"stairs{self.height * 100:.0f}cmx{self.n_steps}"
        if self.kind == "slope":
            return f"slope{self.slope_deg:.0f}deg"
        return f"rough{self.roughness * 100:.0f}cm"


# --------------------------------------------------------------------------
# 高さ関数
# --------------------------------------------------------------------------
def _rough_grid(spec: TerrainSpec) -> tuple[np.ndarray, float, float, float, float]:
    """! @brief 粗面の高さ格子を作る。
    @param spec TerrainSpec [-]
    @return (grid shape (nrow,ncol) [m], x0 [m], y0 [m], dx [m], dy [m])
    @note 乱数種を固定した一様乱数。**格子は MJCF の hfield と高さ関数で共有する**
          （別々に作ると必ず食い違う）。
    """
    x0 = spec.start_x
    y0 = -0.5 * spec.width
    ncol = max(int(round(spec.length / spec.cell)) + 1, 2)
    nrow = max(int(round(spec.width / spec.cell)) + 1, 2)
    rng = np.random.default_rng(int(spec.seed))
    grid = rng.uniform(0.0, 1.0, size=(nrow, ncol))
    dx = spec.length / (ncol - 1)
    dy = spec.width / (nrow - 1)
    return grid, x0, y0, dx, dy


def height_function(spec: TerrainSpec) -> GroundHeightFn:
    """! @brief 地形定義から地面高さ関数を作る。
    @param spec TerrainSpec [-]
    @return h(x, y) -> 地面高さ [m]
    @note MuJoCo の hfield は格子点を**双一次補間**するので、粗面の高さ関数も
          同じ補間を実装する。最近傍で近似すると段差状の誤差（最大 roughness）が残る。
    """
    if spec.kind == "flat":
        return flat_ground

    if spec.kind == "step":
        x0, h, xe = spec.start_x, spec.height, spec.start_x + spec.length
        w2 = 0.5 * spec.width

        def _step(x: float, y: float) -> float:
            return h if (x0 <= x <= xe and abs(y) <= w2) else 0.0

        return _step

    if spec.kind == "stairs":
        x0, h, n, sl = spec.start_x, spec.height, int(spec.n_steps), spec.step_length
        w2 = 0.5 * spec.width

        xe = x0 + spec.length

        def _stairs(x: float, y: float) -> float:
            # footprint の外は 0。地形は有限なので、踏破距離より長く取ること
            if x < x0 or x > xe or abs(y) > w2:
                return 0.0
            k = int(np.floor((x - x0) / sl)) + 1
            return h * float(min(max(k, 0), n))

        return _stairs

    if spec.kind == "slope":
        x0 = spec.start_x
        t = float(np.tan(np.deg2rad(spec.slope_deg)))
        xe = x0 + spec.length
        w2 = 0.5 * spec.width

        def _slope(x: float, y: float) -> float:
            if abs(y) > w2 or x < x0 or x > xe:
                return 0.0
            return t * (x - x0)

        return _slope

    grid, gx0, gy0, dx, dy = _rough_grid(spec)
    amp = spec.roughness
    nrow, ncol = grid.shape

    def _rough(x: float, y: float) -> float:
        u = (x - gx0) / dx
        v = (y - gy0) / dy
        if u < 0.0 or v < 0.0 or u > ncol - 1 or v > nrow - 1:
            return 0.0
        i0, j0 = int(np.floor(v)), int(np.floor(u))
        i1, j1 = min(i0 + 1, nrow - 1), min(j0 + 1, ncol - 1)
        fv, fu = v - i0, u - j0
        # MuJoCo は hfield の 1 セルを **(i,j)-(i+1,j+1) の対角線**で 2 枚の三角形に
        # 割る（本リポジトリで mj_ray により実測して確認した）。双一次補間で近似すると
        # 振幅の 1/4 程度（roughness 0.05 m で 12 mm）ずれるので、同じ分割を実装する。
        if fv < fu:  # 下側の三角形: (i0,j0), (i0,j1), (i1,j1)
            g = grid[i0, j0] * (1.0 - fu) + grid[i0, j1] * (fu - fv) + grid[i1, j1] * fv
        else:  # 上側の三角形: (i0,j0), (i1,j0), (i1,j1)
            g = grid[i0, j0] * (1.0 - fv) + grid[i1, j0] * (fv - fu) + grid[i1, j1] * fu
        return amp * float(g)

    return _rough


# --------------------------------------------------------------------------
# MJCF の生成
# --------------------------------------------------------------------------
def make_terrain_xml(src: Path, spec: TerrainSpec, out_dir: Path | None = None) -> Path | None:
    """! @brief 地形を追加した MJCF を書き出す。
    @param src     元の MJCF パス [-]
    @param spec    TerrainSpec [-]
    @param out_dir 出力先ディレクトリ（None ならシステムの一時ディレクトリ）[-]
    @return 生成した MJCF のパス（平地なら None = 元のまま使う）
    @note 平地で None を返すのは、**元ファイルをそのまま使わせて既存挙動との
          差をゼロにする**ためである（一時ファイル経由にすると、たとえ内容が同じでも
          相対パス解決やキーフレームの扱いで差が入り込む余地が生じる）。
    """
    if spec.is_flat:
        return None
    tree = ET.parse(src)
    root = tree.getroot()
    world = root.find("worldbody")
    if world is None:
        raise ValueError("MJCF に <worldbody> が無い")

    if spec.kind == "rough":
        _add_heightfield(root, world, spec)
    else:
        for i, (pos, size, rot_y) in enumerate(_terrain_boxes(spec)):
            # **euler ではなく quat で書く。** MJCF の <compiler angle> が
            # radian か degree かでオイラー角の解釈が変わり、角度単位を取り違えると
            # 斜面が -10 rad 回転する（実際に一度そうなった）。
            qw, qy = float(np.cos(0.5 * rot_y[1])), float(np.sin(0.5 * rot_y[1]))
            ET.SubElement(
                world,
                "geom",
                {
                    "name": f"q6_terrain_{i}",
                    "type": "box",
                    "pos": " ".join(f"{v:.6f}" for v in pos),
                    "size": " ".join(f"{v:.6f}" for v in size),
                    "quat": f"{qw:.9f} 0 {qy:.9f} 0",
                    "contype": "1",
                    "conaffinity": "1",
                    "condim": "3",
                    "friction": "0.8 0.02 0.001",
                    "rgba": "0.35 0.37 0.42 1",
                },
            )

    # 元ファイルと別のディレクトリに書くため、アセットの相対解決先を明示しておく。
    # 本リポジトリの MJCF は自己完結だが、menagerie 由来のモデルに差し替えたときに
    # メッシュ・テクスチャが見つからなくなるのを防ぐ。
    comp = root.find("compiler")
    if comp is None:
        comp = ET.Element("compiler")
        root.insert(0, comp)
    if "assetdir" not in comp.attrib and "meshdir" not in comp.attrib:
        comp.set("assetdir", str(src.parent))

    d_out = Path(tempfile.gettempdir()) if out_dir is None else Path(out_dir)
    d_out.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="_terrain_", suffix=".xml", dir=str(d_out))
    import os

    os.close(fd)
    out = Path(name)
    tree.write(out, encoding="utf-8", xml_declaration=False)
    return out


def _terrain_boxes(spec: TerrainSpec) -> list[tuple[tuple, tuple, tuple]]:
    """! @brief 地形を構成する box の (pos, size, euler) を返す。
    @param spec TerrainSpec [-]
    @return [(pos(3) [m], size(3) 半長 [m], rot(3) y 軸まわり回転角 [rad] を第 2 成分に持つ)]
    @note MuJoCo の box size は**半長**である。ここを全長にすると地形が 2 倍になる。
    """
    w2 = 0.5 * spec.width
    out: list[tuple[tuple, tuple, tuple]] = []

    if spec.kind == "step":
        h = spec.height
        out.append(
            (
                (spec.start_x + 0.5 * spec.length, 0.0, 0.5 * h),
                (0.5 * spec.length, w2, 0.5 * h),
                (0.0, 0.0, 0.0),
            )
        )
    elif spec.kind == "stairs":
        n = int(spec.n_steps)
        for k in range(n):
            h = spec.height * (k + 1)
            x_lo = spec.start_x + k * spec.step_length
            # 各段は地面まで届く厚みにする（宙に浮いた板にしない）
            out.append(
                (
                    (x_lo + 0.5 * spec.step_length, 0.0, 0.5 * h),
                    (0.5 * spec.step_length, w2, 0.5 * h),
                    (0.0, 0.0, 0.0),
                )
            )
        # 最上段の先の踊り場。高さ関数は最上段の高さを保つので、幾何側も伸ばす。
        # 無いと「関数は 0.30 m、実地形は 0 m」という食い違いになる。
        x_top = spec.start_x + n * spec.step_length
        landing = spec.length - n * spec.step_length
        if landing > 1.0e-6:
            h = spec.height * n
            out.append(
                (
                    (x_top + 0.5 * landing, 0.0, 0.5 * h),
                    (0.5 * landing, w2, 0.5 * h),
                    (0.0, 0.0, 0.0),
                )
            )
    elif spec.kind == "slope":
        a = float(np.deg2rad(spec.slope_deg))
        ll = spec.length / np.cos(a)  # 斜面に沿った長さ [m]
        t = 0.30  # 厚み [m]（地面に潜り込ませるための十分な値）
        # 上面 z = tan(a)(x - x0) の接線方向は (cos a, 0, sin a)、上向き法線は
        # (-sin a, 0, cos a)。box の局所 +z をこの法線に合わせるので、
        # 必要な回転は y 軸まわり -a である。中心は上面中心から法線方向に t/2 下げる。
        xc = spec.start_x + 0.5 * spec.length
        zc_top = 0.5 * spec.length * np.tan(a)
        n = np.array([-np.sin(a), 0.0, np.cos(a)])
        c = np.array([xc, 0.0, zc_top]) - 0.5 * t * n
        out.append(
            (
                (float(c[0]), 0.0, float(c[2])),
                (0.5 * float(ll), w2, 0.5 * t),
                (0.0, -a, 0.0),
            )
        )
    return out


def _add_heightfield(root: ET.Element, world: ET.Element, spec: TerrainSpec) -> None:
    """! @brief 粗面用の hfield アセットと geom を MJCF に追加する。
    @param root  MJCF のルート要素 [-]
    @param world <worldbody> 要素 [-]
    @param spec  TerrainSpec [-]
    @return なし
    @note hfield の中身（`model.hfield_data`）は読込後に `write_heightfield()` で書く。
          MJCF 側には格子サイズと外形だけを宣言する。
    """
    grid, x0, y0, _, _ = _rough_grid(spec)
    nrow, ncol = grid.shape
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    ET.SubElement(
        asset,
        "hfield",
        {
            "name": _HFIELD_NAME,
            "nrow": str(nrow),
            "ncol": str(ncol),
            # size = (radius_x, radius_y, elevation_z, base_z)
            "size": f"{0.5 * spec.length:.6f} {0.5 * spec.width:.6f} "
            f"{max(spec.roughness, 1e-4):.6f} 0.05",
        },
    )
    ET.SubElement(
        world,
        "geom",
        {
            "name": "q6_terrain_hfield",
            "type": "hfield",
            "hfield": _HFIELD_NAME,
            "pos": f"{x0 + 0.5 * spec.length:.6f} {y0 + 0.5 * spec.width:.6f} 0",
            "contype": "1",
            "conaffinity": "1",
            "condim": "3",
            "friction": "0.8 0.02 0.001",
            "rgba": "0.35 0.37 0.42 1",
        },
    )


def write_heightfield(model, spec: TerrainSpec) -> None:
    """! @brief 読み込み済みモデルの hfield_data に粗面の高さを書き込む。
    @param model mjModel [-]
    @param spec  TerrainSpec [-]
    @return なし
    @note `hfield_data` は [0,1] 正規化された値で、実高さは size[2] 倍される。
          `height_function()` と同じ格子・同じ乱数種を使うので両者は一致する。
    """
    if spec.kind != "rough":
        return
    import mujoco

    hid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, _HFIELD_NAME)
    if hid < 0:
        raise KeyError(f"hfield '{_HFIELD_NAME}' がモデルに無い")
    grid, _, _, _, _ = _rough_grid(spec)
    nrow = int(model.hfield_nrow[hid])
    ncol = int(model.hfield_ncol[hid])
    if (nrow, ncol) != grid.shape:
        raise ValueError(f"hfield の格子が不一致: モデル {(nrow, ncol)} / 仕様 {grid.shape}")
    adr = int(model.hfield_adr[hid])
    model.hfield_data[adr : adr + nrow * ncol] = grid.astype(np.float32).ravel()
