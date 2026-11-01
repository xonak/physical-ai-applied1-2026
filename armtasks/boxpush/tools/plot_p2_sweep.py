"""! @file plot_p2_sweep.py
@brief `tools/sweep_p2.py` の掃引（質量 × 前傾 × 重心高さ、押しながら歩く、手書きスケジュール）を図にする。

```bash
python tools/plot_p2_sweep.py ../../outputs/p2_sweep.jsonl ../../outputs/figs/p2_sweep.png
```
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

GOOD, CRIT = "#0ca30c", "#d03b3b"          # 状態色（成功 / 転倒）。形とラベルも併用する
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def main(src: str, dst: str) -> None:
    rows = [json.loads(l) for l in Path(src).read_text(encoding="utf-8").splitlines() if l.strip()]
    plt.rcParams.update({"font.family": ["Noto Sans CJK JP", "DejaVu Sans"], "font.size": 10,
                         "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2})
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
    fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
    ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    # 外力に釣り合う前傾 F/(m ω²)、F = 1.15 μ m_box g
    m_robot, omega, mu = 33.34, 3.90, 0.2
    xs = [5, 45]
    ax.plot(xs, [1.15 * mu * x * 9.81 / (m_robot * omega ** 2) for x in xs], ls="--", lw=1.5, color=INK2, zorder=1)
    ax.text(35.5, 1.15 * mu * 35.5 * 9.81 / (m_robot * omega ** 2) - 0.012, "外力に釣り合う前傾 F / (m ω²)", color=INK2,
            fontsize=8.5, ha="left", va="top", rotation=0)
    for r in rows:
        ok, squat = r["success"], r["dz_com"] < 0
        x = r["mass"] - (1.6 if squat else 0.0)
        y = r["lean_x"]
        if ok:
            ax.scatter(x, y, s=90, marker="o", color=GOOD, edgecolor=SURF, linewidth=2, zorder=3)
        else:
            ax.scatter(x, y, s=90, marker="v" if squat else "X", color=CRIT, edgecolor=SURF, linewidth=1.5, zorder=3)
        lab = f"{r['box_dx']:.2f} m" if ok else f"転倒 {r['t_end']:.1f} s"
        if squat:
            ax.annotate(lab, (x, y), xytext=(-8, -3), textcoords="offset points", fontsize=7.5, color=INK2, ha="right")
        else:
            ax.annotate(lab, (x, y), xytext=(7, -3), textcoords="offset points", fontsize=7.5, color=INK2)
    ax.set_xlabel("箱の質量 [kg]（箱-床の摩擦 0.2、押付け力 = 静止摩擦 × 1.15）")
    ax.set_ylabel("前傾 lean_x [m]（DCM 参照の CMP ずらし）")
    ax.set_xticks([10, 20, 30, 40]); ax.set_xlim(5, 46); ax.set_ylim(-0.015, 0.235)
    ax.set_title("押しながら歩く（0.1 m/s、10 s）: 手書きスケジュールの成否", color=INK, fontsize=11, loc="left")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color=GOOD, ms=8, label="成功（ラベル = 箱の移動）"),
         Line2D([], [], marker="X", ls="", color=CRIT, ms=8, label="転倒（ラベル = 転倒時刻）"),
         Line2D([], [], marker="v", ls="", color=CRIT, ms=8, label="しゃがみ 6 cm を加えて転倒")]
    ax.legend(handles=h, loc="upper left", frameon=False, fontsize=8.5)
    fig.tight_layout()
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dst, facecolor=SURF)
    print("書き出し:", dst)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
