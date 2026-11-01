"""! @file plot_eval.py
@brief 評価結果（eval_policy の JSONL）の主図: 上段 成功率 × 箱の質量、下段 前傾（成功したエピソードの平均）× 箱の質量。

```bash
python tools/plot_eval.py ../../outputs/eval_main.jsonl ../../outputs/figs/eval_main.png
```
2 つの量は単位が違うので、1 つの軸に重ねず上下 2 段に分ける。方策の色は固定順（並び替えで色が変わらない）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#7b61c9"]   # 方策の色（固定順、既定パレットの 1〜6 番）
MARK = ["o", "s", "^", "D", "v", "P"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
NAMES = {"rl": "学習方策", "upright": "手書き（直立）", "schedule": "手書き（前傾、質量既知）", "reactive": "手書き（力に応じて前傾）",
         "analytic": "Analytic（手書き腕 + 脚の解析則）"}


def main(src: str, dst: str) -> None:
    rows = [json.loads(l) for l in Path(src).read_text(encoding="utf-8").splitlines() if l.strip()]
    pols = list(dict.fromkeys(r["policy"] for r in rows))
    masses = sorted({r["mass"] for r in rows})
    plt.rcParams.update({"font.family": ["Noto Sans CJK JP", "DejaVu Sans"], "font.size": 10,
                         "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2})
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.2, 6.4), dpi=150, sharex=True, gridspec_kw={"height_ratios": [1, 1]})
    fig.patch.set_facecolor(SURF)
    for ax in (a1, a2):
        ax.set_facecolor(SURF); ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    # 色は方策に固定（並びや本数で塗り替えない）: 直立 = 橙、前傾 / 解析則 = 緑、力で前傾 = 黄。
    # 学習方策は出てきた順に 青 → 桃 → 紫（条件 A / B / C を並べる想定）
    fixed = {"upright": 1, "schedule": 2, "reactive": 3, "analytic": 2}
    n_learned = 0
    for p in pols:
        if p in fixed:
            i = fixed[p]
        else:
            i = (0, 4, 5)[min(n_learned, 2)]
            n_learned += 1
        c, mk = CAT[i % len(CAT)], MARK[i % len(MARK)]
        sr, ln = [], []
        for m in masses:
            rr = [r for r in rows if r["policy"] == p and r["mass"] == m]
            sr.append(100.0 * np.mean([r["success"] for r in rr]) if rr else np.nan)
            ok = [r["lean_mean"] for r in rr if r["success"] and r["lean_mean"] is not None]
            ln.append(np.mean(ok) if ok else np.nan)
        name = NAMES.get(p, p)
        a1.plot(masses, sr, color=c, lw=2, marker=mk, ms=8, mec=SURF, mew=1.5, label=name)
        a2.plot(masses, ln, color=c, lw=2, marker=mk, ms=8, mec=SURF, mew=1.5, label=name)
    xs = np.linspace(min(masses) - 2, max(masses) + 2, 10)
    a2.plot(xs, 1.15 * 0.2 * xs * 9.81 / (33.34 * 3.9 ** 2), ls="--", lw=1.2, color=INK2, zorder=1)
    a2.text(xs[-1], 1.15 * 0.2 * xs[-1] * 9.81 / (33.34 * 3.9 ** 2), " F/(m ω²)", color=INK2, fontsize=8, va="center")
    a1.set_ylabel("成功率 [%]"); a1.set_ylim(-5, 105)
    a2.set_ylabel("前傾 lean_x [m]（成功時の平均）")
    a2.set_xlabel("箱の質量 [kg]"); a2.set_xticks(masses)
    a1.set_title("押しながら歩く（0.1 m/s、10 s、成功 = 転倒なしで箱 0.5 m 以上）", color=INK, fontsize=11, loc="left")
    a1.legend(loc="lower left", frameon=False, fontsize=8.5)
    a1.set_xlim(min(masses) - 3, max(masses) + 5)
    fig.tight_layout()
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dst, facecolor=SURF)
    print("書き出し:", dst)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
