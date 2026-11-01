"""! @file plotstyle.py
@brief 図と動画で共通の色・名前・matplotlib の体裁（制御の色は固定: 学習 = 青、手書き前傾 = 緑、力で前傾 = 黄、直立 = 橙）。"""

from __future__ import annotations

BLUE, ORANGE, GREEN, YELLOW, PINK, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#7b61c9"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
POL_COLOR = {"rl": BLUE, "schedule": GREEN, "reactive": YELLOW, "upright": ORANGE, "analytic": GREEN,
             "P": BLUE, "P_S": PINK, "P_S_T": PURPLE, "A": BLUE, "B": PINK, "C": PURPLE, "legacy": "#4a4a48"}
POL_NAME = {"rl": "学習方策（f_push, lean_x）", "legacy": "学習方策（f_push, lean_x）",
            "schedule": "手書き（前傾、質量既知）", "reactive": "手書き（力に応じて前傾）",
            "upright": "手書き（直立）", "analytic": "手書き腕 + 脚の解析則",
            "P": "学習 P（計画層の残差）", "P_S": "学習 P_S（+ CoP 残差）", "P_S_T": "学習 P_S_T（+ トルク残差）",
            "A": "学習 A（腕だけ）", "B": "学習 B（+ CoP 残差）", "C": "学習 C（+ トルク残差）"}


def style() -> None:
    """! @brief matplotlib の体裁（日本語フォントは Noto Sans CJK → Meiryo → Yu Gothic の順に探す）。"""
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    have = {f.name for f in font_manager.fontManager.ttflist}
    fam = [f for f in ("Noto Sans CJK JP", "Meiryo", "Yu Gothic", "IPAexGothic") if f in have] + ["DejaVu Sans"]
    plt.rcParams.update({"font.family": fam, "font.size": 9,
                         "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
                         "legend.frameon": False, "legend.fontsize": 8})
