"""! @file plot_trace.py
@brief `boxpush.cli.trace_policy` の記録から図を作る。入力（方策の出力）とロボットの状態を時系列で見る。

```bash
cd armtasks/boxpush
python tools/plot_trace.py ../../outputs/trace
```
書き出し（記録と同じフォルダの figs/）:
- `<label>_<mass>kg.png` : 1 条件の詳細。上 3 段が入力（方策の出力）、下 7 段がロボットの状態
- `compare_<mass>kg.png`  : 同じ質量で制御（学習 / 手書き）を重ねた比較
- `summary.json`          : 条件ごとの要約（閲覧ページの表に使う）

単位の違う量は 1 つの軸に重ねない（段を分ける）。制御の色は固定（学習 = 青、手書き前傾 = 緑、力で前傾 = 黄、直立 = 橙）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from boxpush.plotstyle import (  # noqa: E402
    BLUE, GREEN, GRID, INK2, ORANGE, PINK, POL_COLOR, POL_NAME, PURPLE, SURF, YELLOW, style,
)

M_ROBOT, OMEGA, T_WALK = 33.34, 3.90, 2.0
## 凡例は図の右外に置く（線に重ねない）
LEG = dict(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0, handlelength=1.8)


def _style() -> None:
    style()


def _ax(ax) -> None:
    ax.set_facecolor(SURF)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def load(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files if k != "meta"}
    d["meta"] = json.loads(str(z["meta"]))
    return d


def status_of(m: dict) -> str:
    return "成功" if m["success"] else ("転倒" if m["fell"] else ("箱が倒れた" if m.get("box_tipped") else "未達"))


def summarize(d: dict) -> dict:
    """! @brief 1 条件の要約（押している区間 t ≥ 3 s の平均など）。"""
    m, t = d["meta"], d["t"]
    on = (t >= 3.0) & (t <= t[-1] - 0.2)
    if not on.any():
        on = t >= 0
    first = lambda x, thr: float(t[np.argmax(x > thr)]) if np.any(x > thr) else None  # noqa: E731
    return {
        "label": m["label"], "mass": m["mass"], "mu": m["mu"], "status": status_of(m), "t_end": m["t_end"],
        "box_dx": m["box_dx"], "static_friction_N": m["static_friction_N"],
        "t_push_start": first(d["f_cmd"], 5.0), "t_lean_start": first(d["lean"], 0.01),
        "f_cmd_mean": round(float(d["f_cmd"][on].mean()), 1), "f_meas_mean": round(float(d["f_meas"][on].mean()), 1),
        "lean_mean_cm": round(100 * float(d["lean"][on].mean()), 1),
        "lean_y_mean_cm": round(100 * float(d["lean_y"][on].mean()), 1) if "lean_y" in d else None,
        "z_c_cmd_mean": round(float(d["z_c_cmd"][on].mean()), 3) if "z_c_cmd" in d else None,
        "step_res_mean_cm": [round(100 * float(v), 1) for v in d["step_res"][on].mean(axis=0)] if "step_res" in d else None,
        "lean_balance_cm": round(100 * float(d["f_meas"][on].mean()) / (M_ROBOT * OMEGA ** 2), 1),
        "v_box_mean": round(float(d["v_box"][on].mean()), 3), "v_box_std": round(float(d["v_box"][on].std()), 3),
        "face_c_mean": round(float(d["face_c"][on].mean()), 3), "face_c_std": round(float(d["face_c"][on].std()), 3),
        "pitch_absmax": round(float(np.abs(d["pitch"]).max()), 1),
        "e_dcm_rms_cm": round(100 * float(np.sqrt(np.mean(d["e_dcm"][on] ** 2))), 2),
        "da_mean": [round(float(v), 3) for v in np.abs(np.diff(d["a"], axis=0)).mean(axis=0)] if len(d["a"]) > 1 else None,
    }


def _events(ax, m: dict) -> None:
    ax.axvline(T_WALK, color=INK2, lw=0.8, ls=":")
    if m["fell"] or m.get("box_tipped"):
        ax.axvline(m["t_end"], color=ORANGE, lw=1.0, ls="--")


def plot_one(d: dict, dst: Path) -> None:
    """! @brief 1 条件の詳細図: 入力 3 段 + 状態 7 段。"""
    m, t = d["meta"], d["t"]
    td = d["k_dec"] * 0.002
    a = d["a"]
    plan = any(k in m["actions"] for k in ("cmp_dx", "cmp_dy", "com_dz", "step_dx"))
    n_in = 4 if plan else 3
    fig, axs = plt.subplots(7 + n_in, 1, figsize=(10.0, 15.5 + (1.6 if plan else 0)), dpi=130, sharex=True,
                            gridspec_kw={"height_ratios": [1, 1] + ([0.9] if plan else []) + [0.8, 1, 0.8, 0.8, 0.8, 1, 0.8, 0.8],
                                         "hspace": 0.42})
    fig.subplots_adjust(top=0.955, right=0.74, left=0.08)
    fig.patch.set_facecolor(SURF)
    for ax in axs:
        _ax(ax); _events(ax, m)
    name = POL_NAME.get(m["label"], m["label"])
    fig.suptitle(f"{name}　箱 {m['mass']:.0f} kg（摩擦 {m['mu']:.2f}、静止摩擦 {m['static_friction_N']:.0f} N）"
                 f"　→ {status_of(m)}、箱の移動 {m['box_dx']:.2f} m", x=0.08, y=0.985, ha="left", fontsize=11,
                 fontweight="bold")
    # ---- 入力（方策の出力）
    ax = axs[0]
    ax.step(t, d["f_tgt"], where="post", color=INK2, lw=1.0, label="方策の出力")
    ax.plot(t, d["f_cmd"], color=BLUE, lw=2.0, label="手先タスクへの指令（変化率制限後）")
    ax.axhline(m["static_friction_N"], color=ORANGE, lw=1.0, ls="--", label="静止摩擦 μmg")
    ax.set_title("入力 1　押付け力 [N]"); ax.legend(**LEG)
    ax = axs[1]
    if np.all(np.isfinite(d["lean_tgt"])):
        ax.step(t, 100 * d["lean_tgt"], where="post", color=INK2, lw=1.0, label="方策の出力 x")
    ax.plot(t, 100 * d["lean"], color=BLUE, lw=2.0, label="計画器への指令 x（前が正）")
    if "lean_y" in d and np.any(d["lean_y"]):
        ax.plot(t, 100 * d["lean_y"], color=GREEN, lw=1.4, label="計画器への指令 y（左が正）")
    ax.set_title("入力 2　CMP 参照のオフセット [cm]"); ax.legend(**LEG)
    k_ax = 2
    if plan:
        ax = axs[2]
        ax.plot(t, 100 * (d["z_c_cmd"] - d["z_c_cmd"][0]), color=BLUE, lw=2.0, label="重心高さ指令（初期比）[cm]")
        sr = d["step_res"]
        ax.plot(t, 100 * sr[:, 0], color=GREEN, lw=1.2, label="足配置残差 dx [cm]")
        ax.plot(t, 100 * sr[:, 1], color=YELLOW, lw=1.2, label="足配置残差 dy [cm]")
        ax.plot(t, 100 * sr[:, 2], color=ORANGE, lw=1.0, ls="--", label="単脚支持時間残差 dt [10 ms]")
        ax.set_title("入力 3　計画層の残差（重心高さ、足配置）"); ax.legend(**LEG)
        k_ax = 3
    ax = axs[k_ax]
    cols = (BLUE, ORANGE, GREEN, YELLOW, PINK, PURPLE, INK2, "#8c6d31")
    for i, nm in enumerate(m["actions"]):
        if i < a.shape[1] and i < len(cols):
            ax.step(td, a[:, i], where="post", color=cols[i], lw=1.2, label=nm)
    ax.set_ylim(-1.1, 1.1); ax.set_title(f"入力 {k_ax + 1}　生の行動 a ∈ [-1, 1]（{500.0 / m['decimation']:.0f} Hz、先頭 8 つ）")
    ax.legend(**LEG)
    axs = axs[k_ax + 1:]
    # ---- 状態
    ax = axs[0]
    ax.plot(t, d["f_meas"], color=BLUE, lw=2.0, label="両手の合計")
    ax.plot(t, d["f_L"], color=GREEN, lw=1.0, label="左手")
    ax.plot(t, d["f_R"], color=YELLOW, lw=1.0, label="右手")
    ax.plot(t, d["f_cmd"], color=INK2, lw=1.0, ls="--", label="指令")
    ax.set_title("状態 1　押付け力の実測（手首 F/T）[N]"); ax.legend(**LEG)
    ax = axs[1]
    ax.plot(t, d["box_dx"], color=BLUE, lw=2.0)
    ax.axhline(0.5, color=INK2, lw=0.8, ls="--")
    ax.set_title("状態 2　箱の移動 [m]（破線 = 成功判定 0.5 m）")
    ax = axs[2]
    ax.plot(t, d["v_box"], color=BLUE, lw=1.6)
    ax.axhline(m.get("vx", 0.1), color=INK2, lw=0.8, ls="--")
    ax.set_title("状態 3　箱の速度 [m/s]（破線 = 歩行の指令速度）")
    ax = axs[3]
    ax.plot(t, d["face_c"], color=BLUE, lw=1.6, label="箱の面 − 重心")
    ax.plot(t, d["reach"], color=GREEN, lw=1.2, label="手先 − 骨盤")
    ax.axhline(0.47, color=ORANGE, lw=0.8, ls="--", label="リーチ制限")
    ax.set_title("状態 4　箱との距離 [m]"); ax.legend(**LEG)
    ax = axs[4]
    ax.plot(t, d["c_x"], color=BLUE, lw=2.0, label="重心 x")
    fx = d["foot_x"]
    ax.plot(t, fx[:, 0], color=GREEN, lw=1.0, label="左足 x")
    ax.plot(t, fx[:, 1], color=YELLOW, lw=1.0, label="右足 x")
    ax.plot(t, d["c_z"], color=PURPLE, lw=1.2, label="重心 z")
    ax.set_title("状態 5　重心と足の前後位置、重心高さ [m]"); ax.legend(**LEG)
    ax = axs[5]
    ax.plot(t, d["pitch"], color=BLUE, lw=1.6, label="ピッチ（前が正）")
    ax.plot(t, d["roll"], color=GREEN, lw=1.0, label="ロール")
    ax.set_title("状態 6　骨盤の姿勢 [deg]"); ax.legend(**LEG)
    ax = axs[6]
    ax.plot(t, 100 * d["e_dcm"], color=BLUE, lw=1.4, label="DCM 誤差 [cm]")
    ax.plot(t, 100 * d["tau_sat"], color=ORANGE, lw=1.0, label="トルク飽和率 [%]")
    ax.set_title("状態 7　安定化の余裕"); ax.legend(**LEG)
    axs[-1].set_xlabel("時刻 [s]（点線 = 歩き出し）")
    dst.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dst, facecolor=SURF)
    plt.close(fig)


def plot_compare(ds: list[dict], dst: Path) -> None:
    """! @brief 同じ質量で制御を重ねる（入力 2 段 + 状態 5 段）。"""
    m0 = ds[0]["meta"]
    rows = [("f_cmd", 1, "入力　押付け力の指令 [N]"), ("lean", 100, "入力　前傾 [cm]"),
            ("c_z", 1, "状態　重心高さ [m]"),
            ("f_meas", 1, "状態　押付け力の実測 [N]"), ("box_dx", 1, "状態　箱の移動 [m]"),
            ("v_box", 1, "状態　箱の速度 [m/s]"), ("face_c", 1, "状態　箱の面 − 重心 [m]"), ("pitch", 1, "状態　ピッチ [deg]")]
    fig, axs = plt.subplots(len(rows), 1, figsize=(10.0, 13.0), dpi=130, sharex=True, gridspec_kw={"hspace": 0.38})
    fig.subplots_adjust(top=0.95, right=0.74, left=0.08)
    fig.patch.set_facecolor(SURF)
    fig.suptitle(f"箱 {m0['mass']:.0f} kg（摩擦 {m0['mu']:.2f}）: 制御の比較", x=0.08, y=0.985, ha="left",
                 fontsize=11, fontweight="bold")
    for ax, (key, sc, title) in zip(axs, rows):
        _ax(ax)
        ax.axvline(T_WALK, color=INK2, lw=0.8, ls=":")
        for d in ds:
            m = d["meta"]
            c = POL_COLOR.get(m["label"], PURPLE)
            y = d[key] * sc
            if key == "f_meas":   # 実測は 0.3 s の移動平均（突き・離れのぎざぎざを均す）
                y = np.convolve(y, np.ones(15) / 15, mode="same")
            ax.plot(d["t"], y, color=c, lw=1.8 if m["label"] == "rl" else 1.3,
                    label=f"{POL_NAME.get(m['label'], m['label'])}\n（{status_of(m)}、{m['box_dx']:.2f} m）")
            if m["fell"]:
                ax.plot(d["t"][-1], y[-1], "x", color=c, ms=8, mew=2)
        ax.set_title(title)
    axs[0].axhline(m0["static_friction_N"], color=INK2, lw=0.8, ls="--")
    axs[4].axhline(0.5, color=INK2, lw=0.8, ls="--")
    axs[5].axhline(ds[0]["meta"].get("vx", 0.1), color=INK2, lw=0.8, ls="--")
    axs[0].legend(**LEG)
    axs[-1].set_xlabel("時刻 [s]（点線 = 歩き出し、× = 転倒）")
    fig.savefig(dst, facecolor=SURF)
    plt.close(fig)


def main(src: str) -> None:
    _style()
    src_p = Path(src)
    files = sorted(src_p.glob("*_*kg.npz"))
    if not files:
        raise SystemExit(f"{src_p} に *_<mass>kg.npz が無い（boxpush.cli.trace_policy で作る）")
    figs = src_p / "figs"
    data = [load(f) for f in files]
    summary = []
    for f, d in zip(files, data):
        plot_one(d, figs / f"{f.stem}.png")
        summary.append(summarize(d))
        print(f"  {f.stem}.png")
    for mass in sorted({d["meta"]["mass"] for d in data}):
        ds = [d for d in data if d["meta"]["mass"] == mass]
        order = {k: i for i, k in enumerate(POL_COLOR)}
        ds.sort(key=lambda d: order.get(d["meta"]["label"], 99))
        plot_compare(ds, figs / f"compare_{mass:.0f}kg.png")
        print(f"  compare_{mass:.0f}kg.png")
    (src_p / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"書き出し: {figs}、{src_p / 'summary.json'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "../../outputs/trace")
