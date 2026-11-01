"""! @file render_trace.py
@brief `trace_policy` の記録（qpos）から動画を描く。物理は回さないので、同じ記録から何度でも描き直せる。

1 本 = 1 条件（制御 × 箱の質量）。画面の構成（既定 960 × 720）:
- 上 960 × 540: MuJoCo の描画（ロボットと箱を追う視点）。左上に入力と状態の数値、右上に結果（成功 / 転倒）
- 下 960 × 180: 時系列の帯 3 枚（押付け力 指令 / 実測、前傾、箱の速度）。今の時刻に縦線

```bash
cd armtasks/boxpush
# GPU のある Ubuntu
MUJOCO_GL=egl python -m boxpush.cli.render_trace ../../outputs/trace --workers 8
# GL の無い Linux（VM など）
MUJOCO_GL=glfw xvfb-run -a -s "-screen 0 1280x720x24" python -m boxpush.cli.render_trace ../../outputs/trace
# 遅い環境では --fast（影と映り込みを切る）。さらに 1 本を分けて描き、あとで連結もできる（--part 1/3 ... 3/3 → --concat）
python -m boxpush.cli.render_trace ../../outputs/trace --only rl_40kg --part 1/3
python -m boxpush.cli.render_trace ../../outputs/trace --only rl_40kg --concat
```
書き出し: `<trace>/videos/<label>_<mass>kg.mp4`
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

W, H_SIM, H_STRIP = 960, 540, 180


def _strip_background(d: dict, width: int, height: int):
    """! @brief 下の時系列の帯（全区間）を一度だけ描いて画像と座標変換を返す。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ..plotstyle import BLUE, GRID, INK2, ORANGE, SURF, style

    style()
    t = d["t"]
    m = d["meta"]
    dpi = 100
    fig, axs = plt.subplots(1, 3, figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.patch.set_facecolor(SURF)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.84, bottom=0.17, wspace=0.22)
    panels = [
        ("押付け力 [N]（青 指令 / 灰 実測）", [(d["f_cmd"], BLUE, 1.8), (np.convolve(d["f_meas"], np.ones(15) / 15, "same"), INK2, 1.0)],
         m["static_friction_N"]),
        ("前傾 [cm]", [(100 * d["lean"], BLUE, 1.8)], None),
        ("箱の速度 [m/s]（破線 歩行）", [(d["v_box"], BLUE, 1.4)], m.get("vx", 0.1)),
    ]
    t_max = max(float(m.get("max_time", 10.0)), float(t[-1]))   # 転倒で早く終わっても横軸はそろえる
    for ax, (title, lines, ref) in zip(axs, panels):
        ax.set_facecolor(SURF)
        ax.grid(True, color=GRID, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for y, c, lw in lines:
            ax.plot(t, y, color=c, lw=lw)
        if ref is not None:
            ax.axhline(ref, color=ORANGE, lw=0.8, ls="--")
        ax.set_xlim(0.0, t_max)
        ax.set_title(title, fontsize=9, loc="left")
        ax.tick_params(labelsize=7)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, 0:3].copy()
    # データ座標 → 画素（各パネルの x 範囲と y 範囲）
    boxes = []
    for ax in axs:
        bb = ax.get_window_extent()
        boxes.append((bb.x0, bb.x1, height - bb.y1, height - bb.y0, ax.get_xlim(), ax.get_ylim()))
    plt.close(fig)
    return img, boxes


class TraceRenderer:
    """! @brief 1 条件の記録を描く。"""

    def __init__(self, npz: Path, width: int = W, height: int = H_SIM, strip: int = H_STRIP, fast: bool = False,
                 hand_visual: str = "model") -> None:
        """! @brief 記録 npz を読み、同じモデル（meta の robot）で描画器を用意する。
        @param npz trace_policy の記録 [-]
        @param width, height, strip 画面幅・高さ・下段の高さ [px]
        @param fast 影と映り込みを切る [-]
        @param hand_visual 手の見た目 [-]。"model": モデルのまま、"mesh": ゴム手メッシュ、"sphere": 球、
               "both": 両方（球は半透明）。球の visual geom `*_hand_vis_sphere` が無いモデル
               （g1_box.xml）で sphere / both を指定すると、衝突球と同じ球を描画用に足す
        """
        import mujoco

        import boxpush  # noqa: F401
        from ..cli.run_box import load_box_experiment, resolve_experiment
        from ..env import BoxDomain, BoxPushEnv

        z = np.load(npz, allow_pickle=False)
        self.d = {k: z[k] for k in z.files if k != "meta"}
        self.m = json.loads(str(z["meta"]))
        self.d["meta"] = self.m
        cfg, push = load_box_experiment(resolve_experiment(self.m.get("config", "box_p2_walk_push")),
                                        robot=self.m.get("robot"))
        cfg.output.verbose = False
        mass, mu = self.m["mass"], self.m["mu"]
        self.env = BoxPushEnv(cfg, domain=BoxDomain(mass=(mass, mass), friction=(mu, mu), x_offset=(0, 0)), max_time=1.0)
        self.mj = mujoco
        self.model, self.data = self.env.env.model, self.env.env.data
        _set_hand_visual(self.model, hand_visual)
        self.model.vis.global_.offwidth = max(int(self.model.vis.global_.offwidth), width)
        self.model.vis.global_.offheight = max(int(self.model.vis.global_.offheight), height)
        self.width, self.height, self.strip = width, height, strip
        self.renderer = mujoco.Renderer(self.model, height, width)
        if fast:   # 影と床の映り込みを切る（ソフトウェア GL で 0.8 → 0.18 s/枚）
            self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
            self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 3.2, 80.0, -8.0
        self._box_qadr = self.env.env._box_free[0][0]
        self.strip_img, self.boxes = _strip_background(self.d, width, strip)
        from .record import _font

        self.font, self.font_s, self.font_xs = _font(21), _font(16), _font(14)

    @property
    def n_frames(self) -> int:
        return int(self.d["qpos"].shape[0])

    def _row(self, t: float) -> int:
        return int(np.clip(np.searchsorted(self.d["t"], t), 0, len(self.d["t"]) - 1))

    def _panel(self, img: np.ndarray, t: float, last: bool):
        from PIL import Image, ImageDraw

        from ..plotstyle import POL_NAME

        d, m = self.d, self.m
        i = self._row(t)
        im = Image.fromarray(img)
        dr = ImageDraw.Draw(im, "RGBA")
        title = f"{POL_NAME.get(m['label'], m['label'])}　箱 {m['mass']:.0f} kg（摩擦 {m['mu']:.2f}）"
        inp = [
            "入力（方策の出力 → 指令）",
            f"  押付け力 {d['f_tgt'][i]:5.1f} → {d['f_cmd'][i]:5.1f} N",
            f"  前傾     {100 * d['lean_tgt'][i]:5.1f} → {100 * d['lean'][i]:5.1f} cm",
        ]
        sta = [
            "ロボットの状態",
            f"  押付け力の実測 {d['f_meas'][i]:5.1f} N（左 {d['f_L'][i]:4.1f} / 右 {d['f_R'][i]:4.1f}）",
            f"  箱 移動 {d['box_dx'][i]:4.2f} m　速度 {d['v_box'][i]:5.2f} m/s",
            f"  箱の面 − 重心 {d['face_c'][i]:4.2f} m　ピッチ {d['pitch'][i]:5.1f}°",
            f"  DCM 誤差 {100 * d['e_dcm'][i]:4.1f} cm",
        ]
        lines = inp + sta
        y0, x0, w = 8, 8, 372
        h = 14 + 28 + 20 * len(lines)
        dr.rectangle([x0, y0, x0 + w, y0 + h], fill=(255, 255, 255, 200))
        y = y0 + 4
        dr.text((x0 + 8, y), f"t = {t:4.1f} s", fill=(10, 10, 10, 255), font=self.font)
        y += 28
        for k, ln in enumerate(lines):
            head = k in (0, len(inp))
            dr.text((x0 + 8, y), ln, fill=(20, 20, 20, 255) if head else (55, 55, 55, 255),
                    font=self.font_s if head else self.font_xs)
            y += 20
        tw = dr.textlength(title, font=self.font_s)
        dr.rectangle([self.width - tw - 30, self.height - 40, self.width - 10, self.height - 8], fill=(255, 255, 255, 215))
        dr.text((self.width - tw - 20, self.height - 37), title, fill=(10, 10, 10, 255), font=self.font_s)
        if last:
            st = "成功" if m["success"] else ("転倒" if m["fell"] else ("箱が倒れた" if m.get("box_tipped") else "未達"))
            col = (12, 140, 12, 230) if st == "成功" else (200, 50, 50, 230)
            label = f"{st}　箱 {m['box_dx']:.2f} m"
            tw = dr.textlength(label, font=self.font)
            dr.rectangle([self.width - tw - 40, 14, self.width - 14, 62], fill=col)
            dr.text((self.width - tw - 27, 22), label, fill=(255, 255, 255, 255), font=self.font)
        return np.asarray(im)

    def _strip(self, t: float) -> np.ndarray:
        from PIL import Image, ImageDraw

        im = Image.fromarray(self.strip_img.copy())
        dr = ImageDraw.Draw(im)
        for (x0, x1, ytop, ybot, xl, _yl) in self.boxes:
            x = x0 + (x1 - x0) * (t - xl[0]) / max(xl[1] - xl[0], 1e-9)
            dr.line([(x, ytop), (x, ybot)], fill=(235, 104, 52), width=2)
        return np.asarray(im)

    def frame(self, j: int, hold: bool = False) -> np.ndarray:
        q = self.d["qpos"][j]
        self.data.qpos[:] = q
        self.mj.mj_forward(self.model, self.data)
        pel = np.array(self.data.xpos[self.env.env._base_body])
        box = np.array(q[self._box_qadr:self._box_qadr + 3], dtype=np.float64)
        look = 0.6 * pel + 0.4 * box
        self.cam.lookat[:] = [look[0] - 0.15, look[1], 1.0]   # 少し後ろ・上を見て、体を画面の中央下に
        self.renderer.update_scene(self.data, self.cam)
        img = self.renderer.render()
        t = float(self.d["t_frame"][j])
        top = self._panel(img, t, last=hold or j == self.n_frames - 1)
        return np.concatenate([top, self._strip(t)], axis=0)


def _ffmpeg(out: Path, w: int, h: int, fps: float):
    ff = shutil.which("ffmpeg")
    if ff is None:
        raise RuntimeError("ffmpeg が見つからない（apt install ffmpeg）")
    out.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen([ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                             "-r", f"{fps:g}", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "21",
                             "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)


def _set_hand_visual(model, mode: str) -> None:
    """! @brief 手の見た目を切り替える（geom_rgba の α を書き換えるだけ。物理は変えない）。
    @param model mujoco.MjModel [-]
    @param mode "model" | "mesh" | "sphere" | "both" [-]
    @return なし
    @note visual の手メッシュは名前が無い（原典のまま）ので、手ボディ内の mesh 型 visual geom を探す。
          球の visual geom `<hand>_hand_vis_sphere` が無いモデルでは、衝突 geom `<hand>_hand` が球なら
          それを描画に使う（group を visual の 2 に移す）。衝突がメッシュのモデルで sphere を頼まれたら
          メッシュを消すだけにし、警告を出す（物理は指先接触のままなので、嘘の絵になるため）。
    """
    import mujoco

    if mode == "model":
        return
    if mode not in ("mesh", "sphere", "both"):
        raise ValueError(f"hand_visual は model/mesh/sphere/both: {mode!r}")
    a_mesh = 1.0 if mode in ("mesh", "both") else 0.0
    a_sph = {"mesh": 0.0, "sphere": 1.0, "both": 0.45}[mode]
    for hand in ("left", "right"):
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{hand}_wrist_yaw_link")
        if body < 0:
            continue
        col = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{hand}_hand")
        vis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{hand}_hand_vis_sphere")
        for g in range(model.ngeom):
            if model.geom_bodyid[g] != body or g == col:
                continue
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and model.geom_group[g] == 2:
                model.geom_rgba[g, 3] = a_mesh           # ゴム手メッシュ（visual）
        if vis >= 0:
            model.geom_rgba[vis, 3] = a_sph
        elif a_sph > 0 and col >= 0 and model.geom_type[col] == mujoco.mjtGeom.mjGEOM_SPHERE:
            model.geom_group[col] = 2                    # 衝突球をそのまま描く
            model.geom_rgba[col] = (0.35, 0.35, 0.38, a_sph)
        elif a_sph > 0:
            print(f"警告: {hand} 手の衝突は球でない（g1_box.xml）。--hand-visual {mode} は手メッシュを消すだけになる",
                  file=sys.stderr)


def render_one(job: dict) -> str:
    """! @brief 1 本（または 1 部分）を描く。"""
    npz, out = Path(job["npz"]), Path(job["out"])
    r = TraceRenderer(npz, fast=bool(job.get("fast", False)), hand_visual=str(job.get("hand_visual", "model")))
    n = r.n_frames
    fps = float(r.m.get("fps", 30))
    i, k = job.get("part", (1, 1))
    a, b = (n * (i - 1)) // k, (n * i) // k
    dst = out if k == 1 else out.with_suffix(f".part{i:02d}.mp4")
    p = _ffmpeg(dst, r.width, r.height + r.strip, fps)
    for j in range(a, b):
        p.stdin.write(r.frame(j).tobytes())
    if i == k:                                   # 最後に 1.5 s 止めて結果を出す
        last = r.frame(n - 1, hold=True).tobytes()
        for _ in range(int(1.5 * fps)):
            p.stdin.write(last)
    p.stdin.close()
    p.wait()
    return str(dst)


def concat(out: Path) -> None:
    parts = sorted(out.parent.glob(out.stem + ".part*.mp4"))
    if not parts:
        raise SystemExit(f"{out.stem}.part*.mp4 が無い")
    lst = out.parent / (out.stem + ".parts.txt")
    lst.write_text("".join(f"file '{p.name}'\n" for p in parts), encoding="utf-8")
    subprocess.run([shutil.which("ffmpeg"), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-c", "copy", "-movflags", "+faststart", str(out)], check=True)
    print(f"連結: {out}（{len(parts)} 部分）。部分ファイルは残すので不要なら消す")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="trace_policy の記録から動画を描く")
    ap.add_argument("trace", help="記録のフォルダ（*_<mass>kg.npz）")
    ap.add_argument("--only", default=None, help="描く条件（例 rl_40kg,schedule_40kg）")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--part", default=None, help="i/n: 1 本を n 分割した i 番目だけ描く（遅い環境用）")
    ap.add_argument("--concat", action="store_true", help="--part で描いた部分を連結する")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--fast", action="store_true", help="影と映り込みを切って速く描く（GPU の無い環境向け）")
    ap.add_argument("--hand-visual", default="model", choices=("model", "mesh", "sphere", "both"),
                    help="手の見た目。model: 記録したモデルのまま（g1_box_sphere なら球）、mesh: ゴム手、sphere: 球、both: 両方")
    a = ap.parse_args(sys.argv[1:] if argv is None else argv)

    if os.environ.get("MUJOCO_GL") is None and not os.environ.get("DISPLAY") and sys.platform.startswith("linux"):
        print("表示が無い: MUJOCO_GL=egl（GPU）か、MUJOCO_GL=glfw xvfb-run -a ... で実行する", file=sys.stderr)
    src = Path(a.trace)
    files = sorted(src.glob("*_*kg.npz"))
    if a.only:
        keep = set(a.only.split(","))
        files = [f for f in files if f.stem in keep]
    vdir = src / "videos"
    if a.concat:
        for f in files:
            concat(vdir / f"{f.stem}.mp4")
        return 0
    part = tuple(int(v) for v in a.part.split("/")) if a.part else (1, 1)
    jobs = [{"npz": str(f), "out": str(vdir / f"{f.stem}.mp4"), "part": part, "fast": a.fast, "hand_visual": a.hand_visual}
            for f in files
            if not (a.skip_existing and part == (1, 1) and (vdir / f"{f.stem}.mp4").exists())]
    if a.workers > 1:
        import multiprocessing as mp

        with mp.get_context("spawn").Pool(a.workers) as pool:
            for r in pool.imap_unordered(render_one, jobs):
                print("書き出し:", r, flush=True)
    else:
        for j in jobs:
            print("書き出し:", render_one(j), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
