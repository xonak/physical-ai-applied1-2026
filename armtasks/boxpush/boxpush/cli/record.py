"""! @file record.py
@brief 箱押しの動画を書き出す（オフスクリーン描画 → ffmpeg で mp4）。1 本 = 1 場面（方策 × 箱の質量）。

描画は MuJoCo の `Renderer`。表示の無い環境では OpenGL の実装を選ぶ:
- GPU のある Ubuntu: `MUJOCO_GL=egl python -m boxpush.cli.record ...`
- GL の無い VM: `MUJOCO_GL=glfw xvfb-run -a -s "-screen 0 1280x720x24" python -m boxpush.cli.record ...`
- Windows / 表示のある Linux: そのまま（glfw）

```bash
cd armtasks/boxpush
python -m boxpush.cli.record box_p2_walk_push --policy upright  --mass 20 --out videos/01_upright_20kg.mp4 --title "直立で押す（20 kg）"
python -m boxpush.cli.record box_p2_walk_push --policy schedule --mass 20 --out videos/02_lean_20kg.mp4  --title "前傾して押す（20 kg）"
python -m boxpush.cli.record box_p2_walk_push --policy ../../models/rl_m2 --mass 40 --out videos/04_rl_40kg.mp4 --title "学習方策（40 kg）"
```
画面の左上に時刻・箱の質量・押付け力（指令 / 手首 F/T の実測）・前傾・重心高さ・箱の移動を重ねる。
転倒・成功で止め、最後の状態を 1 秒残す。
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


def _font(size: int):
    from PIL import ImageFont

    for p in (FONT, "C:/Windows/Fonts/meiryo.ttc", "C:/Windows/Fonts/YuGothM.ttc",
              "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"):
        if Path(p).is_file():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


class Recorder:
    """! @brief BoxPushEnv の frame_cb に挿して、一定の時間間隔で描画して ffmpeg に流す。"""

    def __init__(self, env, out: Path, fps: int = 30, width: int = 960, height: int = 540, title: str = "",
                 distance: float = 3.2, azimuth: float = 80.0, elevation: float = -8.0) -> None:
        import mujoco

        self.mj = mujoco
        self.env = env
        self.fps = fps
        self.width, self.height = width, height
        self.title = title
        m = env.env.model
        m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), width)     # 既定 640x480 の描画バッファを広げる
        m.vis.global_.offheight = max(int(m.vis.global_.offheight), height)
        self.renderer = mujoco.Renderer(m, height, width)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.distance, self.cam.azimuth, self.cam.elevation = distance, azimuth, elevation
        self.next_t = 0.0
        self.font, self.font_s = _font(26), _font(20)
        self.status = ""
        ff = shutil.which("ffmpeg")
        if ff is None:
            raise RuntimeError("ffmpeg が見つからない（apt install ffmpeg / conda install ffmpeg）")
        out.parent.mkdir(parents=True, exist_ok=True)
        self.proc = subprocess.Popen(
            [ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(out)],
            stdin=subprocess.PIPE)
        self.n = 0

    def _overlay(self, img: np.ndarray) -> np.ndarray:
        from PIL import Image, ImageDraw

        e = self.env
        ap = e.applier
        st = e.state
        bp = np.asarray(st.box_pose)
        wh = np.asarray(st.wrench_hands)
        f_meas = float(-wh[:, 0].sum()) if wh.shape == (2, 6) else 0.0
        t = e.k * e.dt_ctrl
        lines = [
            f"t = {t:4.1f} s",
            f"箱 {e.episode_info.get('box_mass', 0):.0f} kg（摩擦 {e.episode_info.get('box_friction', 0):.2f}）",
            f"押付け力  指令 {ap.last_vals.get('f_push', 0):5.1f} N / 実測 {f_meas:5.1f} N",
            f"前傾 {100 * ap.last_vals.get('lean_x', 0):4.1f} cm   重心高さ {st.c_W[2]:.3f} m",
            f"箱の移動 {float(bp[0]) - e.box_x0 if bp.size else 0:4.2f} m",
        ]
        im = Image.fromarray(img)
        dr = ImageDraw.Draw(im, "RGBA")
        box_h = 14 + 28 * len(lines) + (34 if self.title else 0)
        y0 = 12                                                # 左上（空の部分）。注視点を高めにして上に空きを作る
        dr.rectangle([12, y0, 400, y0 + box_h], fill=(255, 255, 255, 200))
        y = y0 + 6
        if self.title:
            dr.text((22, y), self.title, fill=(10, 10, 10, 255), font=self.font); y += 34
        for ln in lines:
            dr.text((22, y), ln, fill=(40, 40, 40, 255), font=self.font_s); y += 28
        if self.status:
            col = (12, 140, 12, 230) if "成功" in self.status else (200, 50, 50, 230)
            w = 30 * len(self.status) + 40
            dr.rectangle([self.width - w - 16, 16, self.width - 16, 70], fill=col)
            dr.text((self.width - w + 4, 22), self.status, fill=(255, 255, 255, 255), font=self.font)
        return np.asarray(im)

    def frame(self, env=None, k: int | None = None, force: bool = False) -> None:
        e = self.env
        t = e.k * e.dt_ctrl
        if not force and t + 1e-9 < self.next_t:
            return
        self.next_t += 1.0 / self.fps
        d = e.env.data
        pel = d.xpos[e.env._base_body]
        box = e.env.box_pose(0)
        look = 0.6 * np.asarray(pel) + 0.4 * (np.asarray(box[0:3]) if box.size else np.asarray(pel))
        self.cam.lookat[:] = [look[0], look[1], 0.85]
        self.renderer.update_scene(d, self.cam)
        img = self.renderer.render()
        self.proc.stdin.write(self._overlay(img).tobytes())
        self.n += 1

    def hold(self, seconds: float) -> None:
        for _ in range(int(seconds * self.fps)):
            self.frame(force=True)

    def close(self) -> None:
        self.proc.stdin.close()
        self.proc.wait()


def main(argv: list[str] | None = None) -> int:
    from ..cli.eval_policy import _make_policy
    from ..cli.run_box import load_box_experiment, resolve_experiment
    from ..env import BoxDomain, BoxPushEnv

    ap = argparse.ArgumentParser(description="箱押しの動画（1 場面）")
    ap.add_argument("config")
    ap.add_argument("--policy", default="schedule", help="upright | schedule | reactive | 学習済みディレクトリ")
    ap.add_argument("--mass", type=float, default=20.0)
    ap.add_argument("--mu", type=float, default=0.2)
    ap.add_argument("--vx", type=float, default=0.1)
    ap.add_argument("--max-time", type=float, default=10.0)
    ap.add_argument("--out", default="videos/clip.mp4")
    ap.add_argument("--title", default="")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--azimuth", type=float, default=80.0, help="視点の方位 [deg]（90 で真横）")
    a = ap.parse_args(sys.argv[1:] if argv is None else argv)

    if os.environ.get("MUJOCO_GL") is None and not os.environ.get("DISPLAY") and sys.platform.startswith("linux"):
        print("表示が無い: MUJOCO_GL=egl（GPU）か、MUJOCO_GL=glfw xvfb-run -a ... で実行する", file=sys.stderr)
    cfg, push = load_box_experiment(resolve_experiment(a.config))
    cfg.output.verbose = False
    pol, spec = _make_policy(a.policy, None, push, a.mass, a.mu)
    env = BoxPushEnv(cfg, spec=spec, domain=BoxDomain(mass=(a.mass, a.mass), friction=(a.mu, a.mu), x_offset=(0.0, 0.0)),
                     max_time=a.max_time, vx=a.vx, t_walk=push.t_start, seed=0, base_policy=getattr(pol, "base", None))
    obs = env.reset()
    rec = Recorder(env, Path(a.out), fps=a.fps, width=a.width, height=a.height, title=a.title, azimuth=a.azimuth)
    env.frame_cb = rec.frame
    rec.frame(force=True)
    done = False
    while not done:
        if hasattr(pol, "set_time"):
            pol.set_time(env.k * env.dt_ctrl)
        obs, r, done, info = env.step(pol.act(obs))
    rec.status = "成功" if info.get("success") else ("転倒" if info.get("fell") else ("箱が倒れた" if info.get("box_tipped") else "未達"))
    rec.hold(1.0)
    rec.close()
    print(f"書き出し: {a.out}（{rec.n} フレーム、{rec.n / a.fps:.1f} s）結果: {rec.status}, 箱の移動 {info['box_dx']:.2f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
