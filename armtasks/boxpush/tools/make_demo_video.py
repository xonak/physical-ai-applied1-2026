"""! @file make_demo_video.py
@brief デモ動画（1〜3 分）を作る: 場面ごとに `boxpush.cli.record` で撮り、タイトル板を挟んで連結する。

場面（既定）:
  1. 直立で押す 20 kg（押し負けて転倒）
  2. 前傾して押す 20 kg（成功）
  3. 前傾して押す 40 kg（手書きの限界、転倒）
  4. 学習方策 40 kg / 30 kg（`--learned ../../models/rl_m2` を与えたとき）

```bash
cd armtasks/boxpush
MUJOCO_GL=egl python tools/make_demo_video.py --learned ../../models/rl_m2 --out videos/boxpush_demo.mp4
```
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def title_card(text: str, sub: str, out: Path, seconds: float = 2.5, w: int = 960, h: int = 540) -> None:
    """! @brief タイトル板（静止画 → mp4）。"""
    from PIL import Image, ImageDraw

    sys.path.insert(0, str(HERE))
    from boxpush.cli.record import _font

    im = Image.new("RGB", (w, h), (24, 32, 44))
    dr = ImageDraw.Draw(im)
    f1, f2 = _font(44), _font(24)
    dr.text((60, h // 2 - 60), text, fill=(240, 240, 236), font=f1)
    dr.text((60, h // 2 + 10), sub, fill=(190, 196, 204), font=f2)
    png = out.with_suffix(".png")
    im.save(png)
    subprocess.run([shutil.which("ffmpeg"), "-y", "-loglevel", "error", "-loop", "1", "-i", str(png), "-t", str(seconds),
                    "-r", "30", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)], check=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="デモ動画をまとめて作る")
    ap.add_argument("--learned", default=None, help="学習済み方策のディレクトリ（無ければ手書きの 3 場面だけ）")
    ap.add_argument("--out", default="videos/boxpush_demo.mp4")
    ap.add_argument("--max-time", type=float, default=10.0)
    ap.add_argument("--scenes", default=None, help="撮る場面の番号（例 01,02）。既定は全部")
    ap.add_argument("--reuse", action="store_true", help="_clips/ に既にある場面は撮り直さない")
    a = ap.parse_args(sys.argv[1:] if argv is None else argv)

    out = Path(a.out)
    work = out.parent / "_clips"
    work.mkdir(parents=True, exist_ok=True)
    scenes = [
        ("01", "upright", 20, "手書き、直立（20 kg）", "押付け力に押し返されて後ろに倒れる"),
        ("02", "schedule", 20, "手書き、前傾（20 kg）", "重心を足より前に出すと押し切れる（前傾 = F / (m ω²)）"),
        ("03", "schedule", 40, "手書き、前傾（40 kg）", "必要な前傾 18 cm は足裏の長さを超え、押し切れない"),
    ]
    if a.learned:
        scenes += [("04", a.learned, 40, "学習方策（40 kg）", "質量は知らない。手の力と箱の動きから押し方を決める"),
                   ("05", a.learned, 20, "学習方策（20 kg）", "同じ方策で軽い箱も押す")]
    parts = []
    tc = work / "00_title.mp4"
    title_card("腕付きヒューマノイドで重い箱を押す", "全身制御（受動性ベース）+ 上位方策（強化学習）", tc, 3.0)
    parts.append(tc)
    env = dict(os.environ)
    if a.scenes:
        keep = set(a.scenes.split(","))
        scenes = [s for s in scenes if s[0] in keep]
    for tag, pol, mass, title, sub in scenes:
        card = work / f"{tag}_card.mp4"
        title_card(title, sub, card)
        clip = work / f"{tag}_clip.mp4"
        parts += [card, clip]
        if a.reuse and clip.is_file() and clip.stat().st_size > 10_000:
            continue
        subprocess.run([sys.executable, "-m", "boxpush.cli.record", "box_p2_walk_push", "--policy", pol, "--mass", str(mass),
                        "--max-time", str(a.max_time), "--out", str(clip.resolve()), "--title", title], check=True, cwd=HERE, env=env)
    lst = work / "concat.txt"
    lst.write_text("".join(f"file '{p.resolve().as_posix()}'\n" for p in parts), encoding="utf-8")
    subprocess.run([shutil.which("ffmpeg"), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-c", "copy", str(out)], check=True)
    print("書き出し:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
