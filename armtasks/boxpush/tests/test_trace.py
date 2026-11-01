"""! @file test_trace.py
@brief 再生・記録（trace_policy）→ 図（plot_trace）→ 閲覧ページ（make_trace_viewer）の流れを短いエピソードで固定する。
描画（render_trace）は OpenGL が要るので、MUJOCO_GL が設定されているときだけ 1 枚描く。"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest

import boxpush  # noqa: F401
from boxpush import PKG_ROOT


@pytest.mark.slow
def test_trace_plot_viewer(tmp_path):
    from boxpush.cli.trace_policy import run_one

    job = {"config": str(PKG_ROOT / "configs" / "experiments" / "box_p2_walk_push.yaml"), "policy": "schedule",
           "label": "schedule", "mass": 20.0, "mu": 0.2, "vx": 0.1, "max_time": 0.6, "out": str(tmp_path), "fps": 30}
    meta = run_one(job)
    z = np.load(tmp_path / "schedule_20kg.npz")
    for k in ("t", "f_tgt", "f_cmd", "lean", "f_meas", "box_dx", "face_c", "pitch", "foot_x", "a", "qpos", "t_frame"):
        assert k in z.files
    assert z["qpos"].shape[0] == z["t_frame"].shape[0] >= 18          # 0.6 s × 30 fps
    assert z["foot_x"].shape[1] == 2 and z["a"].shape[1] == 2
    assert json.loads(str(z["meta"]))["max_time"] == 0.6 and not meta["fell"]
    tools = PKG_ROOT / "tools"
    subprocess.run([sys.executable, str(tools / "plot_trace.py"), str(tmp_path)], check=True, capture_output=True)
    assert (tmp_path / "figs" / "schedule_20kg.png").exists() and (tmp_path / "figs" / "compare_20kg.png").exists()
    s = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert s[0]["label"] == "schedule" and s[0]["mass"] == 20.0
    subprocess.run([sys.executable, str(tools / "make_trace_viewer.py"), str(tmp_path)], check=True, capture_output=True)
    html = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "schedule" in html and "figs/" in html
    if os.environ.get("MUJOCO_GL"):
        from boxpush.cli.render_trace import TraceRenderer

        r = TraceRenderer(tmp_path / "schedule_20kg.npz", fast=True)
        img = r.frame(r.n_frames - 1, hold=True)
        assert img.shape == (540 + 180, 960, 3) and img.std() > 5
