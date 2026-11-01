"""! @file make_trace_viewer.py
@brief 記録フォルダに閲覧ページ index.html を作る。質量 × 制御を選んで、動画と図（入力・状態）と要約を見る。

```bash
cd armtasks/boxpush
python tools/make_trace_viewer.py ../../outputs/trace     # → ../../outputs/trace/index.html をブラウザで開く
```
前提: `boxpush.cli.trace_policy`（*.npz）→ `tools/plot_trace.py`（figs/, summary.json）→ `boxpush.cli.render_trace`（videos/）。
ページは相対パスで videos/ と figs/ を参照する（フォルダごと移しても見られる）。動画が無い条件は図だけ出す。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from boxpush.plotstyle import POL_COLOR, POL_NAME  # noqa: E402

PAGE = r"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#fcfcfb;--surf:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--line:#e4e3df;--accent:#2a78d6;--ok:#0c8c0c;--ng:#c83232}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--surf:#1f1f1e;--ink:#f1f0ec;--ink2:#a8a7a2;--line:#353533;--accent:#5b9be6;--ok:#4cc24c;--ng:#ec6a6a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.6 "Noto Sans CJK JP","Meiryo","Yu Gothic",sans-serif}
main{max-width:1280px;margin:0 auto;padding:16px}
h1{font-size:20px;margin:4px 0 2px}h2{font-size:16px;margin:22px 0 8px}.sub{color:var(--ink2);margin:0 0 12px}
table{border-collapse:collapse;background:var(--surf);font-variant-numeric:tabular-nums}
th,td{border:1px solid var(--line);padding:5px 10px;text-align:center}th{color:var(--ink2);font-weight:600}
td.cell{cursor:pointer}td.cell:hover{outline:2px solid var(--accent)}td.sel{background:color-mix(in srgb,var(--accent) 14%,var(--surf))}
.ok{color:var(--ok);font-weight:700}.ng{color:var(--ng);font-weight:700}
.bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:10px 0}
.bar button{border:1px solid var(--line);background:var(--surf);color:var(--ink);padding:5px 12px;border-radius:6px;cursor:pointer;font:inherit}
.bar button.on{border-color:var(--accent);box-shadow:inset 0 0 0 1px var(--accent);font-weight:700}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;vertical-align:middle}
.grid{display:grid;gap:10px;grid-template-columns:repeat(auto-fit,minmax(360px,1fr))}
.card{background:var(--surf);border:1px solid var(--line);border-radius:8px;padding:8px}
.card h3{font-size:14px;margin:2px 0 6px}video{width:100%;display:block;border-radius:4px;background:#000}
img.fig{width:100%;display:block;border-radius:4px;border:1px solid var(--line)}
.note{color:var(--ink2);font-size:12.5px}.wide{overflow-x:auto}
</style></head><body><main>
<h1>__TITLE__</h1>
<p class="sub">__SUB__</p>

<h2>結果の一覧（セルを押すとその条件を表示）</h2>
<div class="wide"><table id="tbl"></table></div>

<h2>表示する条件</h2>
<div class="bar"><span>箱の質量:</span><span id="mass"></span></div>
<div class="bar"><span>表示:</span><span id="mode"></span>
  <button id="play" title="並んだ動画を頭から同時に再生">▶ 同時に再生</button><span class="note" id="speed"></span></div>

<div id="videos" class="grid"></div>

<h2 id="h_detail">入力とロボットの状態</h2>
<div id="detail"></div>

<h2>要約（押している区間 t ≥ 3 s の平均）</h2>
<div class="wide"><table id="sum"></table></div>
<p class="note">前傾の釣り合い = 押付け力の実測 / (m ω²)（外力に釣り合う CMP のずれ）。箱の面 − 重心のばらつきが小さいほど、箱との距離を一定に保って押している。
DCM 誤差は 3 s 以降の RMS。押し始め / 前傾開始は指令が 5 N / 1 cm を超えた時刻。</p>
</main>
<script>
const D = __DATA__;
const NAME = __NAME__, COLOR = __COLOR__;
const labels = D.labels, masses = D.masses;
let mass = masses[masses.length-1], mode = "compare";
const S = Object.fromEntries(D.summary.map(s => [s.label + "_" + s.mass, s]));
const key = (l, m) => `${l}_${Math.round(m)}kg`;
function el(t, a = {}, h = ""){const e = document.createElement(t); Object.assign(e, a); if (h) e.innerHTML = h; return e;}
function stat(s){ return s ? `<span class="${s.status === "成功" ? "ok" : "ng"}">${s.status}</span> ${s.box_dx.toFixed(2)} m` : "–"; }
function drawTable(){
  const t = document.getElementById("tbl"); t.innerHTML = "";
  const hr = el("tr"); hr.append(el("th", {}, "制御 ＼ 箱")); masses.forEach(m => hr.append(el("th", {}, `${m} kg`))); t.append(hr);
  labels.forEach(l => { const tr = el("tr");
    tr.append(el("th", {}, `<span class="dot" style="background:${COLOR[l]||"#7b61c9"}"></span>${NAME[l]||l}`));
    masses.forEach(m => { const s = S[l + "_" + m];
      const td = el("td", {className: "cell" + (m === mass && (mode === "compare" || mode === l) ? " sel" : "")}, stat(s));
      td.onclick = () => { mass = m; mode = l; render(); }; tr.append(td); });
    t.append(tr); });
}
function drawButtons(){
  const bm = document.getElementById("mass"); bm.innerHTML = "";
  masses.forEach(m => { const b = el("button", {className: m === mass ? "on" : ""}, `${m} kg`); b.onclick = () => { mass = m; render(); }; bm.append(b); });
  const bd = document.getElementById("mode"); bd.innerHTML = "";
  const b0 = el("button", {className: mode === "compare" ? "on" : ""}, "制御を並べて比較"); b0.onclick = () => { mode = "compare"; render(); }; bd.append(b0);
  labels.forEach(l => { const b = el("button", {className: mode === l ? "on" : ""}, `<span class="dot" style="background:${COLOR[l]||"#7b61c9"}"></span>${NAME[l]||l}`);
    b.onclick = () => { mode = l; render(); }; bd.append(b); });
}
function drawVideos(){
  const v = document.getElementById("videos"); v.innerHTML = "";
  const ls = mode === "compare" ? labels : [mode];
  v.style.gridTemplateColumns = ls.length > 1 ? "repeat(auto-fit,minmax(360px,1fr))" : "minmax(0,960px)";
  ls.forEach(l => { const k = key(l, mass), s = S[l + "_" + mass]; const c = el("div", {className: "card"});
    c.append(el("h3", {}, `<span class="dot" style="background:${COLOR[l]||"#7b61c9"}"></span>${NAME[l]||l}　${stat(s)}`));
    if (D.videos.includes(k)) { const vd = el("video", {src: `videos/${k}.mp4`, controls: true, muted: true, preload: "metadata", playsInline: true}); c.append(vd); }
    else c.append(el("p", {className: "note"}, "動画なし（boxpush.cli.render_trace で作る）"));
    v.append(c); });
}
function drawDetail(){
  const d = document.getElementById("detail"); d.innerHTML = "";
  const h = document.getElementById("h_detail");
  if (mode === "compare") { h.textContent = `入力とロボットの状態: ${mass} kg、制御を重ねた比較`;
    d.append(el("img", {className: "fig", src: `figs/compare_${mass}kg.png`, alt: "比較図", loading: "lazy"}));
    const g = el("div", {className: "grid"}); g.style.marginTop = "10px";
    labels.forEach(l => { const c = el("div", {className: "card"}); c.append(el("h3", {}, NAME[l]||l));
      c.append(el("img", {className: "fig", src: `figs/${key(l, mass)}.png`, loading: "lazy"})); g.append(c); });
    d.append(g);
  } else { h.textContent = `入力とロボットの状態: ${NAME[mode]||mode}、${mass} kg`;
    d.append(el("img", {className: "fig", src: `figs/${key(mode, mass)}.png`, alt: "詳細図"})); }
}
function drawSummary(){
  const rows = [["結果", s => stat(s)], ["押し始め [s]", s => s.t_push_start ?? "–"], ["前傾開始 [s]", s => s.t_lean_start ?? "–"],
    ["押付け力 指令 [N]", s => s.f_cmd_mean], ["押付け力 実測 [N]", s => s.f_meas_mean], ["静止摩擦 μmg [N]", s => s.static_friction_N],
    ["前傾 [cm]", s => s.lean_mean_cm], ["前傾の釣り合い F/(mω²) [cm]", s => s.lean_balance_cm],
    ["箱の速度 平均 / ばらつき [m/s]", s => `${s.v_box_mean} / ${s.v_box_std}`], ["箱の面 − 重心 平均 / ばらつき [m]", s => `${s.face_c_mean} / ${s.face_c_std}`],
    ["ピッチ 最大 [deg]", s => s.pitch_absmax], ["DCM 誤差 RMS [cm]", s => s.e_dcm_rms_cm],
    ["行動の変化 |Δa| 平均", s => s.da_mean ? s.da_mean.join(" / ") : "–"]];
  const t = document.getElementById("sum"); t.innerHTML = "";
  const hr = el("tr"); hr.append(el("th", {}, `${mass} kg`)); labels.forEach(l => hr.append(el("th", {}, NAME[l]||l))); t.append(hr);
  rows.forEach(([n, f]) => { const tr = el("tr"); tr.append(el("th", {}, n));
    labels.forEach(l => { const s = S[l + "_" + mass]; tr.append(el("td", {}, s ? String(f(s)) : "–")); }); t.append(tr); });
}
function render(){ drawTable(); drawButtons(); drawVideos(); drawDetail(); drawSummary();
  try { history.replaceState(null, "", `#${mode}_${mass}`); } catch (e) {} }
document.getElementById("play").onclick = () => { const vs = [...document.querySelectorAll("#videos video")];
  vs.forEach(v => { v.pause(); v.currentTime = 0; }); vs.forEach(v => v.play()); };
const h = location.hash.slice(1).split("_"); if (h.length === 2 && masses.includes(+h[1])) { mass = +h[1]; if (h[0] === "compare" || labels.includes(h[0])) mode = h[0]; }
render();
</script></body></html>
"""


def main(src: str) -> None:
    d = Path(src)
    summ_p = d / "summary.json"
    if not summ_p.exists():
        raise SystemExit(f"{summ_p} が無い。先に python tools/plot_trace.py {d}")
    summary = json.loads(summ_p.read_text(encoding="utf-8"))
    order = list(POL_COLOR) + sorted({s["label"] for s in summary} - set(POL_COLOR))
    labels = [l for l in order if any(s["label"] == l for s in summary)]
    masses = sorted({int(round(s["mass"])) for s in summary})
    for s in summary:
        s["mass"] = int(round(s["mass"]))
    videos = sorted(p.stem for p in (d / "videos").glob("*_*kg.mp4")) if (d / "videos").exists() else []
    mu = summary[0]["mu"] if summary else 0.0
    data = {"labels": labels, "masses": masses, "summary": summary, "videos": videos}
    title = f"箱押し: 学習方策と手書きの比較（{d.name}）"
    sub = (f"押しながら歩く（指令 0.1 m/s、10 s）。箱-床の摩擦 {mu:.2f}、箱の初期位置は全条件で同じ。"
           f"成功 = 10 s で箱が 0.5 m 以上進み、転倒しない。動画 {len(videos)} 本。")
    html = (PAGE.replace("__TITLE__", title).replace("__SUB__", sub)
            .replace("__DATA__", json.dumps(data, ensure_ascii=False))
            .replace("__NAME__", json.dumps(POL_NAME, ensure_ascii=False))
            .replace("__COLOR__", json.dumps(POL_COLOR)))
    out = d / "index.html"
    out.write_text(html, encoding="utf-8")
    print(f"書き出し: {out}（条件 {len(summary)}、動画 {len(videos)}）")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "../../outputs/trace")
