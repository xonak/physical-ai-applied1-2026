# 実行と評価

基本依存を導入したうえで、次のコマンドは `armtasks/boxpush` から実行します。
WindowsのPythonの指定は `..\..\.venv\Scripts\python.exe`、Linuxでは `../../.venv/bin/python` です。

## 手書き制御で動かす

```powershell
..\..\.venv\Scripts\python.exe -m boxpush.cli.run_box box_p2_walk_push --viewer --realtime
```

`--policy` を省略すると、設定ファイルの力と前傾のスケジュールで動きます。
`eval_policy` の `schedule` は、評価する質量と摩擦に合わせて力を計算するため、上の固定スケジュールとは条件が異なります。

## 手のモデルを選ぶ

標準の `box_p2_walk_push` は、球形の手の `robot_g1_box_sphere.yaml` を使います。
メッシュ接触のモデルと比較する場合は、評価コマンドに `--robot robot_g1_box` を追加します。
球形を明示する場合は `--robot robot_g1_box_sphere` です。
`trace_policy` でも同じ指定が使えます。
`run_box` は実験設定の `robot:` に従います。

記録からの動画生成では、`render_trace --hand-visual model` がモデルの標準の見た目です。
球形モデルの場合、`--hand-visual mesh` で手のメッシュを表示しても、物理的な接触形状は球のままです。

## 評価条件を変える

```powershell
..\..\.venv\Scripts\python.exe -m boxpush.cli.eval_policy box_p2_walk_push --policies schedule,../../models/rl_m2 --labels schedule,learned --masses 10,20,30,40 --seeds 5 --friction 0.15,0.35 --workers 1 --out ../../outputs/eval_multi.jsonl
```

| 指定 | 意味 |
| --- | --- |
| `--masses` | 箱の質量の一覧（kg） |
| `--seeds` | 各質量と方策の試行回数。番号は0から始まる |
| `--friction` | 各試行で選ぶ箱と床の摩擦係数の範囲 |
| `--x-offset` | 箱の初期位置に加える前後方向のずれの範囲（m）。既定は0〜0.02 |
| `--workers` | 並列プロセス数。最初は1を推奨 |
| `--max-time` | 1試行の時間（秒）。既定は10 |
| `--out` | 追記するJSONLファイル |

評価結果には `success`、`fell`、`box_tipped`、`box_dx`、`t_end`、押す力などが含まれます。
`success` は、時間上限に到達して箱が0.5 m以上前進した場合に記録されます。
ロボットのロールまたはピッチが40度を超えると転倒、箱のロールまたはピッチが20度を超えると箱の転倒として終了します。
正常終了の終了コードだけでは、各試行の成功を判断できません。

## 図を作る

リポジトリ直下で `python -m pip install -r requirements-viz.txt` を実行してから、`armtasks/boxpush` に戻ります。

```powershell
..\..\.venv\Scripts\python.exe tools/plot_eval.py ../../outputs/eval.jsonl ../../outputs/eval.png
```

## 動画を保存する

描画用依存に加えて、FFmpegが必要です。
`ffmpeg -version` が実行できるようにしてから使用します。

```powershell
..\..\.venv\Scripts\python.exe -m boxpush.cli.record box_p2_walk_push --policy ../../models/rl_m2 --mass 20 --out ../../outputs/learned_20kg.mp4
```

Linuxの画面のない環境では、OpenGLの環境に応じて `MUJOCO_GL=egl` または仮想画面を用意してください。
この配布版でのLinux描画と動画の全行程は未確認です。

## 時系列を記録する

```powershell
..\..\.venv\Scripts\python.exe -m boxpush.cli.trace_policy box_p2_walk_push --policies ../../models/rl_m2,schedule --labels learned,schedule --masses 20 --workers 1 --out ../../outputs/trace
..\..\.venv\Scripts\python.exe tools/plot_trace.py ../../outputs/trace
..\..\.venv\Scripts\python.exe tools/make_trace_viewer.py ../../outputs/trace
```

`outputs/trace/index.html` をブラウザーで開くと、記録の一覧を確認できます。
動画を追加する場合は、`python -m boxpush.cli.render_trace ../../outputs/trace` を実行します。

## テスト

リポジトリ直下で `python -m pip install -r requirements-dev.txt` を実行してから、`armtasks/boxpush` で `python -m pytest tests` を実行します。
モデル、力の制御、学習環境、方策の入出力、記録から図を作る経路を確認するテストを同梱しています。
テスト中の短いシミュレーションは、長時間の学習や広い条件での性能を保証するものではありません。

## よくあるエラー

| 症状 | 確認すること |
| --- | --- |
| `No module named boxpush` | 使用しているPythonが仮想環境のものか、リポジトリ直下で基本依存を導入したか |
| 同梱の `lococtrl` が見つからない | フォルダー全体が揃っているか。`armtasks/boxpush` だけを移動していないか |
| XMLまたはメッシュが見つからない | `lococtrl/assets/07_unitree_g1/` を含め、配置を変更していないか |
| ビューアーが表示されない | 画面とOpenGLを利用できるか。端末に出るビューアー起動失敗のメッセージを確認 |
| FFmpegが見つからない | FFmpegの実行ファイルをPATHに追加したか |
| 残差層の方策を `run_box` で使えない | 残差層を持つ追加学習の方策は `eval_policy`、`trace_policy`、`record` で実行 |
