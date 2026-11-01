# 重い箱を押して歩く：ヒューマノイドの全身制御と強化学習

MuJoCo上のヒューマノイド（Unitree G1）が、床に置かれた箱を押しながら歩くタスクです。
歩行とバランスを保つ全身制御に、強化学習で求めた上位指令（押す力、前傾など）を組み合わせています。
リポジトリ名は `Humanoid_push_box` です。

- **入力**：歩行状態、箱との相対位置と速度、左右の手の力、重心の高さなど40次元。
- **出力**：同梱の学習済み方策は2つです。RL-M1は押す力、手先高さ、歩行計画の補正（CMP参照の前後と左右、重心高さ、着地位置の前後と左右、単脚支持時間）の8指令、RL-M2は押す力と前傾の2指令を出します。前傾は歩行計画のCMP参照を足に対して前方にずらす量で、胴体の角度ではありません。
- **学習**：PPO。箱の質量と摩擦を変えながら学習します。学習済み方策には質量と摩擦を直接入力しません。
- **成功条件**：10秒間転倒せず、箱を0.5 m以上前進させること。ロボットまたは箱が大きく傾くと失敗で終了します。
- **実行手順**：[詳しい使い方](docs/usage.md)、[再学習](docs/training.md)。
- **動画**：[https://youtu.be/ShN3hf7S7lk](https://youtu.be/ShN3hf7S7lk)

学習済みの重みを同梱しているので、最初の実行に再学習やGPUは必要ありません。

## 動作確認の結果

Windowsの新しいPython 3.12.10環境で、箱20 kgと40 kg、摩擦係数0.2、初期位置のずれ0 m、シード0、球形の手のモデルで各1試行を10秒間評価しました。

| 方策 | 箱の質量 | 成功 | 箱の移動量 | ロボットの転倒 |
| --- | --- | --- | --- | --- |
| 手書き制御（`schedule`） | 20 kg | 1/1 | 0.853 m | なし |
| 手書き制御（`schedule`） | 40 kg | 0/1 | 0.559 m | あり（5.61秒） |
| 同梱の学習済み方策（RL-M2） | 20 kg | 1/1 | 0.924 m | なし |
| 同梱の学習済み方策（RL-M2） | 40 kg | 1/1 | 0.981 m | なし |

これは導入と実行の確認です。質量や摩擦を変えた条件全体の成功率を示すものではありません。
同梱の38件のテストも、この環境で通過しています。
PPOについては、64判断の短い学習と推論用データの書き出しまでを確認しています。学習用依存の新規導入と、長時間の学習による再現は確認していません。

## 全体の構成

```text
学習済み方策（10 Hz）
  押す力、前傾（CMP参照のずらし）などの指令
        ↓
歩行計画とバランスの補正
  足の着地位置、重心の参照、手から受ける反力
        ↓
全身制御（500 Hz）
  足裏の力の配分、脚と腕の関節トルク
        ↓
MuJoCo（G1、箱、床）
        └── 関節、姿勢、箱、手首の力の状態をフィードバック
```

G1の脚12関節と腕14関節を動かします。腰の3関節は固定しています。
標準モデルの手は、手先siteを中心とする半径0.065 mの球です。衝突形状と見た目の両方に球を使用します。
手は押す方向を力で制御し、横方向と高さを位置で制御します。
手から受ける反力を歩行制御に反映し、押しながらバランスを保ちます。

## 動作環境

| 用途 | 環境 | 追加で必要なもの |
| --- | --- | --- |
| シミュレーションと学習済み方策の評価 | Windows、Python 3.12、MuJoCo 3.12.0、CPU | `requirements.txt` |
| ビューアーによる表示 | 上記と同じ | OpenGLが利用できる画面環境 |
| 図と動画の作成 | 上記と同じ | `requirements-viz.txt`、動画にはFFmpeg |
| PPOの再学習 | Python 3.12、CPU | `requirements-train.txt` |
| テスト | 基本環境と描画用ライブラリ | `requirements-dev.txt` |

基本依存の版を固定しています。Linux向けの手順も記載していますが、この配布版の検証対象はWindowsです。

## 環境構築

フォルダー全体を配置し、その直下で実行します。
`lococtrl/`、`armtasks/`、`models/` の位置関係を保ってください。

### Windows（PowerShell）

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Linux（bash）

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

以降はWindowsのコマンドを示します。
Linuxでは、リポジトリ直下の `.venv/bin/python` を使用してください。

## 使い方

### 1. 箱押しを画面なしで実行する

`armtasks/boxpush` に移動して、学習済み方策を実行します。

```powershell
cd armtasks/boxpush
..\..\.venv\Scripts\python.exe -m boxpush.cli.run_box box_p2_walk_push --policy ../../models/rl_m2
```

箱の質量は20 kg、床との摩擦係数は0.2です。設定は `configs/robot_g1_box_sphere.yaml`、モデルは `assets/g1_box_sphere.xml`（いずれも `armtasks/boxpush` 内）です。
終了時に転倒の有無や箱の動きなどを表示します。
成功率を確認する場合は、次の評価コマンドを使用してください。

### 2. 動きを表示する

```powershell
..\..\.venv\Scripts\python.exe -m boxpush.cli.run_box box_p2_walk_push --policy ../../models/rl_m2 --viewer --realtime
```

ビューアーを開き、実時間に合わせて再生します。
画面の利用に失敗した場合は、端末にその理由を表示して画面なしで続行します。

### 3. 同じ条件で手書き制御と比較する

```powershell
..\..\.venv\Scripts\python.exe -m boxpush.cli.eval_policy box_p2_walk_push --policies schedule,../../models/rl_m2 --labels schedule,learned --masses 20 --seeds 1 --friction 0.2,0.2 --x-offset 0,0 --workers 1 --out ../../outputs/eval.jsonl
```

各方策を10秒間評価し、成功、転倒、箱の移動量などを `outputs/eval.jsonl` に保存します。
このファイルには結果を追記します。別の評価では、異なる出力ファイル名を指定してください。
`schedule` は箱の質量と摩擦から押す力を計算する比較用制御です。学習済み方策と入力条件が異なります。

### 説明資料の主評価を再現する

説明資料の成功数（Analytic、RL-M1、RL-M2 の 3 方式 × 4 質量 × 3 試行）は、次のコマンドで再現します。

```powershell
..\..\.venv\Scripts\python.exe -m boxpush.cli.eval_policy box_p2_walk_push --robot robot_g1_box --policies analytic,../../models/rl_m1,../../models/rl_m2 --labels Analytic,RL-M1,RL-M2 --masses 10,20,30,40 --seeds 3 --friction 0.15,0.35 --x-offset 0,0 --workers 1 --out ../../outputs/eval_main.jsonl
```

説明資料の結果は、手の接触を従来のメッシュ形状としたモデルで得ています。
実験設定の既定は球形の手のモデルなので、`--robot robot_g1_box` を指定してください。
箱の初期位置はずらさない条件（`--x-offset 0,0`）で評価しています。
計算環境による数値の差で軌道は完全には一致しません。MuJoCo 3.12.0 での再実行では、36 試行中 35 試行で成否が説明資料と一致しました（転倒の境界に近い 40 kg の 1 試行が異なる）。
36 試行を順に実行するため、`--workers` を増やすと短縮できます。

### 4. 再学習する

追加依存の導入と短い学習確認は [再学習の手順](docs/training.md) にまとめています。
同梱方策と同じ構成で学習する場合は、RL-M1は `--preset P`、RL-M2は `--preset legacy` を指定します。
既定の `--preset A` は押す力と手先高さを学ぶ別の条件です。
並列数（`--n-envs`）は使用するPCに合わせて決めてください。

## ファイルの構成

```text
Humanoid_push_box/
├─ README.md
├─ requirements*.txt       用途ごとの依存
├─ docs/                   実行と再学習の説明
├─ models/rl_m1/           RL-M1（8 指令）の重みと正規化情報
├─ models/rl_m2/           RL-M2（押付け力と前傾の 2 指令）の重みと正規化情報
├─ lococtrl/               歩行計画、全身制御、G1モデル
├─ armtasks/boxpush/        箱押し環境、設定、評価、学習、テスト
└─ outputs/                実行結果の保存先
```

モデルとメッシュを同梱しているため、追加のモデル取得は不要です。
`g1_box.xml` は従来のメッシュ接触モデルとして同梱しています。
XMLを再生成する場合は、`armtasks/boxpush` で `python -m boxpush.model` を実行します。
モデルの利用条件は [外部データの説明](THIRD_PARTY_NOTICES.md) を参照してください。
