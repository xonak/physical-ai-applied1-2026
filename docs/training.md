# PPOによる再学習

基本動作には再学習は必要ありません。
新しい方策を学習する場合は、リポジトリ直下から追加依存を導入します。

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-train.txt
cd armtasks/boxpush
```

制御とシミュレーションをCPUで実行します。
CPU版PyTorchを選ぶ場合は、先に `python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu` を実行してください。

## 手のモデル

学習には実験設定 `box_p2_walk_push` の標準である球形の手のモデル（`robot_g1_box_sphere.yaml`）を使います。
同梱の方策（`models/rl_m1`、`models/rl_m2`）は従来のメッシュ接触のモデルで学習したもので、球形モデルでの再生を確認しています。
そのため、再学習した方策と同梱の方策は、手のモデルの点で学習条件が異なります。

## 短い動作確認

```powershell
..\..\.venv\Scripts\python.exe -m boxpush.train.train_ppo box_p2_walk_push --preset legacy --n-envs 1 --total-steps 64 --n-steps 32 --batch-size 32 --max-time 0.6 --out ../../outputs/train_smoke
```

これは学習、保存、推論用データの書き出しを確認するための設定です。
押せる方策を得るための学習量ではありません。

## 学習を実行する

同梱の2つの方策と同じ構成で学習するコマンドです。
`--n-envs`（並列環境数）は、使用するPCのCPUに合わせて決めてください。物理コア数から1を引いた程度が目安です。
以下の例は8並列です。

### RL-M1（8指令）

```powershell
..\..\.venv\Scripts\python.exe -m boxpush.train.train_ppo box_p2_walk_push --preset P --n-envs 8 --total-steps 300000 --seed 0 --out ../../outputs/train_rl_m1
```

`--preset P` は、押す力、手先高さ、歩行計画の補正（CMP 参照の前後と左右、重心高さ、着地位置の前後と左右、単脚支持時間）の8指令を学習します。
報酬の姿勢項 $w_{\mathrm{post}}$ は0です（`boxpush.policy.PRESET_REWARD`）。

### RL-M2（押す力と前傾の2指令）

```powershell
..\..\.venv\Scripts\python.exe -m boxpush.train.train_ppo box_p2_walk_push --preset legacy --n-envs 8 --total-steps 300000 --seed 0 --out ../../outputs/train_rl_m2
```

`--preset legacy` は、押す力と前傾（CMP 参照を足に対して前へずらす量）の2指令を学習します。
報酬の姿勢項 $w_{\mathrm{post}}$ は1です。

### 共通の条件

箱の質量は10〜40 kg、摩擦係数は0.15〜0.35、箱の初期位置は前後0〜0.04 mの範囲で、試行ごとに変えます。
指定した判断数は、ロールアウト長と並列数に合わせて切り上がることがあります。
PPOのロールアウトは並列数に比例するため、並列数を変えると、同じ判断数でも方策の更新回数が変わります。
学習結果は乱数、並列数、ライブラリの版、モデルと設定に依存します。同じ成功率になるとは限りません。

| 出力 | 用途 |
| --- | --- |
| `export/params.npz` | NumPyで読む推論用の重み |
| `export/meta.json` | 行動、正規化、判断周期など |
| `ppo.zip` | PPOの学習済みモデル |
| `vecnormalize.pkl` | 学習中に使用した正規化の状態 |
| `train_info.json` | 学習設定と処理時間 |
| `tb/` | TensorBoardの学習ログ |

推論用データの評価例です。

```powershell
..\..\.venv\Scripts\python.exe -m boxpush.cli.eval_policy box_p2_walk_push --policies ../../outputs/train_rl_m1/export,../../outputs/train_rl_m2/export --labels RL-M1,RL-M2 --masses 20 --seeds 1 --workers 1 --out ../../outputs/retrained_eval.jsonl
```

## 学習時と推論時の行動の写像

書き出した方策（`export/`）は、方策ネットの出力 $\mu$ に $\tanh$ を掛けて行動 $a = \tanh(\mu) \in [-1, 1]$ とします（`boxpush.policy.MlpBoxPolicy`）。
一方、学習中のPPO（stable-baselines3）は、$\mu$ に探索の雑音を足した値を $[-1, 1]$ でクリップして環境に渡します。
そのため、学習時と推論時の行動の写像は同一ではありません。$|\mu|$ が小さい範囲ではほぼ一致し、$|\mu|$ が大きいほど推論側の行動が小さくなります（$\mu = 1$ で $0.76$）。
また、学習時の観測の正規化（`VecNormalize`）は正規化後の値を $\pm 10$ でクリップしますが、推論側ではクリップしません。
説明資料の評価値は、すべて書き出した方策（$\tanh$、観測のクリップなし）で得たものです。

## 他の学習条件

方策の構成は `boxpush/policy.py` の `PRESETS` にあります。
同梱方策は、RL-M1 が `P`、RL-M2 が `legacy` です。
`A` は押す力と手先高さ、`B` はさらに安定化層の補正、`C` は脚のトルク補正を含みます。
`P_S`、`P_S_T` は `P` に安定化層と脚のトルクの補正を加えた条件、`S`、`S_T` は補正だけを学習する条件です。
これらの条件は説明資料では評価しておらず、学習済みの重みも同梱していません。
