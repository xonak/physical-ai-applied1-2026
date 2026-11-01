"""! @package lococtrl
@brief 二足ロボット（Unitree G1）の歩行制御の検証用実装。

層構成:
- common/         : 型・座標変換・設定・タイミング（依存の最下層）
- model_dynamics/ : 剛体動力学（バックエンド抽象化。既定は MuJoCo）
- sim/            : MuJoCo シミュレータとビューア（**mjModel を触ってよい層**）
- control/        : DCM 安定化・全身制御・手先タスク
- gait/           : 歩容スケジューラ・遊脚軌道
- planner/        : DCM 軌道・オンライン足配置計画
- bench/          : 地形・評価指標
- policy/         : 学習方策アダプタ
- teleop/         : 操作入力（速度指令）
- cli/            : 実行エントリ
"""

__version__ = "0.1.0"
