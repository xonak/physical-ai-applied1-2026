"""! @file model.py
@brief 箱押し用の G1 派生 MJCF（脚 12 + 腕 14、手の衝突、手首 F/T）を lococtrl の生成器で作る。モデルの変更を行なうスクリプト

原典は lococtrl/assets/07_unitree_g1/g1.xml（MuJoCo Menagerie、無改変）。生成物は
armtasks/boxpush/assets/g1_box.xml に書き、メッシュは原典ディレクトリを相対参照する
（`lococtrl.sim.mjcf_adapt.adapt_mjcf` が別ディレクトリ出力時に meshdir を書き換える）。

使い方:

```bash
python -m boxpush.model      # g1_box.xml と g1_box_sphere.xml を生成
```

`g1_box_sphere.xml` は手の衝突をゴム手メッシュでなく球（中心 = 手先 site、半径 0.065 m）にした
別モデル。標準の箱押し実験では g1_box_sphere.xml を使用する。

**原典を更新したら再生成すること。** 生成物は手で編集しない。
"""

from __future__ import annotations

from pathlib import Path

from lococtrl.sim.mjcf_adapt import AdaptSpec, adapt_mjcf

from . import LOCOCTRL_ROOT, PKG_ROOT

__all__ = ["G1_LEG_JOINTS", "G1_ARM_JOINTS", "HAND_COLLISION_BIT", "HAND_SPHERE_RADIUS",
           "make_g1_box", "make_g1_box_sphere"]

G1_LEG_JOINTS = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
G1_ARM_JOINTS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                 "wrist_roll", "wrist_pitch", "wrist_yaw")
G1_WAIST_JOINTS = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
## 原典 g1.xml の actuatorfrcrange [N*m]（左右同一）。手首 pitch / yaw は 5 N*m しかない
G1_TAU_MAX_PER_LEG = (88.0, 139.0, 88.0, 139.0, 50.0, 50.0)
G1_TAU_MAX_PER_ARM = (25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0)
## 手の衝突ビット。lococtrl.sim.mujoco_env.HAND_COLLISION_BIT（箱側の conaffinity）と一致させる
HAND_COLLISION_BIT = 4

SRC = LOCOCTRL_ROOT / "assets" / "07_unitree_g1" / "g1.xml"
DST = PKG_ROOT / "assets" / "g1_box.xml"
DST_SPHERE = PKG_ROOT / "assets" / "g1_box_sphere.xml"
## 手の衝突球の半径 [m]。直径 ≒ 手首〜指先（0.173 − 0.0415）。前端 0.106 + 0.065 = 0.171 ≒ 指先 0.173
HAND_SPHERE_RADIUS = 0.065


def make_g1_box(src: Path = SRC, dst: Path = DST, hand_collision: str = "mesh") -> Path:
    """! @brief 箱押しタスク用の G1 派生 MJCF（脚 12 + 腕 14 = 26 自由度）を生成する。
    @param src 原典 g1.xml [-]
    @param dst 出力先 [-]
    @param hand_collision 手の衝突形状 [-]。"mesh"（ゴム手の凸包）か "sphere"（make_g1_box_sphere）
    @return 生成したパス
    @note lococtrl の `make_g1()`（下半身 12）との違いは 3 つで、それ以外（足裏 box、motor 化、
          床衝突の制限、IMU、床、関節トルクセンサ、足裏 F/T）は同じ設定を使う。
          (a) 削除する関節を腰 3 だけにし、腕 14 を残す（胴体は骨盤に固定のまま）。
          (b) 手のゴム（`*_rubber_hand`）に衝突判定を戻す。contype = 4、conaffinity = 0 なので、
              床・胴体・もう一方の手とは当たらず、conaffinity に 4 を含む物体（箱は
              contype = 1、conaffinity = 5）とだけ当たる。
          (c) 手先 site（ゴム部分の凸包中心、手ボディ系 (0.106, ±0.004, 0.003)）と
              手首 6 軸 F/T センサを足す。
          アクチュエータ順は 左脚 6、右脚 6、左腕 7、右腕 7。
    @note **代償**: 前腕・上腕・胴体は箱と当たらない（衝突クラスがビット 2 のため）。
          押す動作で腕が箱に食い込んでも反力は出ない。
    """
    legs = ("left", "right")
    joints = tuple(f"{leg}_{j}_joint" for leg in legs for j in G1_LEG_JOINTS) + tuple(
        f"{side}_{j}_joint" for side in legs for j in G1_ARM_JOINTS
    )
    spec = AdaptSpec(
        leg_names=legs,
        foot_body_fmt="{leg}_ankle_roll_link",
        foot_class="foot",
        collision_class="collision",
        base_body="pelvis",
        lock_joints=G1_WAIST_JOINTS,
        foot_box_pos=(0.035, 0.0, -0.03),
        foot_box_size=(0.085, 0.0275, 0.005),
        foot_site_pos=(0.035, 0.0, -0.035),
        tau_max=G1_TAU_MAX_PER_LEG * 2 + G1_TAU_MAX_PER_ARM * 2,
        actuator_order=joints,
        imu_pos=(0.04525, 0.0, -0.08339),
        joint_torque_sensors=True,
        foot_ft_sensors=True,
        hand_body_fmt="{hand}_wrist_yaw_link",
        hand_names=legs,
        hand_site_pos=((0.1061, -0.0037, 0.0031), (0.1061, 0.0037, 0.0031)),
        hand_collision_bit=HAND_COLLISION_BIT,
        hand_ft_sensors=True,
        hand_collision=hand_collision,
        hand_sphere_radius=HAND_SPHERE_RADIUS,
        note="原典: lococtrl/assets/07_unitree_g1/g1.xml（MuJoCo Menagerie unitree_g1）。"
             "箱押し用: 腰固定、腕 14 自由度、手の衝突と手首 F/T。生成: python -m boxpush.model"
             + ("。手の衝突は球（中心 = 手先 site、半径 %.3f m）" % HAND_SPHERE_RADIUS
                if hand_collision == "sphere" else ""),
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    return adapt_mjcf(src, dst, spec)


def make_g1_box_sphere(src: Path = SRC, dst: Path = DST_SPHERE) -> Path:
    """! @brief 手の衝突を球にした別モデル g1_box_sphere.xml を生成する。
    @param src 原典 g1.xml [-]
    @param dst 出力先 [-]
    @return 生成したパス
    @note g1_box.xml との違いは手の衝突 geom だけ（名前 `left_hand` / `right_hand`、contype、
          priority、摩擦は同じ）。衝突球は手先 site を中心とする半径 HAND_SPHERE_RADIUS の球で、
          接触法線が常に site を通るため、押付け力 \\f$f_n\\f$ が site まわりに作るモーメントは
          \\f$\\boldsymbol{r}\\times\\boldsymbol{f}_n = 0\\f$ になる。前端は指先とほぼ同じ位置
          （差 2 mm）なので、g1_box.xml で学習した方策をそのまま再生できる（20 kg と 40 kg で確認）。
          visual は手メッシュを透明（α=0）で残し、visual 球 `*_hand_vis_sphere` を足す。
          描画側（render_trace --hand-visual）で切り替えられる。
    """
    return make_g1_box(src, dst, hand_collision="sphere")


if __name__ == "__main__":
    print("生成:", make_g1_box())
    print("生成:", make_g1_box_sphere())
