# 外部モデルの出典と利用条件

## Unitree G1

`lococtrl/assets/07_unitree_g1/` のG1モデルとメッシュは、MuJoCo Menagerieに収録されたUnitree G1モデルを使用しています。
原モデルはUnitree RoboticsのG1 robot descriptionに由来します。
著作権者はHangZhou YuShu TECHNOLOGY CO.,LTD.（Unitree Robotics）です。
再配布条件と免責事項の全文は、同ディレクトリの [LICENSE](lococtrl/assets/07_unitree_g1/LICENSE) に収録しています。

`g1_loco.xml`、`armtasks/boxpush/assets/g1_box.xml`、`armtasks/boxpush/assets/g1_box_sphere.xml` は原モデルから生成した派生モデルです。
箱押しモデルでは腰を固定し、脚12関節と腕14関節を駆動し、手の接触と手首の力覚センサを追加しています。
標準の球形の手のモデルでは、手先siteを中心とする半径0.065 mの球で箱と接触します。
原モデルと派生モデルの関係は `armtasks/boxpush/boxpush/model.py` および `lococtrl/lococtrl/sim/mjcf_adapt.py` に定義されています。

## Pythonライブラリ

実行時に導入するPythonライブラリは、各ライブラリの利用条件に従って使用してください。
ライブラリ本体、仮想環境、インストールキャッシュは配布フォルダーに含めていません。
