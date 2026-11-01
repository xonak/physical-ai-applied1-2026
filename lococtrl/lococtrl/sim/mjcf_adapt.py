"""! @file mjcf_adapt.py
@brief 外部 MJCF（MuJoCo Menagerie 等）を LocoCtrl が要求するモデル仕様に変換する。
足先の site / geom に名前を付ける、position アクチュエータを motor に変える、IMU センサを足す、床を worldbody に足す、など。

Menagerie のモデルは「見る・眺める」用に整っているが、本リポジトリの制御スタックが
要求する次の 4 点を満たさない。手で書き換えると原典との差分が追えなくなるので、
**生成器を通して派生 MJCF を作る**方針を取る。

1. **足先の site / geom に名前が無い。** `index_map.build_index_map()` は
   `cfg.foot_sites` / `cfg.foot_geoms` を名前引きするため、名前が要る
2. **アクチュエータが position 型。** 本実装はトルク指令なので `motor` でなければ
   ならない。`ControlOutput.tau` [N*m] がそのまま `ctrl` に入る
3. **IMU センサが無い。** 状態推定は `imu_acc` / `imu_gyro` を名前引きする
4. **床が別ファイル（scene.xml）にある。** `MujocoEnv` は 1 ファイルを読み、
   `floor` という名前の geom を接触判定に使う

**このファイルは MJCF のテキスト（XML）だけを扱い、mjModel / mjData には触れない。**
生成物は原典と同じディレクトリへ書く。`compiler` の `meshdir` / `texturedir` が
相対パスなので、別の場所へ書くとメッシュが解決できなくなるためである。

記号: nf 脚数 [-]、n 駆動関節数 [-]。
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["AdaptSpec", "adapt_mjcf"]


@dataclass
class AdaptSpec:
    """! @brief MJCF 適合のパラメータ。

    @param leg_names      脚名 shape (nf,) [-]。生成する名前の接頭辞になる
    @param foot_body_fmt  足先を持つボディ名の書式 [-]
    @param foot_class     足先 geom の default クラス名 [-]
    @param base_body      浮遊ベースのボディ名 [-]
    @param tau_max        関節トルク上限 shape (n,) [N*m]。motor の ctrlrange になる
    @param actuator_order アクチュエータを並べる関節名 shape (n,) [-]。
                          None なら原典の `<actuator>` の順序を保つ
    @param site_size      生成する足先 site の描画半径 [m]
    @param imu_site       IMU site の名前 [-]
    @param imu_pos        IMU site の位置（ベース系 {B}）shape (3,) [m]
    @param add_floor      床・光源・材質を worldbody に足すか [-]
    @param floor_friction 床の摩擦 [-]（slide, spin, roll）
    @param collision_class 衝突 geom の default クラス名 [-]
    @param foot_only_collision 足先以外の脚 geom を床と衝突させないか [-]。
                          本実装の接触モデルは**点接触の足先のみ**を前提にする。
                          すねが床に触れて荷重を負担すると、
                          f_meas も接触判定も破綻する
    @param zero_keyframe_ctrl keyframe の ctrl をゼロにするか [-]。
                          position から motor へ変えると ctrl の意味が
                          関節角 [rad] からトルク [N*m] に変わるため、既定で True
    """

    leg_names: tuple[str, ...]
    foot_body_fmt: str = "{leg}_calf"
    foot_class: str = "foot"
    base_body: str = "trunk"
    tau_max: tuple[float, ...] = ()
    actuator_order: tuple[str, ...] | None = None
    site_size: float = 0.008
    imu_site: str = "imu"
    imu_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    collision_class: str = "collision"
    foot_only_collision: bool = True
    add_floor: bool = True
    floor_friction: tuple[float, float, float] = (0.8, 0.02, 0.01)
    zero_keyframe_ctrl: bool = True
    ## 削除する関節名 shape (k,) [-]。二足で上半身を固定するために使う。
    ## joint タグと、その関節を駆動する actuator の両方を消す。weld（equality）
    ## ではなく削除にするのは、自由度そのものを減らして nv を小さく保つためである。
    lock_joints: tuple[str, ...] = ()
    ## 足裏を box 1 個に置き換えるときの (pos, size) [m]。None なら原典のまま。
    ## Menagerie の二足は足裏を複数の球で表すが、本実装の `foot_forces()` は
    ## 1 脚 1 geom を前提にするため、外接する box 1 個にまとめる。
    ## box-plane の接触は複数の接触点を生むので、足裏内 CoP は失われない。
    foot_box_pos: tuple[float, float, float] | None = None
    foot_box_size: tuple[float, float, float] | None = None
    ## 足先 site の位置 [m]。None なら足 geom と同じ位置に置く。
    foot_site_pos: tuple[float, float, float] | None = None
    ## 関節トルクセンサを付けるか [-]。各駆動関節の子ボディに site を置き、
    ## `<torque>` センサ（親子ボディ間の相互作用トルク、site 系表現）を足す。
    ## 関節軸に射影した値が **リンク側の関節トルク**（受動項・外力を含む）になり、
    ## 実機のトルクセンサ付き関節（DLR TORO 等）に相当する。
    ## `jointactuatorfrc` はアクチュエータ指令そのものを返すだけなので使わない。
    joint_torque_sensors: bool = False
    ## 足裏 6 軸力覚センサを付けるか [-]。足先ボディ（foot_body_fmt）の原点に
    ## site を置き `<force>` と `<torque>` を足す。親子ボディ間の相互作用レンチなので、
    ## 足首に付けた実機の F/T センサと同じく、床反力に足の慣性・重力が乗る。
    foot_ft_sensors: bool = False
    ## --- 手（箱押しタスク用）---
    ## 手を持つボディ名の書式 [-]。None なら手の処理をしない（既存モデルは不変）。
    hand_body_fmt: str | None = None
    ## 手の名前 shape (nh,) [-]。生成する名前の接頭辞になる（例: left, right）
    hand_names: tuple[str, ...] = ()
    ## 衝突判定を戻す手の visual メッシュ名の書式 [-]
    hand_mesh_fmt: str = "{hand}_rubber_hand"
    ## 手先 site の位置（手ボディ系）shape (nh,3) [m]。全身制御の手先タスク点になる
    hand_site_pos: tuple[tuple[float, float, float], ...] = ()
    ## 手の衝突ビット [-]。手は contype = bit、conaffinity = 0 にする。
    ## 床（1）・胴体の衝突クラス（2）・もう一方の手とは当たらず、
    ## conaffinity にこのビットを含む物体とだけ当たる。箱は contype = 1、conaffinity = 1|4 = 5
    hand_collision_bit: int = 4
    ## 手の摩擦（slide, spin, roll）[-]。ゴムなので床より高め。手の geom は priority 3 にするので、
    ## 箱（priority 2、床との摩擦を箱側で決めるため）との接触でもこの値が使われる
    hand_friction: tuple[float, float, float] = (1.0, 0.02, 0.001)
    ## 手首 6 軸 F/T センサを付けるか [-]。手ボディの原点（手首 yaw 関節）に site を置く
    hand_ft_sensors: bool = False
    ## 手の衝突形状 [-]。"mesh": ゴム手メッシュの凸包（指先で押す）、
    ## "sphere": 手先 site を中心とする球（接触法線が常に site を通るので、押付け力 \f$f_n\f$ が
    ## site まわりに作るモーメントは \f$\boldsymbol{r}\times\boldsymbol{f}_n = 0\f$）。
    ## "sphere" では visual の手メッシュを透明（rgba α=0）にし、visual 球 `<hand>_hand_vis_sphere`
    ## を足す。描画側で α を書き換えれば手の見た目を切り替えられる
    hand_collision: str = "mesh"
    ## 衝突球の半径 [m]（hand_collision="sphere" のとき）。G1 では手首〜指先 0.173 − 手の根元 0.0415
    ## ≒ 直径 0.13 とし、球の前端 0.106+0.065 = 0.171 が指先 0.173 とほぼ一致する
    hand_sphere_radius: float = 0.065
    ## 衝突球の中心（手ボディ系）shape (nh,3) [m]。空なら hand_site_pos
    hand_sphere_center: tuple[tuple[float, float, float], ...] = ()
    ## visual 球の色 rgba [-]（hand_collision="sphere" のとき）
    hand_sphere_rgba: tuple[float, float, float, float] = (0.35, 0.35, 0.38, 1.0)
    ## 生成物の先頭に入れる注記（原典を書き換えた事実を残す）
    note: str = field(default="")


def _parent_map(root: ET.Element) -> dict[ET.Element, ET.Element]:
    """! @brief 親要素の索引を作る（ElementTree には親参照が無い）。
    @param root ルート要素 [-]
    @return 子 -> 親 の辞書
    """
    return {c: p for p in root.iter() for c in p}


def _default_attrs(root: ET.Element, cls: str, tag: str) -> dict[str, str]:
    """! @brief default クラスの継承を解決して、ある要素種別の既定属性を返す。
    @param root ルート要素 [-]
    @param cls  default クラス名 [-]
    @param tag  要素種別（"geom" など）[-]
    @return 属性の辞書
    @note MJCF の `<default>` は入れ子で継承する。`<geom class="foot"/>` のように
          属性が 1 つも書かれていない要素の実効値を知るには、外側の default から
          順に重ねる必要がある。
    """
    parent = _parent_map(root)
    node = None
    for d in root.iter("default"):
        if d.get("class") == cls:
            node = d
            break
    if node is None:
        raise KeyError(f"default クラス '{cls}' が MJCF に無い")

    chain: list[ET.Element] = []
    n: ET.Element | None = node
    while n is not None and n.tag == "default":
        chain.append(n)
        n = parent.get(n)

    attrs: dict[str, str] = {}
    for d in reversed(chain):  # 外側（一般）から内側（特殊）へ重ねる
        e = d.find(tag)
        if e is not None:
            attrs.update(e.attrib)
    attrs.pop("class", None)
    return attrs


def _find_body(root: ET.Element, name: str) -> ET.Element:
    """! @brief 名前でボディ要素を引く。
    @param root ルート要素 [-]
    @param name ボディ名 [-]
    @return body 要素
    """
    for b in root.iter("body"):
        if b.get("name") == name:
            return b
    raise KeyError(f"body '{name}' が MJCF に無い")


def _name_feet(root: ET.Element, spec: AdaptSpec) -> None:
    """! @brief 足先 geom に名前を付け、同位置に site を足す。
    @param root ルート要素 [-]
    @param spec AdaptSpec [-]
    @return なし
    @note geom の位置は `<default class="foot">` 側に書かれていて要素には無い、
          という書き方が Menagerie では普通である。site を「足先と同じ場所」に
          置くため、default を解決して pos を取り出す。
    """
    d = _default_attrs(root, spec.foot_class, "geom")
    pos_default = d.get("pos", "0 0 0")

    for leg in spec.leg_names:
        body = _find_body(root, spec.foot_body_fmt.format(leg=leg))
        target = None
        for g in body.findall("geom"):
            if g.get("class") == spec.foot_class:
                target = g
                break
        if target is None:
            raise KeyError(f"body '{body.get('name')}' に class='{spec.foot_class}' の geom が無い")

        if spec.foot_box_pos is not None and spec.foot_box_size is not None:
            # 同クラスの geom をすべて取り除き、外接 box を 1 個だけ置く。
            for g in list(body.findall("geom")):
                if g.get("class") == spec.foot_class:
                    body.remove(g)
            target = ET.SubElement(body, "geom", {"class": spec.foot_class})
            target.set("type", "box")
            target.set("pos", " ".join(f"{v}" for v in spec.foot_box_pos))
            target.set("size", " ".join(f"{v}" for v in spec.foot_box_size))

        target.set("name", f"{leg}_foot")
        pos = target.get("pos", pos_default)
        if spec.foot_site_pos is not None:
            pos = " ".join(f"{v}" for v in spec.foot_site_pos)
        ET.SubElement(
            body,
            "site",
            {
                "name": f"{leg}_foot_site",
                "pos": pos,
                "size": f"{spec.site_size}",
                "rgba": "0 1 0 1",
            },
        )


def _lock_joints(root: ET.Element, spec: AdaptSpec) -> None:
    """! @brief 指定した関節とその actuator を削除し、自由度を固定する。
    @param root ルート要素 [-]
    @param spec AdaptSpec [-]
    @return なし
    @note 二足（G1）で上半身（腰 3 + 腕 14）を固定するために使う。
          equality/weld ではなく **joint タグごと削除** するのは、
          nv を実際に減らして動力学計算と IK の次元を小さく保つためである。
          weld は拘束力を解く必要があり、計算も挙動も重くなる。
          削除してもリンクの質量と慣性は body に残るので、
          上半身の重さは重心と慣性行列に正しく効く。
    """
    if not spec.lock_joints:
        return
    targets = set(spec.lock_joints)
    parents = _parent_map(root)
    removed: set[str] = set()
    for j in list(root.iter("joint")):
        name = j.get("name")
        if name in targets:
            parents[j].remove(j)
            removed.add(name)
    missing = targets - removed
    if missing:
        raise KeyError(f"削除対象の関節が MJCF に無い: {sorted(missing)}")

    # 対応する actuator も消す（joint が無いと読み込みで失敗する）
    for act_parent in root.iter("actuator"):
        for a in list(act_parent):
            if a.get("joint") in targets or a.get("name") in targets:
                act_parent.remove(a)

    # keyframe を削除する。qpos / ctrl の長さが nq / nu と合わなくなり、
    # そのままでは MuJoCo が読み込みを拒否するためである。
    # 切り詰めではなく削除にするのは、qpos のどの要素がどの関節に対応するかが
    # 文書順に依存して脆いからで、初期姿勢は configs の q_nominal で与える
    # （`MujocoEnv.reset()` は keyframe が無ければ無視する）。
    for parent in list(root):
        if parent.tag == "keyframe":
            root.remove(parent)


def _foot_only_collision(root: ET.Element, spec: AdaptSpec) -> None:
    """! @brief 足先以外の脚 geom を床と衝突させないようにする。
    @param root ルート要素 [-]
    @param spec AdaptSpec [-]
    @return なし
    @note 本実装の接触モデルは**点接触の足先のみ**である。すねや腿が床を
          突くと、接触力の集計（f_meas）も接触判定も WBC の接触ヤコビアンも
          前提を失う。

          実現方法は contype / conaffinity のビットマスクである。
          衝突クラス全体を group 2（contype=2, conaffinity=2）に落とし、
          足先だけ group 1（=1, =1）に戻す。床は 1 なので
          `1 & 2 == 0` となり、すね同士の自己衝突は残したまま
          床との衝突だけが消える。

          **代償**: 段差の縁にすねが当たらなくなる。点接触を仮定する
          制御則にとっては望ましい単純化だが、より高い段差では
          「すねで乗り上げる」挙動が再現されなくなる。
    """
    if not spec.foot_only_collision:
        return
    node = None
    for d in root.iter("default"):
        if d.get("class") == spec.collision_class:
            node = d
            break
    if node is None:
        raise KeyError(f"default クラス '{spec.collision_class}' が MJCF に無い")
    g = node.find("geom")
    if g is None:
        g = ET.SubElement(node, "geom")
    g.set("contype", "2")
    g.set("conaffinity", "2")

    for leg in spec.leg_names:
        body = _find_body(root, spec.foot_body_fmt.format(leg=leg))
        for e in body.findall("geom"):
            if e.get("name") == f"{leg}_foot":
                e.set("contype", "1")
                e.set("conaffinity", "1")


def _add_imu(root: ET.Element, spec: AdaptSpec) -> None:
    """! @brief ベースに IMU site を置き、sensor ブロックを足す。
    @param root ルート要素 [-]
    @param spec AdaptSpec [-]
    @return なし
    """
    base = _find_body(root, spec.base_body)
    if not any(s.get("name") == spec.imu_site for s in base.findall("site")):
        ET.SubElement(
            base,
            "site",
            {"name": spec.imu_site, "pos": " ".join(f"{v}" for v in spec.imu_pos), "size": "0.01"},
        )
    sensor = root.find("sensor")
    if sensor is None:
        sensor = ET.SubElement(root, "sensor")
    have = {s.get("name") for s in sensor}
    if "imu_quat" not in have:
        ET.SubElement(
            sensor, "framequat", {"name": "imu_quat", "objtype": "site", "objname": spec.imu_site}
        )
    if "imu_gyro" not in have:
        ET.SubElement(sensor, "gyro", {"name": "imu_gyro", "site": spec.imu_site})
    if "imu_acc" not in have:
        ET.SubElement(sensor, "accelerometer", {"name": "imu_acc", "site": spec.imu_site})


def _add_torque_sensors(root: ET.Element, spec: AdaptSpec) -> None:
    """! @brief 関節トルクセンサと足裏 F/T センサを足す。
    @param root ルート要素 [-]
    @param spec AdaptSpec [-]
    @return なし
    @note 関節トルクセンサ: 駆動関節 j の子ボディに site `jts_<joint>` を関節位置に置き、
          `<torque name="jts_<joint>" site=...>` を足す。site は回転を持たないので
          センサ系 = 子ボディ系であり、`mjModel.jnt_axis` との内積がそのまま
          関節軸まわりのトルクになる（符号は `MujocoEnv.joint_torque_meas` で合わせる）。
    @note 足裏 F/T: 足先ボディの原点（足首）に site `ft_<leg>` を置き、
          `ft_<leg>_force` / `ft_<leg>_torque` を足す。足裏中心への移動は読む側で行う。
    @note 名前が既にあれば足さない（再生成で重複しない）。
    """
    sensor = root.find("sensor")
    if sensor is None:
        sensor = ET.SubElement(root, "sensor")
    have = {s.get("name") for s in sensor}
    parent = _parent_map(root)

    if spec.joint_torque_sensors:
        act = root.find("actuator")
        if act is None:
            raise KeyError("actuator ブロックが MJCF に無い")
        driven = [e.get("joint") for e in act if e.get("joint")]
        joints = {j.get("name"): j for j in root.iter("joint") if j.get("name")}
        for jname in driven:
            j = joints.get(jname)
            if j is None:
                raise KeyError(f"駆動関節 '{jname}' が MJCF に無い")
            body = parent[j]
            site_name = f"jts_{jname}"
            if not any(s.get("name") == site_name for s in body.findall("site")):
                ET.SubElement(
                    body, "site",
                    {"name": site_name, "pos": j.get("pos", "0 0 0"), "size": "0.004",
                     "rgba": "1 0 0 0.3"},
                )
            if site_name not in have:
                ET.SubElement(sensor, "torque", {"name": site_name, "site": site_name})

    if spec.foot_ft_sensors:
        for leg in spec.leg_names:
            body = _find_body(root, spec.foot_body_fmt.format(leg=leg))
            site_name = f"ft_{leg}"
            if not any(s.get("name") == site_name for s in body.findall("site")):
                ET.SubElement(
                    body, "site",
                    {"name": site_name, "pos": "0 0 0", "size": "0.006", "rgba": "0 0 1 0.3"},
                )
            if f"{site_name}_force" not in have:
                ET.SubElement(sensor, "force", {"name": f"{site_name}_force", "site": site_name})
            if f"{site_name}_torque" not in have:
                ET.SubElement(sensor, "torque", {"name": f"{site_name}_torque", "site": site_name})


def _add_hands(root: ET.Element, spec: AdaptSpec) -> None:
    """! @brief 手の visual メッシュに衝突判定を戻し、手先 site と手首 F/T センサを足す。
    @param root ルート要素 [-]
    @param spec AdaptSpec [-]
    @return なし
    @note Menagerie の G1 は手のゴム（`*_rubber_hand`）が visual のみで、衝突 geom が無い。
          同じメッシュ・同じ pos/quat で衝突 geom `<hand>_hand` を足す（MuJoCo はメッシュの
          凸包で衝突を解く）。contype = `hand_collision_bit`、conaffinity = 0 にするので、
          床・自分の胴体・もう一方の手とは当たらず、conaffinity にそのビットを含む物体（箱）
          とだけ当たる（MuJoCo は contype1 & conaffinity2 または contype2 & conaffinity1 で判定）。
    @note 手先 site `<hand>_hand_site` は手先インピーダンスの作用点。
          手首 F/T `ft_<hand>_hand_force` / `_torque` は手ボディと親（手首 pitch）の
          相互作用レンチなので、接触力に手の重力・慣性が乗る（足裏 F/T と同じ扱い）。
    @note 名前が既にあれば足さない（再生成で重複しない）。
    @note hand_collision="sphere" のときは衝突 geom を球（中心 hand_sphere_center、半径
          hand_sphere_radius）にし、visual の手メッシュは rgba α=0 で残す（描画側で戻せる）。
          衝突 geom の名前・contype・priority・摩擦は mesh のときと同じ。
    """
    if spec.hand_body_fmt is None:
        return
    if spec.hand_site_pos and len(spec.hand_site_pos) != len(spec.hand_names):
        raise ValueError("hand_site_pos の長さが hand_names と違う")
    if spec.hand_collision not in ("mesh", "sphere"):
        raise ValueError(f"hand_collision は mesh か sphere: {spec.hand_collision!r}")
    centers = spec.hand_sphere_center or spec.hand_site_pos
    if spec.hand_collision == "sphere" and len(centers) != len(spec.hand_names):
        raise ValueError("hand_collision='sphere' には hand_sphere_center か hand_site_pos が要る")
    bit = str(int(spec.hand_collision_bit))
    sensor = root.find("sensor")
    if sensor is None:
        sensor = ET.SubElement(root, "sensor")
    have = {s.get("name") for s in sensor}
    for k, hand in enumerate(spec.hand_names):
        body = _find_body(root, spec.hand_body_fmt.format(hand=hand))
        mesh = spec.hand_mesh_fmt.format(hand=hand)
        vis = None
        for g in body.findall("geom"):
            if g.get("mesh") == mesh:
                vis = g
                break
        if vis is None:
            raise KeyError(f"body '{body.get('name')}' に mesh='{mesh}' の geom が無い")
        gname = f"{hand}_hand"
        if not any(g.get("name") == gname for g in body.findall("geom")):
            attrs = {"name": gname, "class": spec.collision_class}
            if spec.hand_collision == "mesh":
                attrs["mesh"] = mesh
            attrs.update({"contype": bit, "conaffinity": "0", "priority": "3", "condim": "3",
                          "friction": " ".join(f"{v}" for v in spec.hand_friction)})
            if spec.hand_collision == "mesh":
                for a in ("pos", "quat"):
                    if vis.get(a) is not None:
                        attrs[a] = vis.get(a)
            else:
                attrs["type"] = "sphere"
                attrs["size"] = f"{spec.hand_sphere_radius}"
                attrs["pos"] = " ".join(f"{v}" for v in centers[k])
            ET.SubElement(body, "geom", attrs)
        if spec.hand_collision == "sphere":
            vname = f"{hand}_hand_vis_sphere"
            vis.set("rgba", "0.5 0.5 0.5 0")   # 手メッシュは透明にして残す
            if not any(g.get("name") == vname for g in body.findall("geom")):
                ET.SubElement(body, "geom", {
                    "name": vname, "class": "visual", "type": "sphere",
                    "size": f"{spec.hand_sphere_radius}",
                    "pos": " ".join(f"{v}" for v in centers[k]),
                    "rgba": " ".join(f"{v}" for v in spec.hand_sphere_rgba)})
        if spec.hand_site_pos:
            sname = f"{hand}_hand_site"
            if not any(s.get("name") == sname for s in body.findall("site")):
                ET.SubElement(body, "site", {
                    "name": sname, "pos": " ".join(f"{v}" for v in spec.hand_site_pos[k]),
                    "size": f"{spec.site_size}", "rgba": "0 1 0 1"})
        if spec.hand_ft_sensors:
            fname = f"ft_{hand}_hand"
            if not any(s.get("name") == fname for s in body.findall("site")):
                ET.SubElement(body, "site", {"name": fname, "pos": "0 0 0", "size": "0.006",
                                             "rgba": "0 0 1 0.3"})
            if f"{fname}_force" not in have:
                ET.SubElement(sensor, "force", {"name": f"{fname}_force", "site": fname})
            if f"{fname}_torque" not in have:
                ET.SubElement(sensor, "torque", {"name": f"{fname}_torque", "site": fname})


def _to_motor_actuators(root: ET.Element, spec: AdaptSpec) -> None:
    """! @brief position アクチュエータをトルク（motor）に差し替える。
    @param root ルート要素 [-]
    @param spec AdaptSpec [-]
    @return なし
    @note **class 属性は落とす。** MJCF の default は要素種別ごとなので、
          `<position>` 用の既定値（kp や ctrlrange）は motor には適用されないが、
          クラスを残すと読む側が「効いている」と誤解する。ctrlrange は
          `spec.tau_max` [N*m] から明示的に書く。
    """
    act = root.find("actuator")
    if act is None:
        raise KeyError("actuator ブロックが MJCF に無い")

    entries: list[tuple[str, str]] = []  # (アクチュエータ名, 関節名)
    for e in list(act):
        joint = e.get("joint")
        if joint is None:
            raise ValueError(f"joint を持たないアクチュエータがある: {e.tag} {e.attrib}")
        entries.append((e.get("name") or joint, joint))
        act.remove(e)

    if spec.actuator_order is not None:
        table = {j: n for n, j in entries}
        missing = [j for j in spec.actuator_order if j not in table]
        if missing:
            raise KeyError(f"actuator_order の関節がアクチュエータに無い: {missing}")
        entries = [(table[j], j) for j in spec.actuator_order]

    if len(spec.tau_max) not in (0, len(entries)):
        raise ValueError(f"tau_max の長さ {len(spec.tau_max)} が関節数 {len(entries)} と違う")

    for k, (name, joint) in enumerate(entries):
        attrs = {"name": name, "joint": joint, "gear": "1"}
        if spec.tau_max:
            t = float(spec.tau_max[k])
            attrs["ctrlrange"] = f"{-t} {t}"
        ET.SubElement(act, "motor", attrs)


def _zero_keyframe_ctrl(root: ET.Element) -> None:
    """! @brief keyframe の ctrl をゼロにする。
    @param root ルート要素 [-]
    @return なし
    @note position から motor へ変えると ctrl の単位が [rad] から [N*m] になる。
          元の値を残すと、home キーフレームを読んだ瞬間に最大トルクが入る。
    """
    kf = root.find("keyframe")
    if kf is None:
        return
    for key in kf.findall("key"):
        ctrl = key.get("ctrl")
        if ctrl is not None:
            key.set("ctrl", " ".join("0" for _ in ctrl.split()))


def _add_floor(root: ET.Element, spec: AdaptSpec) -> None:
    """! @brief 床・光源・床材質を足す（scene.xml を取り込む代わり）。
    @param root ルート要素 [-]
    @param spec AdaptSpec [-]
    @return なし
    @note geom 名は必ず "floor" にする。`MujocoEnv` が接触判定でこの名前を引く。
    """
    world = root.find("worldbody")
    if world is None:
        world = ET.SubElement(root, "worldbody")
    if any(g.get("name") == "floor" for g in world.findall("geom")):
        return

    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    if not any(m.get("name") == "loco_grid" for m in asset.findall("material")):
        ET.SubElement(
            asset,
            "texture",
            {
                "type": "2d",
                "name": "loco_grid_tex",
                "builtin": "checker",
                "mark": "edge",
                "rgb1": "0.2 0.3 0.4",
                "rgb2": "0.1 0.2 0.3",
                "markrgb": "0.8 0.8 0.8",
                "width": "300",
                "height": "300",
            },
        )
        ET.SubElement(
            asset,
            "material",
            {
                "name": "loco_grid",
                "texture": "loco_grid_tex",
                "texuniform": "true",
                "texrepeat": "5 5",
                "reflectance": "0.2",
            },
        )

    ET.SubElement(
        world,
        "light",
        {"pos": "0 0 1.5", "dir": "0 0 -1", "directional": "true"},
    )
    ET.SubElement(
        world,
        "geom",
        {
            "name": "floor",
            "type": "plane",
            "size": "0 0 0.05",
            "material": "loco_grid",
            "contype": "1",
            "conaffinity": "1",
            "friction": " ".join(f"{v}" for v in spec.floor_friction),
        },
    )


def adapt_mjcf(src: str | Path, dst: str | Path, spec: AdaptSpec) -> Path:
    """! @brief 外部 MJCF を LocoCtrl 互換の派生 MJCF に変換して書き出す。
    @param src 原典 MJCF のパス [-]
    @param dst 出力先のパス [-]。原典と同じディレクトリが基本。別ディレクトリのときは
               compiler の meshdir / texturedir を dst からの相対パスに書き換える
    @param spec AdaptSpec [-]
    @return 書き出したパス
    @note 変換は (0) 指定関節の削除、(1) 足先 geom の命名と site 追加、(2) motor への差し替え、
          (3) 足先以外の床衝突の無効化、(4) IMU site と sensor の追加、
          (5) 床の追加、(6) keyframe の ctrl ゼロ化、
          (7) 関節トルクセンサ・足裏 F/T センサの追加（spec で有効化したとき）、
          (8) 手の衝突 geom・手先 site・手首 F/T センサの追加（hand_body_fmt を与えたとき）。
          原典は書き換えない。
    """
    src = Path(src)
    dst = Path(dst)
    tree = ET.parse(src)
    root = tree.getroot()
    if src.resolve().parent != dst.resolve().parent:
        # 別ディレクトリへ書くときは compiler の meshdir / texturedir を、dst から原典の
        # ディレクトリへの相対パスに書き換える（原典の相対参照を保つ）。
        # 別パッケージ（armtasks/boxpush など）が自前の assets/ に派生 MJCF を持つための口。
        rel = os.path.relpath(src.resolve().parent, dst.resolve().parent)
        comp = root.find("compiler")
        if comp is None:
            comp = ET.SubElement(root, "compiler")
        for key in ("meshdir", "texturedir"):
            cur = comp.get(key)
            if cur is not None and not Path(cur).is_absolute():
                comp.set(key, (Path(rel) / cur).as_posix())
            elif cur is None and key == "meshdir":
                comp.set(key, Path(rel).as_posix())

    _lock_joints(root, spec)
    _name_feet(root, spec)
    _foot_only_collision(root, spec)
    _add_imu(root, spec)
    _to_motor_actuators(root, spec)
    _add_torque_sensors(root, spec)
    _add_hands(root, spec)
    if spec.zero_keyframe_ctrl:
        _zero_keyframe_ctrl(root)
    if spec.add_floor:
        _add_floor(root, spec)

    ET.indent(tree, space="  ")
    head = (
        f"<!-- 自動生成: lococtrl.sim.mjcf_adapt が {src.name} から作成した派生 MJCF。\n"
        f"     直接編集しないこと。原典を更新したら再生成する。\n"
        f"     {spec.note} -->\n"
    )
    dst.write_text(head + ET.tostring(root, encoding="unicode") + "\n", encoding="utf-8")
    return dst
