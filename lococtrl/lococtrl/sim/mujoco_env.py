"""! @file mujoco_env.py
@brief MuJoCo シミュレータのラッパ。
制御層がシミュレータの内部構造に依存しないように、RobotState（入力）/ ControlOutput（出力）で入出力する。

**mjModel / mjData を直接触ってよいのは sim/ 以下と model/backends/ 以下だけ。**
制御層へは RobotState / ControlOutput のみを受け渡す。

freeze_base=True では MJCF から free joint を取り除いた固定ベースモデルを生成する
（空中固定での関節 PD 確認に用いる）。XML を ElementTree で書き換えて
同一ディレクトリに一時ファイルとして書き出すため、相対参照のアセットも解決できる。
"""

from __future__ import annotations

import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..common.config import RobotConfig
from ..common.types import ControlOutput, RobotState
from ..model_dynamics.index_map import build_index_map

__all__ = ["MujocoEnv", "make_fixed_base_xml", "BoxObject", "HAND_COLLISION_BIT"]

## 手の衝突ビット（箱側の conaffinity に含める。手側は armtasks/boxpush/boxpush/model.py の HAND_COLLISION_BIT と一致させる）
HAND_COLLISION_BIT = 4


@dataclass
class BoxObject:
    """! @brief シーンに置く箱（自由物体）。
    @param half_size 半長 (x, y, z) [m]
    @param mass      質量 [kg]
    @param friction  摩擦 (slide, spin, roll) [-]。箱の geom に priority 2 を付けるので、床（0.8）や手（1.0）
                     との接触では **箱側の値がそのまま使われる**（MuJoCo は priority の高い geom の摩擦を採る）。
                     箱-床の静止摩擦力は friction[0] * m * g になる
    @param pos       初期位置 {W} (x, y, z) [m]。z は None なら床に置く（= half_size z）
    @param yaw       初期ヨー [rad]
    @param name      body / geom 名 [-]
    @param rgba      色 [-]
    @note 衝突は contype = 1、conaffinity = 1 | HAND_COLLISION_BIT。床・足・手とだけ当たり、
          脚や胴体の衝突クラス（ビット 2）とは当たらない。
    """

    half_size: tuple[float, float, float] = (0.20, 0.30, 0.25)
    mass: float = 20.0
    friction: tuple[float, float, float] = (0.4, 0.005, 0.0001)
    pos: tuple[float, float, float | None] = (0.8, 0.0, None)
    yaw: float = 0.0
    name: str = "box"
    rgba: tuple[float, float, float, float] = (0.8, 0.55, 0.3, 1.0)


def make_fixed_base_xml(src: Path, base_body: str, out_dir: Path | None = None) -> Path:
    """! @brief ベースの free joint を除去した MJCF を書き出す。
    @param src       元の MJCF パス [-]
    @param base_body ベースボディ名 [-]
    @param out_dir   出力先（None ならシステムの一時ディレクトリ）[-]
    @return 生成した MJCF のパス
    @note freejoint 要素と <joint type="free"> の両記法に対応する。
    @note 出力先を assets/ にすると、後片付けに失敗した環境（ネットワーク共有・
          コンテナのマウント等）で `_fixedbase_*.xml` がリポジトリに溜まる。
          既定をシステムの一時ディレクトリにし、アセットの相対解決先は
          compiler/assetdir で元の場所に固定する。
    """
    tree = ET.parse(src)
    root = tree.getroot()
    target = None
    for body in root.iter("body"):
        if body.get("name") == base_body:
            target = body
            break
    if target is None:
        raise KeyError(f"body '{base_body}' が MJCF に存在しない")

    removed = 0
    for child in list(target):
        if child.tag == "freejoint" or (child.tag == "joint" and child.get("type") == "free"):
            target.remove(child)
            removed += 1
    if removed == 0:
        raise ValueError(f"body '{base_body}' に free joint が無い（既に固定ベース）")

    # キーフレームはベース 7 自由度ぶんの qpos（6 ぶんの qvel）を含むため辻褄を合わせる。
    # 前提: ベースの free joint がモデル先頭の関節であること（脚型の MJCF では成立）。
    # 前提が崩れた場合は MuJoCo が読込時にサイズ不一致で例外を出すため、黙って壊れない。
    for key in root.iter("key"):
        for attr, drop in (("qpos", 7), ("qvel", 6), ("qacc", 6)):
            val = key.get(attr)
            if val is None:
                continue
            vals = val.split()
            key.set(attr, " ".join(vals[drop:])) if len(vals) > drop else key.attrib.pop(attr)

    comp = root.find("compiler")
    if comp is None:
        comp = ET.Element("compiler")
        root.insert(0, comp)
    # 相対指定のアセットパスは元 MJCF の場所を基準に絶対化する。
    # **assetdir だけを見て済ませてはいけない。** Menagerie のモデルは
    # meshdir="assets" のように MJCF 自身からの相対で書いてあり、
    # 一時ディレクトリへ書き出した瞬間に解決できなくなる。
    for key in ("assetdir", "meshdir", "texturedir"):
        v = comp.get(key)
        if v is not None and not Path(v).is_absolute():
            comp.set(key, str((src.parent / v).resolve()))
    if "assetdir" not in comp.attrib and "meshdir" not in comp.attrib:
        comp.set("assetdir", str(src.parent))

    d_out = Path(tempfile.gettempdir()) if out_dir is None else Path(out_dir)
    d_out.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="_fixedbase_", suffix=".xml", dir=str(d_out))
    import os

    os.close(fd)
    out = Path(name)
    tree.write(out, encoding="utf-8", xml_declaration=False)
    return out


class MujocoEnv:
    """! @brief MJCF の読込・reset・step・observe を担うシミュレータ。"""

    def __init__(
        self,
        cfg: RobotConfig,
        dt_sim: float = 0.001,
        freeze_base: bool = False,
        base_height: float | None = None,
        terrain=None,
        solver: str | None = None,
        solref: tuple[float, float] | None = None,
        objects: list[BoxObject] | None = None,
    ) -> None:
        """! @brief シミュレータを構築する。
        @param cfg         RobotConfig [-]
        @param dt_sim      物理積分刻み [s]
        @param freeze_base True でベースを空中固定（free joint を除去）
        @param base_height 固定ベース時の胴体高さ [m]（None なら MJCF の pos を使う）
        @param terrain     `bench.terrain.TerrainSpec`（None / 平地なら元の MJCF のまま）[-]
        @param solver      "Newton" | "CG"（None なら MJCF の設定のまま）[-]
        @param solref      接触の solref (time_const [s], damping_ratio [-])（None で既定）
        @param objects     シーンに足す箱 `BoxObject` の列（None / 空なら MJCF のまま）[-]
        @return なし
        @note objects を与えたときは MjSpec で箱を足してからコンパイルする。重心 c_W / dc_W と
              total_mass はロボットのサブツリーから取る（箱を含めない）。与えないときは従来どおり
              world のサブツリーを使い、既存の数値を変えない。
        @note terrain / solver / solref は sim-to-sim 試験のための入口である。
              **いずれも None のときは MJCF を一切書き換えない。**
        """
        import mujoco

        self._mj = mujoco
        self.cfg = cfg
        self.freeze_base = bool(freeze_base)
        src = cfg.mjcf_abspath()
        if not src.is_file():
            raise FileNotFoundError(f"MJCF が見つからない: {src}")

        self._tmp: Path | None = None
        self.terrain = terrain
        if freeze_base:
            self._tmp = make_fixed_base_xml(src, cfg.base_body)
            if base_height is not None:
                self._retarget_base_height(self._tmp, cfg.base_body, base_height)
            path = self._tmp
        elif terrain is not None and not getattr(terrain, "is_flat", True):
            from ..bench.terrain import make_terrain_xml

            self._tmp = make_terrain_xml(src, terrain)
            path = self._tmp if self._tmp is not None else src
        else:
            path = src

        self.objects = list(objects or [])
        if self.objects:
            spec = mujoco.MjSpec.from_file(str(path))
            for ob in self.objects:
                hz = float(ob.half_size[2])
                z = hz + 1.0e-4 if ob.pos[2] is None else float(ob.pos[2])
                c, s = np.cos(0.5 * ob.yaw), np.sin(0.5 * ob.yaw)
                b = spec.worldbody.add_body(name=ob.name, pos=[float(ob.pos[0]), float(ob.pos[1]), z],
                                            quat=[c, 0.0, 0.0, s])
                b.add_freejoint(name=f"{ob.name}_free")
                b.add_geom(name=f"{ob.name}_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
                           size=[float(v) for v in ob.half_size], mass=float(ob.mass),
                           friction=[float(v) for v in ob.friction], rgba=[float(v) for v in ob.rgba],
                           contype=1, conaffinity=1 | HAND_COLLISION_BIT, condim=3, priority=2)
            self.model = spec.compile()
        else:
            self.model = mujoco.MjModel.from_xml_path(str(path))
        self.model.opt.timestep = float(dt_sim)
        if terrain is not None and not getattr(terrain, "is_flat", True):
            from ..bench.terrain import write_heightfield

            write_heightfield(self.model, terrain)
        if solver is not None:
            self.model.opt.solver = {
                "newton": mujoco.mjtSolver.mjSOL_NEWTON,
                "cg": mujoco.mjtSolver.mjSOL_CG,
                "pgs": mujoco.mjtSolver.mjSOL_PGS,
            }[solver.lower()]
        if solref is not None:
            self.model.geom_solref[:, 0] = float(solref[0])
            self.model.geom_solref[:, 1] = float(solref[1])
        self.data = mujoco.MjData(self.model)
        self.idx = build_index_map(self.model, cfg)

        free = [
            j for j in range(self.model.njnt) if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
            and int(self.model.jnt_bodyid[j]) == int(self.idx.base_body)
        ]
        self._free_qadr = int(self.model.jnt_qposadr[free[0]]) if free else -1
        self._free_vadr = int(self.model.jnt_dofadr[free[0]]) if free else -1
        self._base_body = self.idx.base_body
        # 重心を取るサブツリー。箱が無ければ world（0）のまま（既存の数値を 1 ビットも変えない）
        self._com_body = int(self.model.body_rootid[self._base_body]) if self.objects else 0
        if self.objects:
            self.total_mass = float(self.model.body_subtreemass[self._com_body])
        else:
            self.total_mass = float(np.sum(self.model.body_mass[1:]))
        self._box_free: list[tuple[int, int]] = []  # (qpos adr, qvel adr)
        for ob in self.objects:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{ob.name}_free")
            self._box_free.append((int(self.model.jnt_qposadr[jid]), int(self.model.jnt_dofadr[jid])))
        self._box_init = [np.array(self.model.qpos0[a : a + 7]) for a, _ in self._box_free]
        self._floor_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self._build_sensor_index()

    def __del__(self) -> None:
        """! @brief 一時 MJCF を後片付けする。
        @return なし
        """
        try:
            if getattr(self, "_tmp", None) is not None and self._tmp.is_file():
                self._tmp.unlink()
        except Exception:  # noqa: BLE001 - デストラクタで例外を投げない
            pass

    @staticmethod
    def _retarget_base_height(path: Path, base_body: str, height: float) -> None:
        """! @brief 固定ベース MJCF の胴体高さを書き換える。
        @param path      MJCF パス [-]
        @param base_body ベースボディ名 [-]
        @param height    胴体高さ {W} [m]
        @return なし
        """
        tree = ET.parse(path)
        for body in tree.getroot().iter("body"):
            if body.get("name") == base_body:
                pos = [float(x) for x in (body.get("pos") or "0 0 0").split()]
                pos[2] = float(height)
                body.set("pos", " ".join(f"{v:.6f}" for v in pos))
                break
        tree.write(path, encoding="utf-8", xml_declaration=False)

    # ---- 次元 -------------------------------------------------------------
    @property
    def n_joint(self) -> int:
        """! @brief 駆動関節数 n [-]"""
        return self.idx.n_joint

    @property
    def n_leg(self) -> int:
        """! @brief 脚数 nf [-]"""
        return self.idx.n_leg

    @property
    def dt(self) -> float:
        """! @brief 物理積分刻み [s]"""
        return float(self.model.opt.timestep)

    @property
    def time(self) -> float:
        """! @brief シミュレーション時刻 [s]"""
        return float(self.data.time)

    # ---- 状態設定 ---------------------------------------------------------
    def reset(
        self,
        q_joint: np.ndarray | None = None,
        p_WB: np.ndarray | None = None,
        quat_WB: np.ndarray | None = None,
        keyframe: str | None = "home",
    ) -> RobotState:
        """! @brief 状態を初期化する。
        @param q_joint 関節角 shape (n,) [rad]（None なら keyframe / 公称値）
        @param p_WB    ベース位置 {W} shape (3,) [m]（固定ベース時は無視）
        @param quat_WB ベース姿勢 shape (4,) [-]（固定ベース時は無視）
        @param keyframe 使用するキーフレーム名（無ければ無視）[-]
        @return 初期化後の RobotState
        """
        self._mj.mj_resetData(self.model, self.data)
        if keyframe is not None:
            kid = self._mj.mj_name2id(self.model, self._mj.mjtObj.mjOBJ_KEY, keyframe)
            if kid >= 0:
                self._mj.mj_resetDataKeyframe(self.model, self.data, kid)

        if q_joint is not None:
            self.data.qpos[self.idx.joint_qpos] = np.asarray(q_joint, dtype=np.float64).reshape(
                self.n_joint
            )
        if self._free_qadr >= 0:
            a = self._free_qadr
            if p_WB is not None:
                self.data.qpos[a : a + 3] = np.asarray(p_WB, dtype=np.float64).reshape(3)
            if quat_WB is not None:
                self.data.qpos[a + 3 : a + 7] = np.asarray(quat_WB, dtype=np.float64).reshape(4)
        for (a, _), q0 in zip(self._box_free, self._box_init):
            self.data.qpos[a : a + 7] = q0
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        self._mj.mj_forward(self.model, self.data)
        return self.observe()

    # ---- 実行 -------------------------------------------------------------
    def apply(self, u: ControlOutput) -> np.ndarray:
        """! @brief 制御出力を関節トルクへ変換して ctrl に書く。
        @param u ControlOutput [-]
        @return tau_eff 実効トルク shape (n,) [N*m]
        @note tau_eff = tau + kp*(q_des-q) + kd*(dq_des-dq) をここで計算する。
              関節 PD をシミュレータ側に置くのは、実機の関節ボード相当だからである。
        """
        q = np.array(self.data.qpos[self.idx.joint_qpos], dtype=np.float64)
        dq = np.array(self.data.qvel[self.idx.joint_qvel], dtype=np.float64)
        tau = u.effective_torque(q, dq)
        tau = np.clip(tau, -self.cfg.tau_max, self.cfg.tau_max)
        self.data.ctrl[self.idx.joint_act] = tau
        return tau

    def step(self, n: int = 1) -> None:
        """! @brief 物理を n ステップ進める。
        @param n ステップ数 [-]
        @return なし
        """
        for _ in range(int(n)):
            self._mj.mj_step(self.model, self.data)

    def apply_impulse_base(self, impulse_W: np.ndarray) -> None:
        """! @brief 胴体に並進の力積を与える（外乱注入）。
        @param impulse_W 力積 {W} shape (3,) [N*s]
        @return なし
        @note 速度を直接書き換える近似（Delta v = J / m_total）。
              外乱スイープでも同じ入口を使う。
        """
        if self._free_vadr < 0:
            raise RuntimeError("固定ベースでは力積を与えられない")
        b = self._free_vadr
        self.data.qvel[b : b + 3] += (
            np.asarray(impulse_W, dtype=np.float64).reshape(3) / self.total_mass
        )

    def apply_external_wrench(
        self, force_W: np.ndarray, torque_W: np.ndarray | None = None, body: int | None = None
    ) -> None:
        """! @brief 外力・外トルクを body に加える（次に clear するまで持続）。
        @param force_W  力 {W} shape (3,) [N]
        @param torque_W トルク {W} shape (3,) [N*m]（None で 0）
        @param body     対象ボディ id（None でベース）[-]
        @return なし
        @note **MuJoCo の `xfrc_applied` は明示的にゼロへ戻すまで掛かり続ける。**
              パルス外乱を作るときは終了時に必ず `clear_external_wrench()` を呼ぶこと。
        """
        b = self._base_body if body is None else int(body)
        self.data.xfrc_applied[b, 0:3] = np.asarray(force_W, dtype=np.float64).reshape(3)
        self.data.xfrc_applied[b, 3:6] = (
            np.zeros(3) if torque_W is None else np.asarray(torque_W, dtype=np.float64).reshape(3)
        )

    def clear_external_wrench(self) -> None:
        """! @brief すべての外力・外トルクをゼロに戻す。
        @return なし
        """
        self.data.xfrc_applied[:, :] = 0.0

    # ---- 観測 -------------------------------------------------------------
    def foot_forces(self) -> tuple[np.ndarray, np.ndarray]:
        """! @brief 足先の接触力（{W}）と接触フラグを返す。
        @return (f shape (nf,3) [N], contact shape (nf,) [bool])
        @note 同一足で複数接触点がある場合は合力を取る。
        """
        f = np.zeros((self.n_leg, 3), dtype=np.float64)
        con = np.zeros(self.n_leg, dtype=bool)
        buf = np.zeros(6, dtype=np.float64)
        geom2leg = {int(g): i for i, g in enumerate(self.idx.foot_geom)}
        for c in range(self.data.ncon):
            contact = self.data.contact[c]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            leg = geom2leg.get(g1, geom2leg.get(g2, -1))
            if leg < 0:
                continue
            self._mj.mj_contactForce(self.model, self.data, c, buf)
            R = np.array(contact.frame, dtype=np.float64).reshape(3, 3)
            # contact.frame の行は接触座標系の基底（第 1 行が法線）。力は接触系表現。
            f_w = R.T @ buf[0:3]
            sign = 1.0 if leg == geom2leg.get(g2, -1) else -1.0
            f[leg] += sign * f_w
            con[leg] = True
        return f, con

    def contact_cop(self, z_plane: float = 0.0) -> np.ndarray:
        """! @brief 床との接触点から圧力中心 (CoP) を求める。
        @param z_plane CoP を評価する水平面の高さ [m]
        @return cop shape (3,) [m]（接触が無ければ NaN）
        @note `bench.metrics.center_of_pressure()` は足先 1 点あたり 1 つの力から
              CoP を出すので、**足裏が面で接している二足では足裏内の分布を表せない**
              （足先 site は足裏中心に 1 つしかない）。本関数は MuJoCo の接触点
              そのものを使うため、足裏のどこに荷重が乗っているかが分かる。
              二足の評価（CoP が支持多角形内に留まる割合）はこちらで測る。
        """
        num = np.zeros(2, dtype=np.float64)
        fz_sum = 0.0
        buf = np.zeros(6, dtype=np.float64)
        foot_geoms = {int(g) for g in self.idx.foot_geom}
        for c in range(self.data.ncon):
            con = self.data.contact[c]
            g1, g2 = int(con.geom1), int(con.geom2)
            if g1 not in foot_geoms and g2 not in foot_geoms:
                continue
            self._mj.mj_contactForce(self.model, self.data, c, buf)
            R = np.array(con.frame, dtype=np.float64).reshape(3, 3)
            f_w = R.T @ buf[0:3]
            # geom1 が足側のとき力の符号が逆になる（foot_forces と同じ規約）
            if g1 in foot_geoms:
                f_w = -f_w
            fz = float(f_w[2])
            if fz <= 0.0:
                continue
            pos = np.array(con.pos, dtype=np.float64)
            dz = pos[2] - z_plane
            # 接線力が生むモーメントを含める（metrics.center_of_pressure と同じ式）
            num[0] += pos[0] * fz - dz * f_w[0]
            num[1] += pos[1] * fz - dz * f_w[1]
            fz_sum += fz
        if fz_sum <= 1.0e-9:
            return np.full(3, np.nan)
        return np.array([num[0] / fz_sum, num[1] / fz_sum, z_plane])

    def _sensor(self, name: str, dim: int) -> np.ndarray:
        """! @brief 名前引きでセンサ値を取り出す。
        @param name センサ名 [-]
        @param dim  次元 [-]
        @return 値 shape (dim,) [センサ依存]
        """
        sid = self._mj.mj_name2id(self.model, self._mj.mjtObj.mjOBJ_SENSOR, name)
        if sid < 0:
            raise KeyError(f"センサ '{name}' が MJCF に存在しない")
        adr = int(self.model.sensor_adr[sid])
        return np.array(self.data.sensordata[adr : adr + dim], dtype=np.float64)

    def imu_accelerometer(self) -> np.ndarray:
        """! @brief IMU の加速度計出力（比力、{B} 表現）。
        @return a_meas shape (3,) [m/s^2]
        @note 加速度計は重力を含む比力 a_B = R^T (a_W - g) を出す。
              EKF の予測式 v_dot = R a_meas + g と符号を合わせること。
        """
        return self._sensor("imu_acc", 3)

    def imu_gyro(self) -> np.ndarray:
        """! @brief IMU のジャイロ出力（{B} 表現）。
        @return w_meas shape (3,) [rad/s]
        """
        return self._sensor("imu_gyro", 3)

    # ---- 関節トルクセンサ・足裏 F/T センサ -----------------------------------
    def _build_sensor_index(self) -> None:
        """! @brief 関節トルクセンサと足裏 F/T センサの索引を名前引きで作る。
        @return なし
        @note センサが MJCF に無い機体では空のままにし、
              `observe()` は tau_meas / wrench_feet をゼロで返す。
        """
        mj = self._mj
        m = self.model
        self._jts_adr: list[int] = []
        self._jts_axis: list[np.ndarray] = []
        names = self.cfg.all_joint_names
        for n in names:
            sid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_SENSOR, f"jts_{n}")
            if sid < 0:
                self._jts_adr = []
                self._jts_axis = []
                break
            jid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, n)
            self._jts_adr.append(int(m.sensor_adr[sid]))
            self._jts_axis.append(np.array(m.jnt_axis[jid], dtype=np.float64))
        self._ft: list[tuple[int, int, int, float]] = []  # (force adr, torque adr, site id, 足質量)
        for leg in self.cfg.leg_names:
            sf = mj.mj_name2id(m, mj.mjtObj.mjOBJ_SENSOR, f"ft_{leg}_force")
            st = mj.mj_name2id(m, mj.mjtObj.mjOBJ_SENSOR, f"ft_{leg}_torque")
            site = mj.mj_name2id(m, mj.mjtObj.mjOBJ_SITE, f"ft_{leg}")
            if sf < 0 or st < 0 or site < 0:
                self._ft = []
                break
            # センサ site を含むボディ以下の質量（足首より先の質量。G1 では ankle_roll_link 1 個）
            body = int(m.site_bodyid[site])
            m_foot = float(m.body_subtreemass[body])
            self._ft.append((int(m.sensor_adr[sf]), int(m.sensor_adr[st]), int(site), m_foot))
        # 手首 F/T（箱押し用モデル）。手の名前は脚と同じ left / right（ft_<hand>_hand）
        self._hand_ft: list[tuple[int, int, int, float]] = []
        if self.cfg.n_arm > 0:
            for hand in self.cfg.leg_names[: self.cfg.n_arm]:
                sf = mj.mj_name2id(m, mj.mjtObj.mjOBJ_SENSOR, f"ft_{hand}_hand_force")
                st = mj.mj_name2id(m, mj.mjtObj.mjOBJ_SENSOR, f"ft_{hand}_hand_torque")
                site = mj.mj_name2id(m, mj.mjtObj.mjOBJ_SITE, f"ft_{hand}_hand")
                if sf < 0 or st < 0 or site < 0:
                    self._hand_ft = []
                    break
                body = int(m.site_bodyid[site])
                self._hand_ft.append((int(m.sensor_adr[sf]), int(m.sensor_adr[st]), int(site),
                                      float(m.body_subtreemass[body])))

    @property
    def has_joint_torque_sensors(self) -> bool:
        """! @brief 関節トルクセンサがあるか [-]"""
        return len(self._jts_adr) == self.n_joint

    @property
    def has_foot_ft_sensors(self) -> bool:
        """! @brief 足裏 F/T センサがあるか [-]"""
        return len(self._ft) == self.n_leg

    def joint_torque_meas(self) -> np.ndarray:
        """! @brief 関節トルクセンサの値（関節軸に射影）。
        @return tau_meas shape (n,) [N*m]（センサが無ければ 0）
        @note MuJoCo の `<torque>` センサは親子ボディ間の相互作用トルクを site 系で返す。
              site は回転を持たないので子ボディ系 = 関節軸の定義系であり、
              `jnt_axis` との内積が関節トルクになる。符号はアクチュエータ指令と同符号
              であることを立位で確認済み（ctrl 7.66 N*m に対し 7.63 N*m）。
              受動項（関節ダンピング・リミット）と外力の寄与を含むので、
              指令トルクと一致するのは準静的なときだけである。
        """
        if not self.has_joint_torque_sensors:
            return np.zeros(self.n_joint, dtype=np.float64)
        out = np.empty(self.n_joint, dtype=np.float64)
        for k, (adr, ax) in enumerate(zip(self._jts_adr, self._jts_axis)):
            out[k] = float(ax @ self.data.sensordata[adr : adr + 3])
        return out

    def foot_ft_wrench(self) -> np.ndarray:
        """! @brief 足裏 F/T センサから床反力レンチ（{W}、足先 site まわり）を推定する。
        @return wrench shape (nf,6) [N, N*m]（センサが無ければ 0）
        @note センサは足首（足先ボディの原点）にあり、親子ボディ間の相互作用レンチを
              子ボディ系で返す。立位の実測では z 力が **-(床反力 - 足の重量)** だったので、
              床反力は `f = -f_sens + m_foot g_up`、モーメントは `tau = -tau_sens` に
              足の重力モーメントを足したものとして復元する（足の慣性項は無視する。
              実機の足首 F/T センサで行う補正と同じ）。
              そのうえで足先 site（足裏中心）まわりへ `tau_site = tau + (p_sens - p_site) x f`
              で移す。
        """
        nf = self.n_leg
        if not self.has_foot_ft_sensors:
            return np.zeros((nf, 6), dtype=np.float64)
        out = np.zeros((nf, 6), dtype=np.float64)
        g_up = np.array([0.0, 0.0, 9.81])
        for i, (af, at, site, m_foot) in enumerate(self._ft):
            R = np.array(self.data.site_xmat[site], dtype=np.float64).reshape(3, 3)
            f_s = R @ np.array(self.data.sensordata[af : af + 3], dtype=np.float64)
            t_s = R @ np.array(self.data.sensordata[at : at + 3], dtype=np.float64)
            p_sens = np.array(self.data.site_xpos[site], dtype=np.float64)
            body = int(self.model.site_bodyid[site])
            # 足の重心（サブツリー）はセンサより下にある。重力の寄与を取り除く。
            p_cm = np.array(self.data.subtree_com[body], dtype=np.float64)
            f = -f_s + m_foot * g_up
            tau = -t_s + np.cross(p_cm - p_sens, m_foot * g_up)
            p_site = np.array(self.data.site_xpos[int(self.idx.foot_site[i])], dtype=np.float64)
            tau_site = tau + np.cross(p_sens - p_site, f)
            out[i, 0:3] = f
            out[i, 3:6] = tau_site
        return out

    @property
    def has_hand_ft_sensors(self) -> bool:
        """! @brief 手首 F/T センサがあるか [-]"""
        return self.cfg.n_arm > 0 and len(self._hand_ft) == self.cfg.n_arm

    def hand_ft_wrench(self) -> np.ndarray:
        """! @brief 手首 F/T から、環境が手に加える接触レンチ（{W}、手先 site まわり）を推定する。
        @return wrench shape (n_arm,6) [N, N*m]（センサが無ければ shape (0,6)）
        @note 足裏 F/T と同じ補正（`foot_ft_wrench` の @note）。センサは手首 yaw 関節にあり、
              手ボディ（手首 yaw リンク + ゴムの手）の重力を差し引く。慣性項は無視する。
        """
        if not self.has_hand_ft_sensors:
            return np.zeros((0, 6), dtype=np.float64)
        out = np.zeros((len(self._hand_ft), 6), dtype=np.float64)
        g_up = np.array([0.0, 0.0, 9.81])
        for a, (af, at, site, m_hand) in enumerate(self._hand_ft):
            R = np.array(self.data.site_xmat[site], dtype=np.float64).reshape(3, 3)
            f_s = R @ np.array(self.data.sensordata[af : af + 3], dtype=np.float64)
            t_s = R @ np.array(self.data.sensordata[at : at + 3], dtype=np.float64)
            p_sens = np.array(self.data.site_xpos[site], dtype=np.float64)
            body = int(self.model.site_bodyid[site])
            p_cm = np.array(self.data.subtree_com[body], dtype=np.float64)
            f = -f_s + m_hand * g_up
            tau = -t_s + np.cross(p_cm - p_sens, m_hand * g_up)
            p_site = np.array(self.data.site_xpos[int(self.idx.hand_site[a])], dtype=np.float64) \
                if self.idx.hand_site.size > a else p_sens
            out[a, 0:3] = f
            out[a, 3:6] = tau + np.cross(p_sens - p_site, f)
        return out

    def box_pose(self, i: int = 0) -> np.ndarray:
        """! @brief 箱 i の位置・姿勢・並進速度。
        @param i 箱の番号 [-]
        @return shape (10,) [m, -, m/s]（p 3, quat 4, v 3。箱が無ければ shape (0,)）
        """
        if i >= len(self._box_free):
            return np.zeros(0)
        a, b = self._box_free[i]
        return np.concatenate([self.data.qpos[a : a + 7], self.data.qvel[b : b + 3]]).astype(np.float64)

    def observe(self) -> RobotState:
        """! @brief mjData の真値から RobotState を構築する。
        @return RobotState（p_WF, contact, f_meas, c_W, dc_W を含む）
        @note これは真値観測である（Cheater）。
        """
        self._mj.mj_forward(self.model, self.data)
        self._mj.mj_subtreeVel(self.model, self.data)  # subtree_linvel / subtree_angmom の更新
        if self._free_qadr >= 0:
            a, b = self._free_qadr, self._free_vadr
            p_WB = np.array(self.data.qpos[a : a + 3], dtype=np.float64)
            quat = np.array(self.data.qpos[a + 3 : a + 7], dtype=np.float64)
            v_W = np.array(self.data.qvel[b : b + 3], dtype=np.float64)
            w_B = np.array(self.data.qvel[b + 3 : b + 6], dtype=np.float64)
            R_WB = np.array(self.data.xmat[self._base_body], dtype=np.float64).reshape(3, 3)
            v_B = R_WB.T @ v_W
        else:  # 固定ベース
            p_WB = np.array(self.data.xpos[self._base_body], dtype=np.float64)
            quat = np.array(self.data.xquat[self._base_body], dtype=np.float64)
            v_B = np.zeros(3)
            w_B = np.zeros(3)

        f, con = self.foot_forces()
        return RobotState(
            p_WB=p_WB,
            quat_WB=quat,
            v_B=v_B,
            w_B=w_B,
            q=np.array(self.data.qpos[self.idx.joint_qpos], dtype=np.float64),
            dq=np.array(self.data.qvel[self.idx.joint_qvel], dtype=np.float64),
            p_WF=np.array(self.data.site_xpos[self.idx.foot_site], dtype=np.float64),
            contact=con,
            f_meas=f,
            c_W=np.array(self.data.subtree_com[self._com_body], dtype=np.float64),
            dc_W=np.array(self.data.subtree_linvel[self._com_body], dtype=np.float64),
            t=float(self.data.time),
            tau_meas=self.joint_torque_meas(),
            wrench_feet=self.foot_ft_wrench(),
            wrench_hands=self.hand_ft_wrench(),
            box_pose=self.box_pose(0),
            p_hands=np.array(self.data.site_xpos[self.idx.hand_site], dtype=np.float64).reshape(-1, 3)
            if self.idx.hand_site.size else np.zeros((0, 3)),
        )

    def model_q(self) -> np.ndarray:
        """! @brief 動力学モデル用の一般化座標を組み立てる。
        @return q shape (7+n,) [m, -, rad]
        @note 固定ベース時もモデル側は浮遊ベースであるため、実際の胴体位置・姿勢を用いる。
        """
        s = self.observe()
        return np.concatenate([s.p_WB, s.quat_WB, s.q])

    def model_v(self) -> np.ndarray:
        """! @brief 動力学モデル用の一般化速度を組み立てる（[v_W; w_B; dq]）。
        @return v shape (6+n,) [m/s, rad/s]
        """
        s = self.observe()
        R_WB = np.array(self.data.xmat[self._base_body], dtype=np.float64).reshape(3, 3)
        return np.concatenate([R_WB @ s.v_B, s.w_B, s.dq])
