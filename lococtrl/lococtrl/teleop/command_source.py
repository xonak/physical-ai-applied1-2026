"""! @file command_source.py
@brief テレオペの操作入力（速度指令 vx, vy, wz）の入力源と平滑化。

入力源はすべて同じ口 `poll() -> VelocityCommand` を持つ。制御ループは毎周期 poll し、
`CommandFilter` で加速度を制限してから `OnlineBipedPlanner.set_command()` に渡す。

| 入力源 | 中身 | 依存 |
|---|---|---|
| ScriptedSource | 時刻表 [(t, vx, vy, wz), ...]（試験・再現実験） | なし |
| KeyboardSource | MuJoCo ビューアのキー入力（WASD + QE、Space で停止）| mujoco.viewer |
| GamepadSource  | ゲームパッド（左スティック 並進、右スティック X 旋回） | pygame |
| UdpSource      | UDP の JSON `{"vx":..,"vy":..,"wz":..}`（マスターデバイス・他プロセスから） | なし |

正の向き: vx 前、vy 左、wz 左旋回（パターン座標系 = ロボットの進行方向）。
"""

from __future__ import annotations

import json
import socket
import time
from dataclasses import dataclass

import numpy as np

__all__ = [
    "VelocityCommand",
    "CommandLimits",
    "CommandFilter",
    "ScriptedSource",
    "KeyboardSource",
    "GamepadSource",
    "UdpSource",
    "make_source",
    "load_limits",
    "load_teleop_yaml",
]


@dataclass
class VelocityCommand:
    """! @brief 速度指令。
    @param vx    前進 [m/s]
    @param vy    左 [m/s]
    @param wz    左旋回 [rad/s]
    @param stop  即停止（指令 0、足を揃えて立つ）[-]
    @param quit  終了要求 [-]
    @param push  外乱注入の要求 {W} 力積 shape (3,) [N*s]（None で無し。試験用）
    """

    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0
    stop: bool = False
    quit: bool = False
    push: np.ndarray | None = None

    def as_array(self) -> np.ndarray:
        return np.array([0.0, 0.0, 0.0]) if self.stop else np.array([self.vx, self.vy, self.wz], dtype=np.float64)


@dataclass
class CommandLimits:
    """! @brief 指令の上限と加速度制限（configs/teleop.yaml の limits 節）。
    @param vx_max / vx_min 前進・後退の上限 [m/s]
    @param vy_max          横の上限 [m/s]
    @param wz_max          旋回の上限 [rad/s]
    @param a_max           並進の加速度上限 [m/s^2]
    @param alpha_max       旋回の加速度上限 [rad/s^2]
    @param deadzone        スティックの不感帯 [-]
    """

    vx_max: float = 0.4
    vx_min: float = -0.15
    vy_max: float = 0.15
    wz_max: float = 0.6
    a_max: float = 0.5
    alpha_max: float = 1.5
    deadzone: float = 0.12


class CommandFilter:
    """! @brief 速度指令の飽和と加速度制限（指令の飛びで DCM 参照が跳ねないように）。"""

    def __init__(self, limits: CommandLimits, dt: float) -> None:
        self.lim = limits
        self.dt = float(dt)
        self.v = np.zeros(3)

    def reset(self) -> None:
        self.v = np.zeros(3)

    def __call__(self, cmd: VelocityCommand) -> np.ndarray:
        """! @brief 1 周期ぶん更新する。
        @param cmd 生の指令 [-]
        @return 平滑化した [vx, vy, wz]
        """
        L = self.lim
        target = cmd.as_array()
        target[0] = float(np.clip(target[0], L.vx_min, L.vx_max))
        target[1] = float(np.clip(target[1], -L.vy_max, L.vy_max))
        target[2] = float(np.clip(target[2], -L.wz_max, L.wz_max))
        dv = target - self.v
        n_xy = float(np.linalg.norm(dv[0:2]))
        lim_xy = L.a_max * self.dt
        if n_xy > lim_xy:
            dv[0:2] *= lim_xy / n_xy
        dv[2] = float(np.clip(dv[2], -L.alpha_max * self.dt, L.alpha_max * self.dt))
        self.v = self.v + dv
        return self.v.copy()


# ----------------------------------------------------------------------
# 入力源
# ----------------------------------------------------------------------
class ScriptedSource:
    """! @brief 時刻表による指令（試験・再現用）。
    @param schedule [(t, vx, vy, wz), ...]。t [s] 以降その指令。省略した wz は 0
    @param pushes   [(t, jx, jy, jz), ...] 力積の注入時刻（試験用）
    """

    def __init__(self, schedule, pushes=None) -> None:
        self.sched = sorted((float(r[0]), float(r[1]), float(r[2]), float(r[3]) if len(r) > 3 else 0.0) for r in schedule)
        self.pushes = sorted((float(r[0]), np.array(r[1:4], dtype=np.float64)) for r in (pushes or []))
        self.t = 0.0
        self._next_push = 0

    def poll(self, t: float | None = None) -> VelocityCommand:
        if t is not None:
            self.t = float(t)
        cur = (0.0, 0.0, 0.0)
        for ts, vx, vy, wz in self.sched:
            if self.t + 1.0e-9 >= ts:
                cur = (vx, vy, wz)
        push = None
        if self._next_push < len(self.pushes) and self.t + 1.0e-9 >= self.pushes[self._next_push][0]:
            push = self.pushes[self._next_push][1]
            self._next_push += 1
        return VelocityCommand(*cur, push=push)

    def close(self) -> None:
        pass


class KeyboardSource:
    """! @brief MuJoCo ビューアのキー入力。
    W/S 前後、A/D 左右、Q/E 旋回（押している間だけ）、Space 停止、P で前押し（試験）、Esc 終了。
    `key_callback(keycode)` を `mujoco.viewer.launch_passive(..., key_callback=...)` に渡す。
    GLFW はキーの押し離しを別々に通知しないため、押下で 1 段（step）だけ増減する方式にする。
    """

    def __init__(self, limits: CommandLimits, step_v: float = 0.05, step_w: float = 0.1,
                 push_impulse: float = 10.0) -> None:
        self.lim = limits
        self.step_v = float(step_v)
        self.step_w = float(step_w)
        self.push_impulse = float(push_impulse)
        self.cmd = VelocityCommand()
        self._push_pending = None

    def key_callback(self, keycode: int) -> None:
        k = chr(keycode).upper() if 0 <= keycode < 256 else ""
        c = self.cmd
        if k == "W":
            c.vx = min(c.vx + self.step_v, self.lim.vx_max)
        elif k == "S":
            c.vx = max(c.vx - self.step_v, self.lim.vx_min)
        elif k == "A":
            c.vy = min(c.vy + self.step_v, self.lim.vy_max)
        elif k == "D":
            c.vy = max(c.vy - self.step_v, -self.lim.vy_max)
        elif k == "Q":
            c.wz = min(c.wz + self.step_w, self.lim.wz_max)
        elif k == "E":
            c.wz = max(c.wz - self.step_w, -self.lim.wz_max)
        elif k == " ":
            c.vx = c.vy = c.wz = 0.0
        elif k == "P":
            self._push_pending = np.array([self.push_impulse, 0.0, 0.0])
        elif keycode == 256:  # Esc
            c.quit = True

    def poll(self, t: float | None = None) -> VelocityCommand:  # noqa: ARG002
        out = VelocityCommand(self.cmd.vx, self.cmd.vy, self.cmd.wz, quit=self.cmd.quit, push=self._push_pending)
        self._push_pending = None
        return out

    def close(self) -> None:
        pass


@dataclass
class GamepadMap:
    """! @brief ゲームパッドの軸・ボタン割当（configs/teleop.yaml の gamepad 節）。
    @param axis_vx / axis_vy / axis_wz 軸番号 [-]（Xbox: 左 X=0, 左 Y=1, 右 X=3）
    @param invert_vx / invert_vy / invert_wz 符号反転 [-]（スティック上が負の機種が多い）
    @param button_stop / button_quit / button_push ボタン番号 [-]
    """

    axis_vx: int = 1
    axis_vy: int = 0
    axis_wz: int = 3
    invert_vx: bool = True
    invert_vy: bool = True
    invert_wz: bool = True
    button_stop: int = 0
    button_quit: int = 6
    button_push: int = 1
    push_impulse: float = 10.0


class GamepadSource:
    """! @brief ゲームパッド（pygame.joystick）。スティックを上限速度に線形写像、不感帯あり。"""

    def __init__(self, limits: CommandLimits, mapping: GamepadMap | None = None, index: int = 0) -> None:
        import pygame  # 依存は使うときだけ

        self.pg = pygame
        self.lim = limits
        self.map = GamepadMap() if mapping is None else mapping
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() <= index:
            raise RuntimeError(f"ゲームパッドが見つからない（接続数 {pygame.joystick.get_count()}）")
        self.js = pygame.joystick.Joystick(index)
        self.js.init()
        self._push_pending = None
        self._prev_push = False
        print(f"[teleop] gamepad: {self.js.get_name()} axes={self.js.get_numaxes()} buttons={self.js.get_numbuttons()}")

    def _axis(self, i: int, invert: bool) -> float:
        if i >= self.js.get_numaxes():
            return 0.0
        a = float(self.js.get_axis(i))
        a = -a if invert else a
        dz = self.lim.deadzone
        if abs(a) < dz:
            return 0.0
        return float(np.sign(a) * (abs(a) - dz) / (1.0 - dz))

    def poll(self, t: float | None = None) -> VelocityCommand:  # noqa: ARG002
        self.pg.event.pump()
        m = self.map
        ax = self._axis(m.axis_vx, m.invert_vx)
        vx = ax * (self.lim.vx_max if ax >= 0 else -self.lim.vx_min)
        vy = self._axis(m.axis_vy, m.invert_vy) * self.lim.vy_max
        wz = self._axis(m.axis_wz, m.invert_wz) * self.lim.wz_max
        nb = self.js.get_numbuttons()
        stop = bool(m.button_stop < nb and self.js.get_button(m.button_stop))
        quit_ = bool(m.button_quit < nb and self.js.get_button(m.button_quit))
        push_now = bool(m.button_push < nb and self.js.get_button(m.button_push))
        push = np.array([m.push_impulse, 0.0, 0.0]) if (push_now and not self._prev_push) else None
        self._prev_push = push_now
        return VelocityCommand(vx, vy, wz, stop=stop, quit=quit_, push=push)

    def close(self) -> None:
        try:
            self.js.quit()
            self.pg.joystick.quit()
        except Exception:  # noqa: BLE001
            pass


class UdpSource:
    """! @brief UDP の JSON 1 行を受ける（マスターデバイス・LabVIEW・ROS 2 ブリッジ・別 PC から）。

    受信形式: `{"vx": 0.2, "vy": 0.0, "wz": 0.1}`（任意に `"stop": true`, `"quit": true`, `"push": [jx, jy, jz]`）。
    最後に受けた指令を保持し、`timeout` [s] 以上届かなければ停止指令に落とす（フェイルセーフ）。
    """

    def __init__(self, port: int = 5005, host: str = "0.0.0.0", timeout: float = 0.5) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, int(port)))
        self.sock.setblocking(False)
        self.timeout = float(timeout)
        self.last = VelocityCommand()
        self.t_last = -1.0
        self.n_rx = 0
        print(f"[teleop] udp: listening {host}:{port}")

    @staticmethod
    def parse(msg: bytes | str) -> VelocityCommand:
        d = json.loads(msg if isinstance(msg, str) else msg.decode("utf-8"))
        push = d.get("push")
        return VelocityCommand(
            float(d.get("vx", 0.0)), float(d.get("vy", 0.0)), float(d.get("wz", 0.0)),
            stop=bool(d.get("stop", False)), quit=bool(d.get("quit", False)),
            push=None if push is None else np.asarray(push, dtype=np.float64).reshape(3),
        )

    def poll(self, t: float | None = None) -> VelocityCommand:
        now = time.monotonic()
        push = None
        while True:
            try:
                data, _ = self.sock.recvfrom(4096)
            except BlockingIOError:
                break
            except OSError:
                break
            try:
                c = self.parse(data)
            except (ValueError, UnicodeDecodeError):
                continue
            if c.push is not None:
                push = c.push
            self.last = c
            self.t_last = now
            self.n_rx += 1
        if self.t_last < 0.0 or now - self.t_last > self.timeout:
            return VelocityCommand(stop=True, quit=self.last.quit)
        return VelocityCommand(self.last.vx, self.last.vy, self.last.wz, stop=self.last.stop, quit=self.last.quit, push=push)

    def close(self) -> None:
        self.sock.close()


def load_teleop_yaml(path: str | None = None) -> dict:
    """! @brief configs/teleop.yaml（デバイス設定）を読む。無ければ空。
    @param path 別のファイル名（None で teleop.yaml）[-]
    @return dict
    """
    from ..common.config import load_yaml

    try:
        return load_yaml("teleop.yaml" if path is None else path) or {}
    except FileNotFoundError:
        return {}


def load_limits(path: str | None = None) -> CommandLimits:
    """! @brief configs/teleop.yaml の limits 節から CommandLimits を作る。
    @param path 別のファイル名 [-]
    @return CommandLimits
    """
    d = load_teleop_yaml(path).get("limits") or {}
    return CommandLimits(**{k: v for k, v in d.items() if k in CommandLimits.__dataclass_fields__})


def make_source(kind: str, limits: CommandLimits, cfg: dict | None = None):
    """! @brief 名前から入力源を作る。
    @param kind   "keyboard" | "gamepad" | "udp" | "scripted"
    @param limits CommandLimits [-]
    @param cfg    configs/teleop.yaml の該当節（gamepad / udp / scripted）[-]
    @return 入力源
    """
    cfg = cfg or {}
    if kind == "keyboard":
        return KeyboardSource(limits, **{k: v for k, v in cfg.items() if k in ("step_v", "step_w", "push_impulse")})
    if kind == "gamepad":
        m = GamepadMap(**{k: v for k, v in cfg.items() if k in GamepadMap.__dataclass_fields__})
        return GamepadSource(limits, m, index=int(cfg.get("index", 0)))
    if kind == "udp":
        return UdpSource(port=int(cfg.get("port", 5005)), host=str(cfg.get("host", "0.0.0.0")),
                         timeout=float(cfg.get("timeout", 0.5)))
    if kind == "scripted":
        return ScriptedSource(cfg.get("schedule", []), cfg.get("pushes"))
    raise ValueError(f"未知の入力源: {kind}（keyboard | gamepad | udp | scripted）")
