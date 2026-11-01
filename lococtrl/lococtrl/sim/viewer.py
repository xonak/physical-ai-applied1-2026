"""! @file viewer.py
@brief mujoco.viewer の薄いラッパ。UIの設定や接触力の矢印描画を提供する。 

- ヘッドレス環境（CI・サーバ）では launch に失敗するため、生成失敗時は
  「何もしないビューア」に自動フォールバックする。テストがビューアの有無に依存しない。
- 接触力ベクトルの矢印描画を add_arrow() として用意する。
"""

from __future__ import annotations

import numpy as np

__all__ = ["Viewer"]


class Viewer:
    """! @brief 対話ビューア。使用不可なら自動的に無効化される。"""

    def __init__(self, env, enable: bool = True, key_callback=None) -> None:
        """! @brief ビューアを起動する。
        @param env          MujocoEnv [-]
        @param enable       False なら起動しない
        @param key_callback キー入力のコールバック (keycode:int) -> None（テレオペのキーボード入力）[-]
        @return なし
        """
        self._env = env
        self._v = None
        self._n_user_geom = 0
        if not enable:
            return
        try:
            # mujoco.viewer は GLFW 初期化に失敗すると FatalError でプロセスごと落ちるので、先に自分で確かめる
            import warnings

            import glfw

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if not glfw.init():
                    raise RuntimeError("GLFW を初期化できない（DISPLAY が無い等）")
            import mujoco.viewer as mjv

            self._v = (mjv.launch_passive(env.model, env.data, key_callback=key_callback)
                       if key_callback is not None else mjv.launch_passive(env.model, env.data))
            self._v.user_scn.ngeom = 0
        except KeyboardInterrupt:
            raise
        except BaseException as e:  # noqa: BLE001 - 表示できないだけで実験は続行する（GLFW は SystemExit を投げる）
            print(f"[viewer] 起動できないためヘッドレスで続行する: {e!r}")
            self._v = None

    @property
    def enabled(self) -> bool:
        """! @brief ビューアが有効か [-]"""
        return self._v is not None

    def is_running(self) -> bool:
        """! @brief ウィンドウが開いているか [-]"""
        return bool(self._v.is_running()) if self._v is not None else True

    def clear_arrows(self) -> None:
        """! @brief 追加描画をクリアする。
        @return なし
        """
        if self._v is None:
            return
        self._v.user_scn.ngeom = 0
        self._n_user_geom = 0

    def add_arrow(
        self,
        origin: np.ndarray,
        vec: np.ndarray,
        scale: float = 0.002,
        rgba: tuple[float, float, float, float] = (1.0, 0.2, 0.2, 0.9),
        width: float = 0.006,
    ) -> None:
        """! @brief 矢印を 1 本追加する（接触力の可視化用）。
        @param origin 始点 {W} shape (3,) [m]
        @param vec    ベクトル shape (3,) [N など]
        @param scale  長さ換算係数 [m/N]
        @param rgba   色 [-]
        @param width  軸半径 [m]
        @return なし
        """
        if self._v is None:
            return
        import mujoco

        scn = self._v.user_scn
        if scn.ngeom >= scn.maxgeom:
            return
        p0 = np.asarray(origin, dtype=np.float64).reshape(3)
        p1 = p0 + np.asarray(vec, dtype=np.float64).reshape(3) * float(scale)
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(
            g,
            int(mujoco.mjtGeom.mjGEOM_ARROW),
            np.zeros(3),
            np.zeros(3),
            np.zeros(9),
            np.array(rgba, dtype=np.float32),
        )
        mujoco.mjv_connector(g, int(mujoco.mjtGeom.mjGEOM_ARROW), float(width), p0, p1)
        scn.ngeom += 1

    def sync(self) -> None:
        """! @brief 画面を更新する。
        @return なし
        """
        if self._v is not None:
            self._v.sync()

    def close(self) -> None:
        """! @brief ビューアを閉じる。
        @return なし
        """
        if self._v is not None:
            self._v.close()
            self._v = None
