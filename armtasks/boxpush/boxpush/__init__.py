"""箱押し環境と同梱の歩行制御基盤。"""
from __future__ import annotations

import sys
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PKG_ROOT.parents[1]
LOCOCTRL_ROOT = REPO_ROOT / "lococtrl"
if not (LOCOCTRL_ROOT / "lococtrl" / "__init__.py").is_file():
    raise ImportError("同梱の lococtrl が見つかりません。配布フォルダー全体を展開してください。")
sys.path.insert(0, str(LOCOCTRL_ROOT))
import lococtrl
if Path(lococtrl.__file__).resolve().parent != (LOCOCTRL_ROOT / "lococtrl").resolve():
    raise ImportError("別の lococtrl が読み込まれています。新しい仮想環境で起動してください。")

__all__ = ["PKG_ROOT", "REPO_ROOT", "LOCOCTRL_ROOT"]
