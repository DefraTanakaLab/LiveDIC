"""
settings_store.py  —  GUI の設定を次回起動時に復元する

ウィジェットを名前で登録しておき、終了時に JSON へ保存・起動時に読み戻すだけの薄い仕組み。
保存先は `~/.livedic/settings.json`（exe 版でも書ける場所。PyInstaller の onedir は
Program Files に置かれることがあるので実行ファイルの隣には置かない）。

方針:
  ・壊れた設定で起動できなくなるのが最悪なので、読み書きとも**個別に try で囲み**、
    失敗した項目は既定値のまま進む（例外を外へ出さない）
  ・ウィジェットの種類から値の取り方を決めるので、登録するのは名前と参照だけでよい
  ・コンボボックスは**表示文字列**で保存する（カメラ一覧のように順番が変わるものがあるため）
"""
from __future__ import annotations
import json
import pathlib
from typing import Any, Dict, Optional, Callable

from PyQt6.QtWidgets import (
    QWidget, QCheckBox, QRadioButton, QAbstractButton, QComboBox,
    QSpinBox, QDoubleSpinBox, QSlider, QLineEdit,
)

SETTINGS_PATH = pathlib.Path.home() / ".livedic" / "settings.json"


def _get(w: QWidget) -> Any:
    if isinstance(w, (QCheckBox, QRadioButton)) or (isinstance(w, QAbstractButton) and w.isCheckable()):
        return bool(w.isChecked())
    if isinstance(w, QComboBox):
        return w.currentText()
    if isinstance(w, (QSpinBox, QSlider)):
        return int(w.value())
    if isinstance(w, QDoubleSpinBox):
        return float(w.value())
    if isinstance(w, QLineEdit):
        return w.text()
    raise TypeError(type(w).__name__)


def _set(w: QWidget, v: Any) -> None:
    if isinstance(w, (QCheckBox, QRadioButton)) or (isinstance(w, QAbstractButton) and w.isCheckable()):
        w.setChecked(bool(v))
    elif isinstance(w, QComboBox):
        i = w.findText(str(v))
        if i >= 0:                      # 見つからないとき（カメラが繋がっていない等）は触らない
            w.setCurrentIndex(i)
    elif isinstance(w, (QSpinBox, QSlider)):
        w.setValue(int(v))
    elif isinstance(w, QDoubleSpinBox):
        w.setValue(float(v))
    elif isinstance(w, QLineEdit):
        w.setText(str(v))
    else:
        raise TypeError(type(w).__name__)


class SettingsStore:
    """ウィジェットと「任意の値」を名前で登録し、まとめて保存・復元する。"""

    def __init__(self, path: Optional[pathlib.Path] = None):
        # 既定値を引数に書くと定義時に束縛されてしまい、SETTINGS_PATH を差し替えても効かない
        self.path = pathlib.Path(path) if path else SETTINGS_PATH
        self._widgets: Dict[str, QWidget] = {}
        self._custom: Dict[str, tuple] = {}     # name -> (getter, setter)
        self._data: Dict[str, Any] = {}
        self.load_file()

    # ---- 登録 ----
    def add(self, name: str, widget: QWidget) -> None:
        self._widgets[name] = widget

    def add_many(self, **widgets: QWidget) -> None:
        self._widgets.update(widgets)

    def add_custom(self, name: str, getter: Callable[[], Any], setter: Callable[[Any], None]) -> None:
        """ウィジェットでない状態（ROI、カスタムスケールなど）用。"""
        self._custom[name] = (getter, setter)

    # ---- ファイル ----
    def load_file(self) -> None:
        try:
            self._data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(self._data, dict):
                self._data = {}
        except Exception:
            self._data = {}

    def save(self) -> bool:
        """登録済みの値を集めて書き出す。失敗しても例外は出さない。"""
        out = dict(self._data)
        for name, w in self._widgets.items():
            try:
                out[name] = _get(w)
            except Exception:
                pass
        for name, (getter, _s) in self._custom.items():
            try:
                out[name] = getter()
            except Exception:
                pass
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(self.path)       # 書き込み途中で落ちても既存の設定を壊さない
            self._data = out
            return True
        except Exception:
            return False

    def restore(self, only: Optional[list] = None) -> int:
        """保存されている値をウィジェットへ戻す。戻り値は復元できた項目数。"""
        n = 0
        for name, w in self._widgets.items():
            if only is not None and name not in only:
                continue
            if name not in self._data:
                continue
            try:
                _set(w, self._data[name]); n += 1
            except Exception:
                pass
        for name, (_g, setter) in self._custom.items():
            if only is not None and name not in only:
                continue
            if name not in self._data:
                continue
            try:
                setter(self._data[name]); n += 1
            except Exception:
                pass
        return n

    def get(self, name: str, default: Any = None) -> Any:
        return self._data.get(name, default)
