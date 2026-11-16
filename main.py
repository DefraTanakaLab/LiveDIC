"""
LiveDIC  —  インターバル撮影画像のリアルタイムDIC解析ツール
フォルダを監視し、新しい画像が来るたびに自動でDIC計算・表示を更新する。

モード:
  2点間モード : 登録ゲージ点のひずみをリアルタイムグラフ表示
  マップモード : フルフィールドひずみマップをバックグラウンド計算・スライダーで閲覧
"""
from __future__ import annotations
import sys, os, csv, re, time, threading, pathlib, platform, multiprocessing, traceback
import cv2
from datetime import datetime
from typing import Optional, List, Dict, Tuple
import numpy as np

# ---- matplotlib 日本語フォント設定（PyQt6 より先に）----
import matplotlib
matplotlib.use("QtAgg")
from matplotlib import font_manager as _fm
_JP_FONTS = {"Windows": ["Yu Gothic", "Meiryo", "MS Gothic"],
             "Darwin":  ["Hiragino Sans", "Hiragino Maru Gothic Pro"],
             "Linux":   ["IPAexGothic", "Noto Sans CJK JP"]}
_avail = {f.name for f in _fm.fontManager.ttflist}
_found = next((f for f in _JP_FONTS.get(platform.system(), []) if f in _avail), None)
if _found:
    matplotlib.rcParams["font.family"] = _found
matplotlib.rcParams["axes.unicode_minus"] = False

import matplotlib.colors as mcolors
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle as MplRectangle

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QTabWidget, QGroupBox, QLabel, QPushButton, QLineEdit,
    QComboBox, QSpinBox, QDoubleSpinBox, QRadioButton, QButtonGroup,
    QTableWidget, QTableWidgetItem, QHeaderView, QSlider,
    QSizePolicy, QFormLayout, QAbstractItemView, QFileDialog,
    QRubberBand, QScrollArea, QCheckBox, QMessageBox, QStatusBar,
    QStackedWidget, QPlainTextEdit,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QRect, QSize, QPoint, QTimer
from PyQt6.QtGui import (QPixmap, QImage, QPainter, QPen, QColor,
                          QFont, QBrush, QResizeEvent)

import dic_core
import rt_panel
import rt_source
import settings_store

# ===========================================================================
# 定数
# ===========================================================================
IMG_EXTS = {".bmp", ".png", ".tif", ".tiff", ".jpg", ".jpeg"}
STRAIN_KEYS = ["exx", "eyy", "exy"]
STRAIN_LABELS = {"exx": "εxx", "eyy": "εyy", "exy": "εxy"}

DARK = """
QMainWindow,QWidget          {background:#1e1e1e;color:#e0e0e0;}
QGroupBox                    {border:1px solid #444;border-radius:4px;
                              margin-top:10px;padding-top:6px;}
QGroupBox::title             {color:#aaa;subcontrol-origin:margin;left:8px;}
QPushButton                  {background:#333;border:1px solid #555;
                              border-radius:3px;padding:4px 12px;}
QPushButton:hover            {background:#444;}
QPushButton:disabled         {background:#2a2a2a;color:#555;}
QPushButton#btn_start        {background:#2e7d32;color:#fff;font-weight:bold;}
QPushButton#btn_start:hover  {background:#388e3c;}
QPushButton#btn_stop         {background:#c62828;color:#fff;font-weight:bold;}
QPushButton#btn_stop:hover   {background:#d32f2f;}
QLineEdit,QSpinBox,QDoubleSpinBox,QComboBox
                             {background:#2a2a2a;border:1px solid #555;
                              border-radius:2px;padding:2px 4px;color:#e0e0e0;}
QTableWidget                 {background:#252525;gridline-color:#3a3a3a;
                              alternate-background-color:#282828;}
QHeaderView::section         {background:#333;color:#ccc;
                              border:1px solid #444;padding:4px;}
QTabWidget::pane             {border:1px solid #444;}
QTabBar::tab                 {background:#2a2a2a;color:#999;
                              padding:6px 18px;border:1px solid #444;}
QTabBar::tab:selected        {background:#333;color:#fff;}
QSlider::groove:horizontal   {background:#333;height:4px;border-radius:2px;}
QSlider::handle:horizontal   {background:#4fc3f7;width:14px;height:14px;
                              border-radius:7px;margin:-5px 0;}
QSplitter::handle            {background:#333;}
QCheckBox,QRadioButton       {color:#e0e0e0;spacing:5px;}
QCheckBox::indicator,QRadioButton::indicator
                             {width:14px;height:14px;background:#3c3c3c;
                              border:1px solid #9a9a9a;border-radius:3px;}
QRadioButton::indicator      {border-radius:8px;}
QCheckBox::indicator:hover,QRadioButton::indicator:hover
                             {border-color:#4fc3f7;}
QCheckBox::indicator:checked,QRadioButton::indicator:checked
                             {background:#4fc3f7;border-color:#4fc3f7;}
QCheckBox::indicator:disabled,QRadioButton::indicator:disabled
                             {background:#2a2a2a;border-color:#555;}
/* チェック済みのまま無効化される項目（解析中の GPU・IC-GN 等）が未チェックに見えないよう、
   :disabled の後に :checked:disabled を置いて「暗いが点いている」色に戻す */
QCheckBox::indicator:checked:disabled,QRadioButton::indicator:checked:disabled
                             {background:#2f7490;border-color:#3f8aa8;}
QCheckBox:disabled,QRadioButton:disabled {color:#666;}
QScrollBar:vertical          {background:#252525;width:10px;}
QScrollBar::handle:vertical  {background:#555;border-radius:4px;}
"""

# ===========================================================================
# ユーティリティ
# ===========================================================================

def _natural_key(p: pathlib.Path):
    """Windows エクスプローラーと同じ自然順ソートキー（数字部分を数値比較）。"""
    return [int(c) if c.isdigit() else c.lower()
            for c in re.split(r'(\d+)', p.name)]


def _shift_image(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """画像を(dx, dy)ピクセル平行移動する。はみ出た部分はゼロ埋め。"""
    result = np.zeros_like(img)
    h, w = img.shape[:2]
    sx0 = max(0, -dx); sx1 = min(w, w - dx)
    dx0 = max(0,  dx); dx1 = min(w, w + dx)
    sy0 = max(0, -dy); sy1 = min(h, h - dy)
    dy0 = max(0,  dy); dy1 = min(h, h + dy)
    if sx1 > sx0 and sy1 > sy0 and dx1 > dx0 and dy1 > dy0:
        result[dy0:dy1, dx0:dx1] = img[sy0:sy1, sx0:sx1]
    return result


def list_images_sorted(folder: pathlib.Path, mode: str) -> List[pathlib.Path]:
    files = [p for p in folder.iterdir()
             if p.is_file() and p.suffix.lower() in IMG_EXTS]
    if mode == "mtime":
        return sorted(files, key=lambda p: p.stat().st_mtime_ns)
    return sorted(files, key=_natural_key)


def ndarray_to_qpixmap(img: np.ndarray) -> QPixmap:
    """numpy uint8 グレースケールを QPixmap に変換。"""
    if img.dtype != np.uint8:
        mn, mx = float(np.nanmin(img)), float(np.nanmax(img))
        if mx > mn:
            img = ((img - mn) / (mx - mn) * 255).astype(np.uint8)
        else:
            img = np.zeros_like(img, dtype=np.uint8)
    h, w = img.shape
    arr = np.ascontiguousarray(img)
    qi = QImage(arr.tobytes(), w, h, w, QImage.Format.Format_Grayscale8)
    return QPixmap.fromImage(qi)


def strain_to_qpixmap(
    data: np.ndarray,
    vmin: Optional[float], vmax: Optional[float],
    cmap_name: str = "viridis",
    X: Optional[np.ndarray] = None,
    Y: Optional[np.ndarray] = None,
    img_shape: Optional[tuple] = None,
) -> QPixmap:
    """ひずみ配列をカラーマップで QPixmap に変換。
    X, Y, img_shape を渡すとグリッド座標に基づき元画像サイズに正しく配置する。
    渡さない場合はデータ配列をそのままピクセルにする（後方互換）。
    """
    valid = data[np.isfinite(data)]
    if vmin is None:
        vmin = float(valid.min()) if valid.size else -0.01
    if vmax is None:
        vmax = float(valid.max()) if valid.size else 0.01
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap_fn = matplotlib.colormaps[cmap_name]   # cm.get_cmap は matplotlib 3.9 で削除
    rgba = cmap_fn(norm(data), bytes=True)   # (ny, nx, 4) uint8
    rgba[~np.isfinite(data)] = [0, 0, 0, 0]  # NaN は透明

    if X is not None and Y is not None and img_shape is not None:
        ny, nx = data.shape
        H, W = img_shape[:2]
        xs = X[0, :].astype(int)   # 列方向グリッド座標
        ys = Y[:, 0].astype(int)   # 行方向グリッド座標
        step_x = int(round(float(xs[1] - xs[0]))) if nx > 1 else 20
        step_y = int(round(float(ys[1] - ys[0]))) if ny > 1 else 20
        half_x = max(1, step_x // 2)
        half_y = max(1, step_y // 2)

        # グリッド点ごとにセル（half×half の矩形）を元画像サイズのキャンバスに描く
        rgba_full = np.zeros((H, W, 4), dtype=np.uint8)  # 透明で初期化
        for j, cy in enumerate(ys):
            y0 = max(0, cy - half_y)
            y1 = min(H, cy + half_y + 1)
            if y1 <= y0:
                continue
            for i, cx in enumerate(xs):
                x0 = max(0, cx - half_x)
                x1 = min(W, cx + half_x + 1)
                if x1 <= x0:
                    continue
                rgba_full[y0:y1, x0:x1] = rgba[j, i]

        arr = np.ascontiguousarray(rgba_full)
        qi = QImage(arr.tobytes(), W, H, W * 4, QImage.Format.Format_RGBA8888)
    else:
        h, w = data.shape
        arr = np.ascontiguousarray(rgba)
        qi = QImage(arr.tobytes(), w, h, w * 4, QImage.Format.Format_RGBA8888)

    return QPixmap.fromImage(qi)


# ===========================================================================
# 画像表示ウィジェット（クリック・ROI選択対応）
# ===========================================================================

class ImageCanvas(QLabel):
    """
    画像表示 + ROI矩形ドラッグ + 点クリック通知。
    signal_clicked(x_img, y_img)  : 画像座標でのクリック
    signal_roi(x1,y1,x2,y2)       : 画像座標でのROI確定
    """
    signal_clicked = pyqtSignal(int, int)
    signal_roi = pyqtSignal(int, int, int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background:#1a1a1a;border:1px solid #3a3a3a;")
        self.setMinimumSize(200, 150)

        self._pixmap_orig: Optional[QPixmap] = None
        self._overlays: List[Dict] = []   # {"type":"point","x","y","color","label"} etc.
        self._roi_rubber: Optional[QRubberBand] = None
        self._roi_origin: Optional[QPoint] = None
        self._mode = "none"   # "none" | "roi" | "click"
    # ---- 外部API ----
    def set_image(self, pix: QPixmap):
        self._pixmap_orig = pix
        self._refresh()

    def set_mode(self, mode: str):
        self._mode = mode
        self.setCursor(Qt.CursorShape.CrossCursor if mode != "none"
                       else Qt.CursorShape.ArrowCursor)

    def add_overlay(self, item: Dict):
        self._overlays.append(item)
        self._refresh()

    def clear_overlays(self):
        self._overlays.clear()
        self._refresh()

    # ---- 内部 ----
    def _widget_to_img(self, wx: int, wy: int) -> Tuple[int, int]:
        """ウィジェット座標 → 画像座標変換。"""
        if self._pixmap_orig is None:
            return wx, wy
        pw, ph = self._pixmap_orig.width(), self._pixmap_orig.height()
        ww, wh = self.width(), self.height()
        scale = min(ww / pw, wh / ph)
        ox = (ww - pw * scale) / 2
        oy = (wh - ph * scale) / 2
        ix = int((wx - ox) / scale)
        iy = int((wy - oy) / scale)
        return np.clip(ix, 0, pw - 1), np.clip(iy, 0, ph - 1)

    def _img_to_widget(self, ix: int, iy: int) -> Tuple[float, float]:
        if self._pixmap_orig is None:
            return float(ix), float(iy)
        pw, ph = self._pixmap_orig.width(), self._pixmap_orig.height()
        ww, wh = self.width(), self.height()
        scale = min(ww / pw, wh / ph)
        ox = (ww - pw * scale) / 2
        oy = (wh - ph * scale) / 2
        return ox + ix * scale, oy + iy * scale

    def _refresh(self):
        if self._pixmap_orig is None:
            self.setText("画像なし")
            return
        pw, ph = self._pixmap_orig.width(), self._pixmap_orig.height()
        ww, wh = max(1, self.width()), max(1, self.height())
        scale = min(ww / pw, wh / ph)
        disp = self._pixmap_orig.scaled(
            int(pw * scale), int(ph * scale),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation)
        ox = (ww - disp.width()) / 2
        oy = (wh - disp.height()) / 2

        canvas = QPixmap(ww, wh)
        canvas.fill(QColor("#1a1a1a"))
        p = QPainter(canvas)
        p.drawPixmap(int(ox), int(oy), disp)

        for item in self._overlays:
            t = item.get("type")
            color = QColor(item.get("color", "#ff4444"))
            if t == "point":
                wx, wy = self._img_to_widget(item["x"], item["y"])
                p.setPen(QPen(color, 2))
                r = 6
                p.drawEllipse(int(wx) - r, int(wy) - r, 2 * r, 2 * r)
                if "label" in item:
                    p.setPen(QPen(color))
                    p.drawText(int(wx) + 8, int(wy) + 4, item["label"])
            elif t == "line":
                wx1, wy1 = self._img_to_widget(item["x1"], item["y1"])
                wx2, wy2 = self._img_to_widget(item["x2"], item["y2"])
                p.setPen(QPen(color, 1, Qt.PenStyle.DashLine))
                p.drawLine(int(wx1), int(wy1), int(wx2), int(wy2))
            elif t == "rect":
                wx1, wy1 = self._img_to_widget(item["x1"], item["y1"])
                wx2, wy2 = self._img_to_widget(item["x2"], item["y2"])
                p.setPen(QPen(color, 2))
                p.setBrush(QBrush(Qt.BrushStyle.NoBrush))
                p.drawRect(int(wx1), int(wy1),
                           int(wx2 - wx1), int(wy2 - wy1))
            elif t == "grid":
                pts = item.get("points", [])
                p.setPen(QPen(QColor("#44aaff"), 1))
                for gx, gy in pts:
                    wx_, wy_ = self._img_to_widget(gx, gy)
                    p.drawPoint(int(wx_), int(wy_))
        p.end()
        self.setPixmap(canvas)

    def resizeEvent(self, ev: QResizeEvent):
        super().resizeEvent(ev)
        self._refresh()

    # ---- マウスイベント ----
    def mousePressEvent(self, ev):
        if self._mode == "click":
            ix, iy = self._widget_to_img(ev.pos().x(), ev.pos().y())
            self.signal_clicked.emit(ix, iy)
        elif self._mode == "roi":
            self._roi_origin = ev.pos()
            self._roi_rubber = QRubberBand(QRubberBand.Shape.Rectangle, self)
            self._roi_rubber.setGeometry(QRect(self._roi_origin, QSize()))
            self._roi_rubber.show()

    def mouseMoveEvent(self, ev):
        if self._mode == "roi" and self._roi_rubber:
            self._roi_rubber.setGeometry(
                QRect(self._roi_origin, ev.pos()).normalized())

    def mouseReleaseEvent(self, ev):
        if self._mode == "roi" and self._roi_rubber:
            self._roi_rubber.hide()
            self._roi_rubber = None
            ix1, iy1 = self._widget_to_img(
                self._roi_origin.x(), self._roi_origin.y())
            ix2, iy2 = self._widget_to_img(ev.pos().x(), ev.pos().y())
            if abs(ix2 - ix1) > 5 and abs(iy2 - iy1) > 5:
                self.signal_roi.emit(
                    min(ix1, ix2), min(iy1, iy2),
                    max(ix1, ix2), max(iy1, iy2))
            self.set_mode("none")


# ===========================================================================
# グラフウィジェット
# ===========================================================================

class LiveGraph(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        fig = Figure(facecolor="#1e1e1e")
        super().__init__(fig)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.ax = fig.add_subplot(111, facecolor="#2a2a2a")
        self._style_ax()
        fig.tight_layout(pad=1.5)
        self._lines: Dict[str, object] = {}
        self._xs: Dict[str, list] = {}
        self._ys: Dict[str, list] = {}

    def _style_ax(self):
        ax = self.ax
        for obj in [ax.xaxis.label, ax.yaxis.label]:
            obj.set_color("#ccc")
        ax.tick_params(colors="#ccc", labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#555")
        ax.grid(True, alpha=0.15)

    def set_labels(self, xlabel: str, ylabel: str):
        self.ax.set_xlabel(xlabel, fontsize=9)
        self.ax.set_ylabel(ylabel, fontsize=9)
        self.draw_idle()

    def add_series(self, key: str, label: str, color: str):
        (line,) = self.ax.plot([], [], "o-", ms=4, lw=1.5,
                               color=color, label=label)
        self._lines[key] = line
        self._xs[key] = []
        self._ys[key] = []
        self.ax.legend(facecolor="#333", labelcolor="#ccc",
                       fontsize=8, framealpha=0.8)

    def update_series(self, key: str, x: float, y: float):
        if key not in self._lines:
            return
        self._xs[key].append(x)
        self._ys[key].append(y)
        self._lines[key].set_data(self._xs[key], self._ys[key])
        self.ax.relim()
        self.ax.autoscale_view()
        # Y軸が逆転していたら修正
        ymin, ymax = self.ax.get_ylim()
        if ymin > ymax:
            self.ax.set_ylim(ymax, ymin)
        self.draw_idle()

    def clear_all(self):
        for key in list(self._lines.keys()):
            self._lines[key].set_data([], [])
            self._xs[key] = []
            self._ys[key] = []
        self.ax.relim()
        self.draw_idle()


# ===========================================================================
# 手動アライメントキャンバス
# ===========================================================================

class AlignmentCanvas(QWidget):
    """参照画像と変形画像を半透明合成し、マウスドラッグでオフセット調整するキャンバス。
    QWidget + paintEvent で実装：ページ切替後に正しいサイズで確実に描画される。
    """
    sig_offset_changed = pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(200, 150)
        self.setStyleSheet("background:#1a1a1a;border:1px solid #3a3a3a;")
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self._ref: Optional[np.ndarray] = None
        self._def: Optional[np.ndarray] = None
        self._dx = 0
        self._dy = 0
        self._alpha = 0.5
        self._drag_start: Optional[QPoint] = None
        self._drag_offset_start = (0, 0)
        self._blend_arr: Optional[np.ndarray] = None   # uint8 blended array

    def set_images(self, ref: np.ndarray, def_: np.ndarray):
        self._ref = ref.astype(np.float32)
        self._def = def_.astype(np.float32)
        self._dx = 0
        self._dy = 0
        self._rebuild()

    def set_alpha(self, alpha: float):
        self._alpha = max(0.0, min(1.0, alpha))
        self._rebuild()

    def get_offset(self) -> Tuple[int, int]:
        return self._dx, self._dy

    def _rebuild(self):
        if self._ref is None or self._def is None:
            return
        try:
            h, w = self._ref.shape
            shifted = np.zeros_like(self._def)
            sx0 = max(0, -self._dx); sx1 = min(w, w - self._dx)
            dx0 = max(0,  self._dx); dx1 = min(w, w + self._dx)
            sy0 = max(0, -self._dy); sy1 = min(h, h - self._dy)
            dy0 = max(0,  self._dy); dy1 = min(h, h + self._dy)
            if sx1 > sx0 and sy1 > sy0 and dx1 > dx0 and dy1 > dy0:
                shifted[dy0:dy1, dx0:dx1] = self._def[sy0:sy1, sx0:sx1]
            blended = ((1 - self._alpha) * self._ref
                       + self._alpha * shifted).clip(0, 255).astype(np.uint8)
            self._blend_arr = np.ascontiguousarray(blended)
        except Exception:
            self._blend_arr = None
        self.update()   # paintEvent を要求（表示タイミングで正しいサイズが使われる）

    def _img_scale(self) -> float:
        if self._blend_arr is None:
            return 1.0
        h, w = self._blend_arr.shape
        return min(max(1, self.width()) / max(1, w),
                   max(1, self.height()) / max(1, h))

    def paintEvent(self, ev):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#1a1a1a"))
        if self._blend_arr is not None:
            h, w = self._blend_arr.shape
            ww, wh = self.width(), self.height()
            scale = min(ww / max(1, w), wh / max(1, h))
            disp_w = max(1, int(w * scale))
            disp_h = max(1, int(h * scale))
            ox = (ww - disp_w) // 2
            oy = (wh - disp_h) // 2
            qi = QImage(self._blend_arr.data, w, h, w,
                        QImage.Format.Format_Grayscale8)
            pix = QPixmap.fromImage(qi).scaled(
                disp_w, disp_h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation)
            painter.drawPixmap(ox, oy, pix)
        painter.end()

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            self._drag_start = ev.pos()
            self._drag_offset_start = (self._dx, self._dy)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, ev):
        if self._drag_start is not None:
            scale = self._img_scale()
            ddx = int(round((ev.pos().x() - self._drag_start.x()) / scale))
            ddy = int(round((ev.pos().y() - self._drag_start.y()) / scale))
            self._dx = self._drag_offset_start[0] + ddx
            self._dy = self._drag_offset_start[1] + ddy
            self._rebuild()
            self.sig_offset_changed.emit(self._dx, self._dy)

    def mouseReleaseEvent(self, ev):
        self._drag_start = None
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self.update()


# ===========================================================================
# サブセット画像ウィンドウ
# ===========================================================================

class SubsetWindow(QWidget):
    """追跡点のサブセット画像を表示する独立ウィンドウ。"""

    DISP = 60  # 画像表示サイズ [px]

    def __init__(self, parent=None):
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle("サブセット画像（追跡点）")
        self.setStyleSheet(DARK)
        self.setMinimumWidth(200)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("background:#1a1a1a;border:none;")
        self._inner = QWidget()
        self._lay = QHBoxLayout(self._inner)
        self._lay.setSpacing(8)
        self._lay.addStretch()
        scroll.setWidget(self._inner)
        lay.addWidget(scroll)

        self._adjust_height()

    def _adjust_height(self):
        h = self.DISP + 20 + 6 * 2 + 12   # 画像 + ラベル + マージン + タイトルバー余白
        self.setFixedHeight(h)

    def update_patches(self, gauges: list, gauge_results: list):
        lay = self._lay
        while lay.count() > 1:
            item = lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for gi, g in enumerate(gauges):
            gr = gauge_results[gi] if gi < len(gauge_results) else {}
            color = g["color"]
            for pt_label, patch_key in [("点1", "patch1"), ("点2", "patch2")]:
                patch = gr.get(patch_key)
                frame_w = QWidget()
                flay = QVBoxLayout(frame_w)
                flay.setSpacing(2)
                flay.setContentsMargins(0, 0, 0, 0)

                img_lbl = QLabel()
                img_lbl.setFixedSize(self.DISP, self.DISP)
                img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                img_lbl.setStyleSheet(f"border:2px solid {color};background:#111;")
                if patch is not None and patch.size > 0:
                    arr = np.ascontiguousarray(patch)
                    h, w = arr.shape
                    qi = QImage(arr.tobytes(), w, h, w, QImage.Format.Format_Grayscale8)
                    pix = QPixmap.fromImage(qi).scaled(
                        self.DISP, self.DISP,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.FastTransformation)
                    img_lbl.setPixmap(pix)
                else:
                    img_lbl.setText("—")

                cap = QLabel(f"{g['name']} {pt_label}")
                cap.setStyleSheet(f"color:{color};font-size:8px;")
                cap.setAlignment(Qt.AlignmentFlag.AlignCenter)

                flay.addWidget(img_lbl)
                flay.addWidget(cap)
                lay.insertWidget(lay.count() - 1, frame_w)

        self._inner.adjustSize()


# ===========================================================================
# スペックル品質評価ダイアログ
# ===========================================================================

class QualityDialog(QWidget):
    """SSSIG マップ・スペックルサイズ 3×3・MIG を表示するウィンドウ。"""

    def __init__(self, mig: float, sssig: np.ndarray,
                 X: np.ndarray, Y: np.ndarray, img_shape: tuple,
                 speckle_sizes: np.ndarray,
                 roi_xyxy, subset_size: int = 31, parent=None):
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle("スペックル品質評価")
        self.setStyleSheet(DARK)
        self.resize(1100, 660)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        lbl = QLabel(f"MIG（平均輝度勾配） = {mig:.4f}")
        lbl.setStyleSheet("font-size:13px;font-weight:bold;color:#4fc3f7;padding:2px;")
        root.addWidget(lbl)

        # layout="constrained" でカラーバーが自動的に収まる（tight_layout より確実）
        fig = Figure(facecolor="#1e1e1e", figsize=(13, 4.8), layout="constrained")

        # ---- 左: SSSIG マップ ----
        ax_s = fig.add_subplot(1, 2, 1, facecolor="#2a2a2a")
        ax_s.set_title("SSSIG マップ", color="#ccc", fontsize=10)
        H, W = img_shape[:2]
        xs = X[0, :].astype(int)
        ys = Y[:, 0].astype(int)
        step_x = int(xs[1] - xs[0]) if len(xs) > 1 else 20
        step_y = int(ys[1] - ys[0]) if len(ys) > 1 else 20
        # pcolormesh 用のセル境界を生成
        xe = np.concatenate([[xs[0]  - step_x // 2],
                              (xs[:-1] + xs[1:]) / 2,
                              [xs[-1] + step_x // 2]])
        ye = np.concatenate([[ys[0]  - step_y // 2],
                              (ys[:-1] + ys[1:]) / 2,
                              [ys[-1] + step_y // 2]])
        im_s = ax_s.pcolormesh(xe, ye, sssig, cmap="hot", shading="flat")
        ax_s.set_xlim(0, W); ax_s.set_ylim(H, 0)
        ax_s.set_aspect("equal")
        if roi_xyxy:
            x1r, y1r, x2r, y2r = roi_xyxy
            ax_s.add_patch(MplRectangle(
                (x1r, y1r), x2r - x1r, y2r - y1r,
                edgecolor="#ffaa00", facecolor="none", lw=1.5))
        cb_s = fig.colorbar(im_s, ax=ax_s, fraction=0.046, pad=0.04)
        cb_s.ax.yaxis.set_tick_params(color="#ccc", labelcolor="#ccc")
        ax_s.set_xlabel("X [px]", color="#ccc", fontsize=8)
        ax_s.set_ylabel("Y [px]", color="#ccc", fontsize=8)
        ax_s.tick_params(colors="#ccc", labelsize=7)
        for sp in ax_s.spines.values():
            sp.set_color("#555")

        # ---- 右: スペックルサイズ 3×3 ----
        ax_p = fig.add_subplot(1, 2, 2, facecolor="#2a2a2a")
        ax_p.set_title("スペックルサイズ [px]（3×3 分割）", color="#ccc", fontsize=10)
        n = speckle_sizes.shape[0]
        vmin_p = float(speckle_sizes.min())
        vmax_p = float(speckle_sizes.max())
        if vmax_p <= vmin_p:
            vmax_p = vmin_p + 1.0
        if roi_xyxy:
            rx1, ry1, rx2, ry2 = roi_xyxy
            cell_aspect = abs(ry2 - ry1) / max(1, abs(rx2 - rx1))
        else:
            cell_aspect = H / max(1, W)
        im_p = ax_p.imshow(speckle_sizes, cmap="YlOrRd",
                            vmin=vmin_p, vmax=vmax_p, aspect=cell_aspect)
        for i in range(n):
            for j in range(n):
                v = float(speckle_sizes[i, j])
                ax_p.text(j, i, f"{v:.1f}",
                          ha="center", va="center", fontsize=14, fontweight="bold",
                          color="black")
        ax_p.set_xticks([]); ax_p.set_yticks([])
        for sp in ax_p.spines.values():
            sp.set_color("#555")
        cb_p = fig.colorbar(im_p, ax=ax_p, fraction=0.046, pad=0.04)
        cb_p.ax.yaxis.set_tick_params(color="#ccc", labelcolor="#ccc")

        canvas = FigureCanvasQTAgg(fig)
        canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        root.addWidget(canvas)

        # ---- 判定基準パネル ----
        sssig_thr = subset_size ** 2 * 0.03   # Pan et al. 推奨閾値
        valid_sssig = sssig[sssig > 0]
        sssig_med = float(np.median(valid_sssig)) if valid_sssig.size > 0 else 0.0
        sp_med    = float(np.median(speckle_sizes))

        def _judge(ok: bool) -> str:
            return "<span style='color:#81c784;'>✓ 良好</span>" if ok \
                   else "<span style='color:#ffb74d;'>△ 要確認</span>"

        mig_ok    = mig >= 0.05
        sssig_ok  = sssig_med >= sssig_thr
        sp_ok     = 3.0 <= sp_med <= 5.0

        guide_html = (
            "<table style='border-collapse:collapse;width:100%;font-size:10px;'>"
            "<tr style='background:#252525;'>"
            "<th style='padding:4px 10px;color:#aaa;text-align:left;'>指標</th>"
            "<th style='padding:4px 10px;color:#aaa;text-align:center;'>今回の値</th>"
            "<th style='padding:4px 10px;color:#aaa;text-align:center;'>判定基準</th>"
            "<th style='padding:4px 10px;color:#aaa;text-align:center;'>判定</th>"
            "<th style='padding:4px 10px;color:#aaa;text-align:left;'>解説</th>"
            "</tr>"
            f"<tr>"
            f"<td style='padding:3px 10px;color:#e0e0e0;'>MIG（平均輝度勾配）</td>"
            f"<td style='padding:3px 10px;color:#4fc3f7;text-align:center;'>{mig:.4f}</td>"
            f"<td style='padding:3px 10px;color:#888;text-align:center;'>≥ 0.05</td>"
            f"<td style='padding:3px 10px;text-align:center;'>{_judge(mig_ok)}</td>"
            f"<td style='padding:3px 10px;color:#aaa;'>画像全体のコントラスト指標。0.05未満はスペックルが薄すぎる</td>"
            f"</tr>"
            f"<tr style='background:#222;'>"
            f"<td style='padding:3px 10px;color:#e0e0e0;'>SSSIG（中央値）</td>"
            f"<td style='padding:3px 10px;color:#4fc3f7;text-align:center;'>{sssig_med:.0f}</td>"
            f"<td style='padding:3px 10px;color:#888;text-align:center;'>≥ {sssig_thr:.0f}　（サブセット{subset_size}×{subset_size}px の場合）</td>"
            f"<td style='padding:3px 10px;text-align:center;'>{_judge(sssig_ok)}</td>"
            f"<td style='padding:3px 10px;color:#aaa;'>サブセット内の輝度勾配の豊かさ。低い点はマッチング精度が落ちる</td>"
            f"</tr>"
            f"<tr>"
            f"<td style='padding:3px 10px;color:#e0e0e0;'>スペックルサイズ（中央値）</td>"
            f"<td style='padding:3px 10px;color:#4fc3f7;text-align:center;'>{sp_med:.1f} px</td>"
            f"<td style='padding:3px 10px;color:#888;text-align:center;'>3–5 px が最適</td>"
            f"<td style='padding:3px 10px;text-align:center;'>{_judge(sp_ok)}</td>"
            f"<td style='padding:3px 10px;color:#aaa;'>小さすぎ（&lt;2 px）はエイリアシング発生、大きすぎ（&gt;10 px）は変位分解能が低下</td>"
            f"</tr>"
            "</table>"
        )
        guide_lbl = QLabel(guide_html)
        guide_lbl.setTextFormat(Qt.TextFormat.RichText)
        guide_lbl.setStyleSheet(
            "background:#1a1a1a;border:1px solid #3a3a3a;"
            "border-radius:3px;padding:2px;")
        root.addWidget(guide_lbl)


# ===========================================================================
# バックグラウンドワーカー（フォルダ監視 + DIC計算）
# ===========================================================================

class DICWorker(QThread):
    """
    フォルダを監視し、新画像を検出したらシグナルで通知。
    DIC計算はユーザーが trigger_dic() を呼ぶまで待機する。
    """
    sig_frame     = pyqtSignal(dict)   # フレーム結果
    sig_log       = pyqtSignal(str)    # ログメッセージ
    sig_error     = pyqtSignal(str)    # エラーメッセージ
    sig_new_image = pyqtSignal(str, str)  # (ref_path, def_path) DIC実行待ち

    def __init__(self, cfg: dict, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self._stop_flag = threading.Event()
        self._run_event = threading.Event()
        self._run_dx    = 0
        self._run_dy    = 0
        self._reanalyze_event  = threading.Event()
        self._reanalyze_dx     = 0
        self._reanalyze_dy     = 0
        self._last_def_path:    Optional[str]          = None
        self._last_run_dx      = 0
        self._last_run_dy      = 0
        self._reanalyze_ref_img: Optional[np.ndarray] = None
        self._last_u_contrib:   Optional[np.ndarray]  = None
        self._last_v_contrib:   Optional[np.ndarray]  = None
        self._last_frame_meta:  Optional[dict]         = None

    def stop(self):
        self._stop_flag.set()
        self._run_event.set()   # waitを解除してスレッドを終了させる

    def trigger_dic(self, dx: int, dy: int):
        """GUIの「DIC実行」ボタンから呼ぶ。アライメントオフセットを受け取り計算を開始。"""
        self._run_dx = dx
        self._run_dy = dy
        self._run_event.set()

    def trigger_reanalyze(self, dx: int, dy: int):
        """直近フレームをアライメント調整済みで再計算する。"""
        self._reanalyze_dx = dx
        self._reanalyze_dy = dy
        self._reanalyze_event.set()

    def run(self):
        cfg = self.cfg
        folder      = pathlib.Path(cfg["folder"])
        sort_mode   = cfg["sort_mode"]
        ref_mode    = cfg["ref_mode"]      # "fixed" | "incremental"
        subset_size = cfg["subset_size"]
        step        = cfg["step"]
        search_x    = cfg["search_x"]
        search_y    = cfg["search_y"]
        method      = cfg["subpixel"]
        zncc_thr    = cfg["zncc_threshold"]
        mode        = cfg["mode"]          # "twopoint" | "map"
        gauges      = cfg.get("gauges", [])  # [{cx1,cy1,cx2,cy2,comp}]
        n_workers   = cfg.get("n_workers", -1)

        seen: set = set()
        ref_img: Optional[np.ndarray] = None
        ref_path: Optional[pathlib.Path] = None
        prev_path: Optional[pathlib.Path] = None   # アライメント表示用（直前フレームのパス）
        frame_idx = 0
        prev_u_map = prev_v_map = None    # 前フレームの変位（探索初期値用）
        align_cum_dx = 0  # 固定参照モード用：image0 からの累積アライメントオフセット
        align_cum_dy = 0
        cum_u_map: Optional[np.ndarray] = None   # 増分参照モード用：累積変位マップ
        cum_v_map: Optional[np.ndarray] = None
        gauge_prev_u  = [0.0] * len(gauges)
        gauge_prev_v  = [0.0] * len(gauges)
        gauge_cum_strain = [0.0] * len(gauges)  # 増分参照モード用累積ひずみ
        gauge_cum_u1  = [0.0] * len(gauges)     # 点1の累積変位（サブセット表示用）
        gauge_cum_v1  = [0.0] * len(gauges)
        gauge_cum_u2  = [0.0] * len(gauges)     # 点2の累積変位
        gauge_cum_v2  = [0.0] * len(gauges)
        t_start: Optional[float] = None

        self.sig_log.emit("監視開始")

        while not self._stop_flag.is_set():
            # ---- 再解析リクエストの処理 ----
            if (self._reanalyze_event.is_set()
                    and self._last_def_path is not None
                    and self._reanalyze_ref_img is not None
                    and self._last_frame_meta is not None):
                self._reanalyze_event.clear()
                self.sig_log.emit("再解析中...")
                try:
                    r_def_raw = dic_core.read_gray(self._last_def_path)
                    align_cum_dx = align_cum_dx - self._last_run_dx + self._reanalyze_dx
                    align_cum_dy = align_cum_dy - self._last_run_dy + self._reanalyze_dy
                    self._last_run_dx = self._reanalyze_dx
                    self._last_run_dy = self._reanalyze_dy
                    r_def = (_shift_image(r_def_raw, align_cum_dx, align_cum_dy)
                             if (align_cum_dx != 0 or align_cum_dy != 0) else r_def_raw)
                    if (ref_mode == "incremental" and cum_u_map is not None
                            and self._last_u_contrib is not None):
                        mask_rb = np.isfinite(self._last_u_contrib) & np.isfinite(self._last_v_contrib)
                        cum_u_map[mask_rb] -= self._last_u_contrib[mask_rb]
                        cum_v_map[mask_rb]  -= self._last_v_contrib[mask_rb]
                    iu_ra = iv_ra = None
                    if ref_mode == "fixed" and prev_u_map is not None:
                        iu_ra = np.where(np.isfinite(prev_u_map), prev_u_map + float(align_cum_dx), 0.0)
                        iv_ra = np.where(np.isfinite(prev_v_map), prev_v_map + float(align_cum_dy), 0.0)
                    map_ra = dic_core.run_dic_full(
                        self._reanalyze_ref_img, r_def,
                        subset_size=subset_size, step=step,
                        search_x=search_x, search_y=search_y, method=method,
                        zncc_threshold=zncc_thr, roi_xyxy=self.cfg.get("roi_xyxy"),
                        n_workers=n_workers, init_u_grid=iu_ra, init_v_grid=iv_ra,
                        icgn=self.cfg.get("icgn", False),
                    )
                    if ref_mode == "fixed":
                        _pu = map_ra["U"].copy(); _pv = map_ra["V"].copy()
                        _pu[np.isfinite(_pu)] -= float(align_cum_dx)
                        _pv[np.isfinite(_pv)] -= float(align_cum_dy)
                        prev_u_map, prev_v_map = _pu, _pv
                        map_ra.update({"cum_exx": map_ra["exx"],
                                       "cum_eyy": map_ra["eyy"],
                                       "cum_exy": map_ra["exy"]})
                    else:
                        u_ra = map_ra["U"]; v_ra = map_ra["V"]
                        if cum_u_map is None or cum_u_map.shape != u_ra.shape:
                            cum_u_map = np.zeros_like(u_ra); cum_v_map = np.zeros_like(v_ra)
                        mask_ra = np.isfinite(u_ra) & np.isfinite(v_ra)
                        cum_u_map[mask_ra] += u_ra[mask_ra]
                        cum_v_map[mask_ra] += v_ra[mask_ra]
                        cum_s = dic_core.calc_strain_field(
                            map_ra["X"], map_ra["Y"], cum_u_map, cum_v_map, step)
                        map_ra.update({"cum_exx": cum_s["exx"],
                                       "cum_eyy": cum_s["eyy"],
                                       "cum_exy": cum_s["exy"]})
                        self._last_u_contrib   = u_ra.copy()
                        self._last_v_contrib   = v_ra.copy()
                        self._reanalyze_ref_img = r_def
                        ref_img = r_def
                    self.sig_frame.emit({
                        **self._last_frame_meta,
                        "is_ref": False, "is_reanalyze": True,
                        "gauges": [], "map": map_ra,
                    })
                    self.sig_log.emit("再解析完了")
                except Exception as e:
                    self.sig_error.emit(f"再解析エラー: {e}")

            images = list_images_sorted(folder, sort_mode)
            new_images = [p for p in images if p not in seen]

            for path in new_images:
                if self._stop_flag.is_set():
                    break

                # 書き込み完了を待つ（ファイルサイズが安定するまで）
                try:
                    size0 = path.stat().st_size
                    time.sleep(0.3)
                    if path.stat().st_size != size0:
                        continue   # まだ書き込み中
                except Exception:
                    continue

                seen.add(path)
                save_time = datetime.fromtimestamp(path.stat().st_mtime)

                # 参照画像の設定
                if ref_img is None:
                    try:
                        ref_img = dic_core.read_gray(str(path))
                        ref_path = path
                        prev_path = path   # 最初のフレームが直前パスの初期値
                        t_start = save_time.timestamp()
                        frame_idx += 1
                        self.sig_log.emit(f"参照画像設定: {path.name}")
                        self.sig_frame.emit({
                            "frame":     frame_idx,
                            "path":      path,
                            "save_time": save_time,
                            "elapsed":   0.0,
                            "is_ref":    True,
                            "gauges":    [],
                            "map":       None,
                        })
                    except Exception as e:
                        self.sig_error.emit(f"参照画像読み込みエラー: {e}")
                    continue

                # 変形画像
                try:
                    def_img = dic_core.read_gray(str(path))
                except Exception as e:
                    self.sig_error.emit(f"画像読み込みエラー: {path.name} — {e}")
                    continue
                orig_def_img = def_img   # アライメント補正前の生画像（参照更新用）

                frame_idx += 1
                elapsed = save_time.timestamp() - t_start
                self.sig_log.emit(f"Frame {frame_idx}: {path.name}  計算中...")

                result = {
                    "frame":     frame_idx,
                    "path":      path,
                    "save_time": save_time,
                    "elapsed":   elapsed,
                    "is_ref":    False,
                    "gauges":    [],
                    "map":       None,
                }

                if mode == "twopoint":
                    gauge_results = []
                    for gi, g in enumerate(gauges):
                        cx1, cy1 = g["cx1"], g["cy1"]
                        cx2, cy2 = g["cx2"], g["cy2"]
                        comp = g.get("comp", "exx")
                        try:
                            u1, v1, z1 = dic_core.compute_single_point(
                                ref_img, def_img, cx1, cy1, subset_size,
                                search_x, search_y,
                                method, gauge_prev_u[gi], gauge_prev_v[gi])
                            u2, v2, z2 = dic_core.compute_single_point(
                                ref_img, def_img, cx2, cy2, subset_size,
                                search_x, search_y,
                                method, gauge_prev_u[gi], gauge_prev_v[gi])

                            zncc_avg = (z1 + z2) / 2
                            ok = z1 >= zncc_thr and z2 >= zncc_thr

                            # DICが返した生の変位（サブセット表示用）
                            disp_u1, disp_v1 = u1, v1
                            disp_u2, disp_v2 = u2, v2

                            if ok:
                                strain_dict = dic_core.two_point_strain(
                                    cx1, cy1, u1, v1, cx2, cy2, u2, v2)
                                strain_val = strain_dict.get(comp, np.nan)

                                if ref_mode == "fixed":
                                    gauge_prev_u[gi] = (u1 + u2) / 2
                                    gauge_prev_v[gi] = (v1 + v2) / 2
                                else:
                                    gauge_cum_u1[gi] += u1; gauge_cum_v1[gi] += v1
                                    gauge_cum_u2[gi] += u2; gauge_cum_v2[gi] += v2

                                if ref_mode == "incremental" and np.isfinite(strain_val):
                                    gauge_cum_strain[gi] += strain_val
                                    plot_strain = gauge_cum_strain[gi]
                                else:
                                    plot_strain = strain_val
                            else:
                                u1 = gauge_prev_u[gi]; v1 = gauge_prev_v[gi]
                                u2 = gauge_prev_u[gi]; v2 = gauge_prev_v[gi]
                                plot_strain = np.nan

                            # サブセット切り出し：サブピクセル精度で追跡位置から切り出す
                            def_u8 = (def_img * 255).clip(0, 255).astype(np.uint8)
                            if ref_mode == "fixed":
                                fc_u1, fc_v1 = disp_u1, disp_v1
                                fc_u2, fc_v2 = disp_u2, disp_v2
                            else:
                                fc_u1, fc_v1 = gauge_cum_u1[gi], gauge_cum_v1[gi]
                                fc_u2, fc_v2 = gauge_cum_u2[gi], gauge_cum_v2[gi]
                            patch1 = cv2.getRectSubPix(
                                def_u8, (subset_size, subset_size),
                                (float(cx1 + fc_u1), float(cy1 + fc_v1)))
                            patch2 = cv2.getRectSubPix(
                                def_u8, (subset_size, subset_size),
                                (float(cx2 + fc_u2), float(cy2 + fc_v2)))
                            self.sig_log.emit(
                                f"  G{gi+1} u1={disp_u1:.3f} v1={disp_v1:.3f} "
                                f"cum=({fc_u1:.2f},{fc_v1:.2f}) ok={ok}")

                            gauge_results.append({
                                "gi": gi, "comp": comp,
                                "strain": plot_strain,
                                "zncc": zncc_avg,
                                "zncc1": z1, "zncc2": z2,
                                "ok": ok,
                                "u1": u1, "v1": v1,
                                "u2": u2, "v2": v2,
                                "cum_u1": fc_u1, "cum_v1": fc_v1,
                                "cum_u2": fc_u2, "cum_v2": fc_v2,
                                "patch1": patch1,
                                "patch2": patch2,
                            })
                        except Exception as e:
                            gauge_results.append({"gi": gi, "comp": comp,
                                                  "strain": np.nan, "zncc": 0.0,
                                                  "ok": False,
                                                  "u1": gauge_prev_u[gi], "v1": gauge_prev_v[gi],
                                                  "u2": gauge_prev_u[gi], "v2": gauge_prev_v[gi]})
                            self.sig_log.emit(f"  ゲージ{gi+1} エラー: {e}")
                    result["gauges"] = gauge_results

                elif mode == "map":
                    # ユーザーが「DIC実行」ボタンを押すまで待機
                    self._run_dx = 0
                    self._run_dy = 0
                    self._run_event.clear()
                    self.sig_new_image.emit(str(prev_path), str(path))  # 直前フレームと新フレームを送る
                    while not self._stop_flag.is_set():
                        if self._run_event.wait(timeout=2.0):
                            break
                    if self._stop_flag.is_set():
                        break
                    # アライメント補正適用
                    orig_def_img = def_img   # 生画像をここで保存（参照更新用）
                    # 固定・更新両モードとも、フレームごとのオフセットを累積して
                    # image0（最初のフレーム）座標系に統一する。
                    # 【更新参照モードで必要な理由】
                    #   ref_img = 前フレームのアライメント補正済み画像（image0座標系）
                    #   def_img = 現フレームを累積オフセットでシフト → image0座標系に合わせる
                    #   これにより X,Y グリッドが常に image0 の同じ素材点に対応する。
                    align_cum_dx += self._run_dx
                    align_cum_dy += self._run_dy
                    eff_dx, eff_dy = align_cum_dx, align_cum_dy
                    if eff_dx != 0 or eff_dy != 0:
                        def_img = _shift_image(def_img, eff_dx, eff_dy)
                        self.sig_log.emit(
                            f"  アライメント補正: dx={eff_dx:+d} dy={eff_dy:+d} (累積)")
                    try:
                        # 固定参照モードのみ前フレームの変位をinitとして使う。
                        # インクリメンタルモードは隣接フレーム間の変位が小さいので
                        # initに前フレームの累積変位を使うと探索窓がずれて失敗する。
                        if ref_mode == "incremental":
                            iu, iv = None, None
                        else:
                            if prev_u_map is not None:
                                # prev_u_map には「純粋な変位（累積アライメント除き）」が保存されている。
                                # 今回の累積アライメントを加算して探索初期値を構成する。
                                # NaN → 0 に置換してクラッシュ防止（int(round(nan)) エラー対策）。
                                iu = np.where(np.isfinite(prev_u_map),
                                              prev_u_map + float(align_cum_dx), 0.0)
                                iv = np.where(np.isfinite(prev_v_map),
                                              prev_v_map + float(align_cum_dy), 0.0)
                            else:
                                iu, iv = None, None
                        map_result = dic_core.run_dic_full(
                            ref_img, def_img,
                            subset_size=subset_size, step=step,
                            search_x=search_x, search_y=search_y, method=method,
                            zncc_threshold=zncc_thr,
                            roi_xyxy=self.cfg.get("roi_xyxy"),
                            n_workers=n_workers,
                            init_u_grid=iu,
                            init_v_grid=iv,
                            icgn=self.cfg.get("icgn", False),
                        )
                        if ref_mode == "fixed":
                            # 累積アライメントを除いた純粋な変位を保存する。
                            _pu = map_result["U"].copy()
                            _pv = map_result["V"].copy()
                            _pu[np.isfinite(_pu)] -= float(align_cum_dx)
                            _pv[np.isfinite(_pv)] -= float(align_cum_dy)
                            prev_u_map = _pu
                            prev_v_map = _pv
                            # 固定参照モードは通常ひずみがそのまま累積ひずみ
                            map_result["cum_exx"] = map_result["exx"]
                            map_result["cum_eyy"] = map_result["eyy"]
                            map_result["cum_exy"] = map_result["exy"]
                        else:
                            # 増分参照モード：差分変位を積算して累積ひずみを計算
                            u_inc = map_result["U"]
                            v_inc = map_result["V"]
                            X_g   = map_result["X"]
                            Y_g   = map_result["Y"]
                            if cum_u_map is None or cum_u_map.shape != u_inc.shape:
                                cum_u_map = np.zeros_like(u_inc)
                                cum_v_map = np.zeros_like(v_inc)
                            mask = np.isfinite(u_inc) & np.isfinite(v_inc)
                            cum_u_map[mask] += u_inc[mask]
                            cum_v_map[mask] += v_inc[mask]
                            cum_s = dic_core.calc_strain_field(X_g, Y_g, cum_u_map, cum_v_map, step)
                            map_result["cum_exx"] = cum_s["exx"]
                            map_result["cum_eyy"] = cum_s["eyy"]
                            map_result["cum_exy"] = cum_s["exy"]
                        result["map"] = map_result
                        # 再解析用に状態を保存（成功時のみ）
                        self._last_def_path    = str(path)
                        self._last_run_dx      = self._run_dx
                        self._last_run_dy      = self._run_dy
                        self._reanalyze_ref_img = ref_img.copy()
                        self._last_frame_meta  = {
                            "frame": frame_idx, "path": path,
                            "save_time": save_time, "elapsed": elapsed,
                        }
                        if ref_mode == "incremental":
                            self._last_u_contrib = map_result["U"].copy()
                            self._last_v_contrib = map_result["V"].copy()
                    except Exception as e:
                        self.sig_error.emit(f"マップDICエラー: {e}")

                # incremental の場合は参照を更新
                if ref_mode == "incremental":
                    if mode == "map":
                        # マップモード：累積アライメント補正済みの def_img を次フレームの参照に使う。
                        # これにより ref_img が常に image0 座標系に保たれ、
                        # ひずみマップが毎フレーム同じ画素位置（同じ素材点）に対応する。
                        ref_img = def_img
                    else:
                        # 2点間モード：アライメント補正なし、生画像をそのまま参照にする。
                        ref_img = orig_def_img
                    ref_path = path

                prev_path = path   # 次のフレームのアライメント表示用に更新

                self.sig_frame.emit(result)
                self.sig_log.emit(f"Frame {frame_idx}: 完了")

            time.sleep(0.5)

        self.sig_log.emit("監視停止")


# ===========================================================================
# メインウィンドウ
# ===========================================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LiveDIC")
        self.resize(1350, 860)
        self.setStyleSheet(DARK)

        # 状態
        self._worker: Optional[DICWorker] = None
        self._gauges: List[Dict] = []   # {cx1,cy1,cx2,cy2,comp,name,color}
        self._gauge_click_step = 0      # 0=待機 1=点1選択中 2=点2選択中
        self._gauge_draft: Dict = {}
        self._roi_xyxy: Optional[Tuple] = None
        self._align_dx: int = 0   # 確定済みアライメントオフセット
        self._align_dy: int = 0
        self._ref_pixmap: Optional[QPixmap] = None
        self._ref_gray: Optional[np.ndarray] = None
        self._map_frames: List[Dict] = []   # マップモード用フレームバッファ
        self._frame_rows: List[Dict] = []   # フレームテーブル行データ
        self._graph_x_mode = "elapsed"      # "elapsed"|"load"|"stress"
        self._graph_strain_comp = "exx"
        self._pending_reanalyze   = False
        self._reanalyze_ref_path: Optional[str] = None
        self._reanalyze_def_path: Optional[str] = None
        self._latest_img_path:    Optional[str] = None

        # 色リスト（ゲージ色）
        self._colors = ["#4fc3f7", "#ff8a65", "#81c784", "#ce93d8",
                        "#fff176", "#ef9a9a", "#80cbc4", "#ffcc02"]

        # モード別探索範囲の保存値（タブ切替時に復元）
        self._search_defaults = {
            0: {"x": 60, "y": 10},   # 2点間モード
            1: {"x": 20, "y": 20},   # マップモード
        }

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(6)
        root.setContentsMargins(8, 8, 8, 8)

        # 右側：タブ + フレームテーブルを縦に並べる（フレームテーブルはリアルタイムタブでは隠す）
        self._right_vsplit = QSplitter(Qt.Orientation.Vertical)
        self._right_vsplit.addWidget(self._build_tabs())
        self._frame_table_widget = self._build_frame_table()
        self._right_vsplit.addWidget(self._frame_table_widget)
        self._right_vsplit.setSizes([580, 240])
        self._right_vsplit_sizes = [580, 240]
        right_vsplit = self._right_vsplit

        # 最上位：設定パネル（全高）| 右側ペイン
        main_hsplit = QSplitter(Qt.Orientation.Horizontal)
        main_hsplit.addWidget(self._build_settings())
        main_hsplit.addWidget(right_vsplit)
        main_hsplit.setSizes([280, 1050])

        root.addWidget(main_hsplit)
        self._rt_panel.bind_host()
        self._init_settings_store()

        self.statusBar().showMessage("待機中 — フォルダを選択して ▶ Start を押してください")
        self.statusBar().setStyleSheet("background:#252525;color:#777;")

    # ===================================================================
    # 設定の保存と復元（~/.livedic/settings.json）
    # ===================================================================

    def _init_settings_store(self):
        """全ウィジェットが揃ってから呼ぶ。登録 → 前回値を復元。"""
        self.settings = settings_store.SettingsStore()
        self.settings.add_many(
            # 左パネル（2点間・マップ・リアルタイムで共通のもの）
            subset=self.sp_subset, step=self.sp_step, icgn=self.chk_icgn,
            search_x=self.sp_search_x, search_y=self.sp_search_y,
            zncc=self.sp_zncc, subpixel=self.cb_subpixel, workers=self.sp_workers,
            sort=self.cb_sort, ref_fixed=self.rb_ref_fixed,
            manual_align=self.chk_manual_align, align_alpha=self.sl_align_alpha,
            auto_scale=self.chk_auto_scale, vmin=self.sp_vmin, vmax=self.sp_vmax,
            width_mm=self.sp_width, thick_mm=self.sp_thick,
            xaxis=self.cb_xaxis, yaxis=self.cb_yaxis,
        )
        self._rt_panel.register_settings(self.settings)
        # ウィンドウの大きさと分割位置も覚える
        self.settings.add_custom("win_geom", lambda: [self.x(), self.y(), self.width(), self.height()],
                                 lambda v: self.setGeometry(*[int(x) for x in v]))
        self.settings.add_custom("split_right", lambda: self._right_vsplit_sizes,
                                 lambda v: self._restore_right_split([int(x) for x in v]))
        try:
            n = self.settings.restore()
            if n:
                self._append_log(f"前回の設定を復元しました（{n} 項目）")
        except Exception as e:
            self._append_log(f"[WARN] 設定の復元に失敗: {e}")

    def _restore_right_split(self, sizes):
        self._right_vsplit_sizes = sizes
        if self._frame_table_widget.isVisible():
            self._right_vsplit.setSizes(sizes)

    def _save_settings(self):
        try:
            if self._right_vsplit.sizes()[1] > 0:      # テーブルが見えているときの比率を保つ
                self._right_vsplit_sizes = self._right_vsplit.sizes()
            self.settings.save()
        except Exception:
            pass

    # ===================================================================
    # 設定パネル
    # ===================================================================

    def _build_settings(self) -> QWidget:
        panel = QWidget()
        panel.setMaximumWidth(295)
        lay = QVBoxLayout(panel)
        lay.setSpacing(8)

        # ---- 入力設定 ----
        grp = QGroupBox("入力設定")
        form = QFormLayout(grp)
        form.setSpacing(6)

        row_folder = QHBoxLayout()
        self.edit_folder = QLineEdit()
        self.edit_folder.setPlaceholderText("画像フォルダを選択...")
        btn_browse = QPushButton("…")
        btn_browse.setFixedWidth(28)
        btn_browse.clicked.connect(self._browse_folder)
        row_folder.addWidget(self.edit_folder)
        row_folder.addWidget(btn_browse)
        form.addRow("画像フォルダ:", row_folder)

        self.cb_sort = QComboBox()
        self.cb_sort.addItems(["保存時刻順", "ファイル名順"])
        form.addRow("ソート順:", self.cb_sort)

        ref_w = QWidget()
        ref_lay = QVBoxLayout(ref_w); ref_lay.setContentsMargins(0, 0, 0, 0); ref_lay.setSpacing(2)
        self.rb_ref_incr  = QRadioButton("更新（前フレーム）")
        self.rb_ref_fixed = QRadioButton("固定（最初の1枚）")
        self.rb_ref_incr.setChecked(True)
        ref_lay.addWidget(self.rb_ref_incr)
        ref_lay.addWidget(self.rb_ref_fixed)
        form.addRow("参照画像:", ref_w)

        lay.addWidget(grp)

        # ---- DICパラメータ ----
        grp2 = QGroupBox("DICパラメータ")
        form2 = QFormLayout(grp2)
        form2.setSpacing(6)

        self.sp_subset = QSpinBox()
        self.sp_subset.setRange(7, 999); self.sp_subset.setSingleStep(2); self.sp_subset.setValue(31)
        form2.addRow("サブセット [px]:", self.sp_subset)

        self.sp_step = QSpinBox()
        self.sp_step.setRange(4, 999); self.sp_step.setValue(20)
        form2.addRow("ステップ [px]:", self.sp_step)

        search_row = QHBoxLayout()
        self.sp_search_x = QSpinBox()
        self.sp_search_x.setRange(1, 999); self.sp_search_x.setValue(60)
        self.sp_search_y = QSpinBox()
        self.sp_search_y.setRange(1, 999); self.sp_search_y.setValue(10)
        search_row.addWidget(QLabel("X:"))
        search_row.addWidget(self.sp_search_x)
        search_row.addWidget(QLabel("Y:"))
        search_row.addWidget(self.sp_search_y)
        form2.addRow("探索範囲 [±px]:", search_row)

        self.sp_zncc = QDoubleSpinBox()
        self.sp_zncc.setRange(0.0, 1.0); self.sp_zncc.setSingleStep(0.05)
        self.sp_zncc.setValue(0.2); self.sp_zncc.setDecimals(2)
        form2.addRow("ZNCC閾値:", self.sp_zncc)

        self.cb_subpixel = QComboBox()
        self.cb_subpixel.addItems(["parabolic", "gaussian", "spline"])
        form2.addRow("サブピクセル:", self.cb_subpixel)

        self.chk_icgn = QCheckBox("IC-GN 精密化")
        self.chk_icgn.setChecked(True)
        self.chk_icgn.setToolTip("整数探索＋放物線フィットの結果を初期値に、1次形状関数＋B-spline補間の\n"
                                 "IC-GN 反復でサブピクセル変位を求め直す（ピクセルロッキングを1桁以上低減）。\n"
                                 "マップモードとリアルタイムマップで有効。計算時間は増える")
        form2.addRow("", self.chk_icgn)

        self.sp_workers = QSpinBox()
        self.sp_workers.setRange(-1, 32)
        self.sp_workers.setValue(max(1, (os.cpu_count() or 2) - 1))
        self.sp_workers.setSpecialValueText("全コア (-1)")
        self._workers_row = form2.addRow("コア数:", self.sp_workers)
        if getattr(sys, 'frozen', False):
            # exe では joblib 並列が使えないため非表示
            form2.labelForField(self.sp_workers).setVisible(False)
            self.sp_workers.setVisible(False)

        lay.addWidget(grp2)

        # ---- 荷重・応力 ----
        grp3 = QGroupBox("荷重・応力")
        form3 = QFormLayout(grp3)
        form3.setSpacing(6)
        self.sp_width = QDoubleSpinBox()
        self.sp_width.setRange(0.001, 999999); self.sp_width.setValue(1.0)
        self.sp_width.setDecimals(3); self.sp_width.setSuffix(" mm")
        self.sp_width.valueChanged.connect(self._update_area_label)
        form3.addRow("幅:", self.sp_width)
        self.sp_thick = QDoubleSpinBox()
        self.sp_thick.setRange(0.001, 999999); self.sp_thick.setValue(1.0)
        self.sp_thick.setDecimals(3); self.sp_thick.setSuffix(" mm")
        self.sp_thick.valueChanged.connect(self._update_area_label)
        form3.addRow("厚さ:", self.sp_thick)
        self.lbl_area = QLabel("断面積: 1.000 mm²")
        self.lbl_area.setStyleSheet("color:#aaa;font-size:9px;")
        form3.addRow(self.lbl_area)
        lay.addWidget(grp3)

        # ---- Start / Stop (2点間モード用) ----
        self._start_stop_widget = QWidget()
        ss_lay = QVBoxLayout(self._start_stop_widget)
        ss_lay.setContentsMargins(0, 0, 0, 0)
        ss_lay.setSpacing(4)
        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("▶  Start")
        self.btn_start.setObjectName("btn_start")
        self.btn_start.setMinimumHeight(38)
        self.btn_start.clicked.connect(self._on_start)
        self.btn_stop = QPushButton("■  Stop")
        self.btn_stop.setObjectName("btn_stop")
        self.btn_stop.setMinimumHeight(38)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._on_stop)
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_stop)
        ss_lay.addLayout(btn_row)
        self.btn_reset = QPushButton("↺  Reset")
        self.btn_reset.setMinimumHeight(28)
        self.btn_reset.clicked.connect(self._on_reset)
        ss_lay.addWidget(self.btn_reset)
        lay.addWidget(self._start_stop_widget)

        # ---- ROIボタン (マップモード用、タブ切替で表示) ----
        self._roi_btn_widget = QWidget()
        roi_lay = QVBoxLayout(self._roi_btn_widget)
        roi_lay.setContentsMargins(0, 0, 0, 0)
        roi_lay.setSpacing(3)
        roi_row = QHBoxLayout()
        self.btn_roi_set = QPushButton("ROI設定（ドラッグ）")
        self.btn_roi_set.setCheckable(True)
        self.btn_roi_set.toggled.connect(self._on_roi_btn_toggled)
        self.btn_roi_clear = QPushButton("ROIクリア")
        self.btn_roi_clear.clicked.connect(self._on_roi_clear)
        roi_row.addWidget(self.btn_roi_set)
        roi_row.addWidget(self.btn_roi_clear)
        roi_lay.addLayout(roi_row)
        self.lbl_roi = QLabel("ROI: 全域")
        self.lbl_roi.setStyleSheet("color:#888;font-size:9px;")
        roi_lay.addWidget(self.lbl_roi)
        self._roi_btn_widget.setVisible(False)
        lay.addWidget(self._roi_btn_widget)

        # ---- アライメント確定ボタン (アライメント画面表示中のみ) ----
        self._align_confirm_widget = QWidget()
        ac_lay = QVBoxLayout(self._align_confirm_widget)
        ac_lay.setContentsMargins(0, 0, 0, 0)
        ac_lay.setSpacing(3)
        self.btn_align_ok = QPushButton("✔  アライメント確定")
        self.btn_align_ok.setObjectName("btn_start")
        self.btn_align_ok.setMinimumHeight(30)
        self.btn_align_ok.clicked.connect(self._on_align_confirm)
        ac_lay.addWidget(self.btn_align_ok)
        self._align_confirm_widget.setVisible(False)
        lay.addWidget(self._align_confirm_widget)

        # ---- DIC実行ボタン (マップモード用) ----
        self.btn_dic_run = QPushButton("▶  DIC実行")
        self.btn_dic_run.setObjectName("btn_start")
        self.btn_dic_run.setMinimumHeight(38)
        self.btn_dic_run.setEnabled(False)
        self.btn_dic_run.clicked.connect(self._on_dic_run)
        self.btn_dic_run.setVisible(False)  # 初期は2点間タブなので非表示
        lay.addWidget(self.btn_dic_run)

        self.btn_reanalyze = QPushButton("↺  最新フレームを再解析")
        self.btn_reanalyze.setMinimumHeight(30)
        self.btn_reanalyze.setEnabled(False)
        self.btn_reanalyze.clicked.connect(self._on_reanalyze_request)
        self.btn_reanalyze.setVisible(False)
        lay.addWidget(self.btn_reanalyze)

        self.btn_quality = QPushButton("🔍  品質評価")
        self.btn_quality.setMinimumHeight(30)
        self.btn_quality.setEnabled(False)
        self.btn_quality.clicked.connect(self._on_quality_check)
        self.btn_quality.setVisible(False)
        lay.addWidget(self.btn_quality)

        # ---- ログ（残りスペースを全部使う） ----
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setStyleSheet("background:#141414;color:#888;font-size:9px;")
        lay.addWidget(self.log_box, stretch=1)

        return panel

    # ===================================================================
    # タブエリア
    # ===================================================================

    def _build_tabs(self) -> QTabWidget:
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_twopoint_tab(), "2点間モード")
        self.tabs.addTab(self._build_map_tab(),      "マップモード")
        self._rt_panel = rt_panel.RTMapPanel(self, ImageCanvas)
        self.tabs.addTab(self._rt_panel,             "リアルタイムマップ")
        self._cur_tab = 0
        self.tabs.currentChanged.connect(self._on_tab_changed)
        return self.tabs

    # ---- 2点間タブ ----
    def _build_twopoint_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(4)

        # ── 上段：左（画像）| 右（グラフ）のスプリッター ──
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # 左：参照画像
        left = QWidget()
        ll = QVBoxLayout(left)

        self.ref_canvas = ImageCanvas()
        self.ref_canvas.signal_clicked.connect(self._on_canvas_click)
        ll.addWidget(self.ref_canvas, stretch=1)

        btn_row = QHBoxLayout()
        self.btn_add_gauge = QPushButton("ゲージ追加（2点クリック）")
        self.btn_add_gauge.clicked.connect(self._start_gauge_add)
        self.btn_subset_win = QPushButton("サブセット画像を表示")
        self.btn_subset_win.setCheckable(True)
        self.btn_subset_win.toggled.connect(self._on_subset_win_toggle)
        btn_row.addWidget(self.btn_add_gauge)
        btn_row.addWidget(self.btn_subset_win)
        ll.addLayout(btn_row)
        self.lbl_click_hint = QLabel("")
        self.lbl_click_hint.setStyleSheet("color:#aaa;font-size:9px;")
        ll.addWidget(self.lbl_click_hint)

        # ゲージ一覧
        grp_g = QGroupBox("登録ゲージ")
        gg = QVBoxLayout(grp_g)
        self.gauge_table = QTableWidget(0, 5)
        self.gauge_table.setHorizontalHeaderLabels(
            ["名称", "点1 (x,y)", "点2 (x,y)", "成分", "削除"])
        self.gauge_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self.gauge_table.setMaximumHeight(110)
        gg.addWidget(self.gauge_table)
        ll.addWidget(grp_g)

        splitter.addWidget(left)

        # 右：グラフのみ（サブセットパネルは下段に移動）
        right = QWidget()
        rl = QVBoxLayout(right)
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("X軸:"))
        self.cb_xaxis = QComboBox()
        self.cb_xaxis.addItems(["経過時間 [s]", "荷重 [N]", "応力 [MPa]"])
        self.cb_xaxis.currentIndexChanged.connect(self._on_xaxis_change)
        ctrl.addWidget(self.cb_xaxis)
        ctrl.addSpacing(16)
        ctrl.addWidget(QLabel("Y軸:"))
        self.cb_yaxis = QComboBox()
        self.cb_yaxis.addItems(["εxx", "εyy", "εxy"])
        self.cb_yaxis.currentIndexChanged.connect(self._on_yaxis_change)
        ctrl.addWidget(self.cb_yaxis)
        ctrl.addStretch()
        rl.addLayout(ctrl)

        self.graph = LiveGraph()
        rl.addWidget(self.graph, stretch=1)

        splitter.addWidget(right)
        splitter.setSizes([480, 560])
        lay.addWidget(splitter, stretch=1)

        # サブセットウィンドウ（別ウィンドウ）
        self._subset_win = SubsetWindow()

        return w

    # ---- マップタブ ----
    def _build_map_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)

        # ---- 上段コントロール ----
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("ひずみ成分:"))
        self._map_comp_bg = QButtonGroup(w)
        for i, key in enumerate(STRAIN_KEYS):
            rb = QRadioButton(STRAIN_LABELS[key])
            if i == 0: rb.setChecked(True)
            rb.toggled.connect(lambda checked, k=key: self._on_map_comp(k, checked))
            self._map_comp_bg.addButton(rb)
            ctrl.addWidget(rb)
        ctrl.addSpacing(20)
        ctrl.addWidget(QLabel("Min:"))
        self.sp_vmin = QDoubleSpinBox()
        self.sp_vmin.setRange(-999999, 999999)
        self.sp_vmin.setDecimals(4); self.sp_vmin.setValue(-0.01); self.sp_vmin.setFixedWidth(80)
        ctrl.addWidget(self.sp_vmin)
        ctrl.addWidget(QLabel("Max:"))
        self.sp_vmax = QDoubleSpinBox()
        self.sp_vmax.setRange(-999999, 999999)
        self.sp_vmax.setDecimals(4); self.sp_vmax.setValue(0.02); self.sp_vmax.setFixedWidth(80)
        ctrl.addWidget(self.sp_vmax)
        self.chk_auto_scale = QCheckBox("自動カラースケール")
        self.chk_auto_scale.setChecked(True)
        ctrl.addWidget(self.chk_auto_scale)
        ctrl.addSpacing(20)
        ctrl.addWidget(QLabel("表示:"))
        self._map_disp_bg = QButtonGroup(w)
        self._rb_map_cum = QRadioButton("累積")
        self._rb_map_inc = QRadioButton("差分")
        self._rb_map_cum.setChecked(True)
        self._rb_map_cum.toggled.connect(lambda checked: self._refresh_map_display() if checked else None)
        self._rb_map_inc.toggled.connect(lambda checked: self._refresh_map_display() if checked else None)
        self._map_disp_bg.addButton(self._rb_map_cum)
        self._map_disp_bg.addButton(self._rb_map_inc)
        ctrl.addWidget(self._rb_map_cum)
        ctrl.addWidget(self._rb_map_inc)
        ctrl.addSpacing(20)
        self.chk_manual_align = QCheckBox("手動アライメント")
        self.chk_manual_align.setChecked(True)
        self.chk_manual_align.toggled.connect(self._on_manual_align_toggled)
        ctrl.addWidget(self.chk_manual_align)
        ctrl.addStretch()
        lay.addLayout(ctrl)

        # ---- スタックウィジェット：通常表示 / アライメント表示 ----
        self._map_stack = QStackedWidget()

        # ページ0：通常マップ表示
        map_page = QWidget()
        mp_lay = QVBoxLayout(map_page)
        mp_lay.setContentsMargins(0, 0, 0, 0)
        self.map_canvas = ImageCanvas()
        self.map_canvas.signal_roi.connect(self._on_map_roi)
        mp_lay.addWidget(self.map_canvas, stretch=1)
        sl_row = QHBoxLayout()
        sl_row.addWidget(QLabel("フレーム:"))
        self.map_slider = QSlider(Qt.Orientation.Horizontal)
        self.map_slider.setRange(0, 0)
        self.map_slider.valueChanged.connect(self._on_map_slider)
        sl_row.addWidget(self.map_slider)
        self.lbl_map_frame = QLabel("—")
        self.lbl_map_frame.setStyleSheet("color:#888;font-size:9px;")
        sl_row.addWidget(self.lbl_map_frame)
        mp_lay.addLayout(sl_row)
        self._map_stack.addWidget(map_page)

        # ページ1：手動アライメント表示
        align_page = QWidget()
        al_lay = QVBoxLayout(align_page)
        al_lay.setContentsMargins(0, 0, 0, 0)
        lbl_hint = QLabel("変形画像（半透明）をドラッグして参照画像と合わせ、「DIC実行」を押してください")
        lbl_hint.setStyleSheet("color:#aaa;font-size:9px;padding:2px;")
        al_lay.addWidget(lbl_hint)
        self.align_canvas = AlignmentCanvas()
        self.align_canvas.sig_offset_changed.connect(self._on_align_offset_changed)
        al_lay.addWidget(self.align_canvas, stretch=1)
        # 行1：オフセット表示 + 不透明度スライダー
        al_ctrl = QHBoxLayout()
        self.lbl_align_offset = QLabel("dx=0  dy=0")
        self.lbl_align_offset.setStyleSheet("color:#4fc3f7;font-size:10px;")
        al_ctrl.addWidget(self.lbl_align_offset)
        al_ctrl.addStretch()
        al_ctrl.addWidget(QLabel("不透明度:"))
        self.sl_align_alpha = QSlider(Qt.Orientation.Horizontal)
        self.sl_align_alpha.setRange(0, 100)
        self.sl_align_alpha.setValue(50)
        self.sl_align_alpha.setFixedWidth(100)
        self.sl_align_alpha.valueChanged.connect(
            lambda v: self.align_canvas.set_alpha(v / 100))
        al_ctrl.addWidget(self.sl_align_alpha)
        al_lay.addLayout(al_ctrl)
        self._map_stack.addWidget(align_page)

        lay.addWidget(self._map_stack, stretch=1)

        self._map_comp = "exx"
        return w

    # ===================================================================
    # フレームテーブル
    # ===================================================================

    def _build_frame_table(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(4)

        hdr = QHBoxLayout()
        lbl = QLabel("フレームテーブル")
        lbl.setStyleSheet("font-weight:bold;font-size:11px;")
        hdr.addWidget(lbl)
        hdr.addStretch()
        btn_csv = QPushButton("CSV出力")
        btn_csv.setFixedWidth(80)
        btn_csv.clicked.connect(self._export_csv)
        hdr.addWidget(btn_csv)
        lay.addLayout(hdr)

        self._tbl_fixed_cols = ["フレーム", "ファイル名", "保存時刻", "経過時間 [s]", "MIG"]
        self._tbl_load_cols  = ["荷重 [N]", "応力 [MPa]"]
        self.frame_table = QTableWidget(0, len(self._tbl_fixed_cols) + len(self._tbl_load_cols))
        self._update_table_headers()
        self.frame_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents)
        self.frame_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch)
        self.frame_table.setAlternatingRowColors(True)
        self.frame_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self.frame_table.itemChanged.connect(self._on_table_item_changed)
        lay.addWidget(self.frame_table)
        return w

    def _update_table_headers(self):
        gauge_cols = []
        for i, g in enumerate(self._gauges):
            gauge_cols.append(f"ゲージ{i+1} {g.get('comp','exx')}")
            gauge_cols.append(f"ゲージ{i+1} ZNCC1")
            gauge_cols.append(f"ゲージ{i+1} ZNCC2")
        cols = self._tbl_fixed_cols + gauge_cols + self._tbl_load_cols
        self.frame_table.setColumnCount(len(cols))
        self.frame_table.setHorizontalHeaderLabels(cols)
        self._gauge_col_offset = len(self._tbl_fixed_cols)
        self._load_col_offset  = self._gauge_col_offset + len(gauge_cols)

    # ===================================================================
    # イベントハンドラ
    # ===================================================================

    def _on_tab_changed(self, idx: int):
        """タブ切替時：探索範囲を保存・復元。マップタブへの切替はフォルダがあれば自動監視開始。"""
        prev = self._cur_tab
        self._cur_tab = idx
        if prev in self._search_defaults:
            self._search_defaults[prev]["x"] = self.sp_search_x.value()
            self._search_defaults[prev]["y"] = self.sp_search_y.value()
        if idx in self._search_defaults:
            self.sp_search_x.setValue(self._search_defaults[idx]["x"])
            self.sp_search_y.setValue(self._search_defaults[idx]["y"])
        # フレームテーブルはフォルダ監視モード（2点間・マップ）専用。リアルタイムタブでは隠して
        # 画像表示に高さを回す。分割位置は隠す前のものを覚えておき、戻ったときに復元する
        if idx == 2 and self._frame_table_widget.isVisible():
            self._right_vsplit_sizes = self._right_vsplit.sizes()
            self._frame_table_widget.setVisible(False)
        elif idx != 2 and not self._frame_table_widget.isVisible():
            self._frame_table_widget.setVisible(True)
            self._right_vsplit.setSizes(self._right_vsplit_sizes)
        if idx == 2:  # リアルタイムマップタブ：操作はタブ内で完結。左パネルのモード別ボタンは全て隠す
            self._start_stop_widget.setVisible(False)
            self.btn_dic_run.setVisible(False)
            self.btn_reanalyze.setVisible(False)
            self.btn_quality.setVisible(False)
            self._roi_btn_widget.setVisible(False)
            self._align_confirm_widget.setVisible(False)
        elif idx == 1:  # マップタブ
            self._start_stop_widget.setVisible(False)
            self.btn_dic_run.setVisible(True)
            self.btn_reanalyze.setVisible(True)
            self.btn_quality.setVisible(True)
            self._roi_btn_widget.setVisible(True)
            self._restart_worker()
        else:  # 2点間タブ
            self._start_stop_widget.setVisible(True)
            self.btn_dic_run.setVisible(False)
            self.btn_reanalyze.setVisible(False)
            self.btn_quality.setVisible(False)
            self._roi_btn_widget.setVisible(False)
            self._align_confirm_widget.setVisible(False)

    def _on_new_image(self, prev_path_str: str, def_path_str: str):
        """ワーカーが新画像を検出 → アライメント画面表示またはDIC実行待ちを通知。"""
        self._align_dx, self._align_dy = 0, 0   # 新画像ごとにオフセットをリセット
        self._reanalyze_ref_path = prev_path_str
        self._reanalyze_def_path = def_path_str
        self._latest_img_path = def_path_str
        self.btn_quality.setEnabled(True)
        self.btn_dic_run.setEnabled(True)
        def_name = pathlib.Path(def_path_str).name
        if self.chk_manual_align.isChecked():
            try:
                prev_g = dic_core.imread_safe(prev_path_str)
                def_g  = dic_core.imread_safe(def_path_str)
            except Exception as e:
                self._append_log(f"[ERROR] アライメント画像読み込みエラー: {e}")
                self._on_align_confirm_skip()  # 補正なしで確定してpage 0に戻る
                return
            self.align_canvas.set_images(prev_g, def_g)
            self.lbl_align_offset.setText("dx=0  dy=0")
            self._map_stack.setCurrentIndex(1)
            self._align_confirm_widget.setVisible(True)
            self.statusBar().showMessage(
                f"新しい画像: {def_name} — アライメント後「▶ DIC実行」を押してください")
        else:
            self._map_stack.setCurrentIndex(0)
            self._align_confirm_widget.setVisible(False)
            self.statusBar().showMessage(
                f"新しい画像: {def_name} — 「▶ DIC実行」を押してください")

    def _on_align_confirm(self):
        """✔ アライメント確定 → オフセットを保存してmap pageに戻る（DICはまだ実行しない）。"""
        dx, dy = self.align_canvas.get_offset()
        self._align_dx, self._align_dy = dx, dy
        self._map_stack.setCurrentIndex(0)
        self._align_confirm_widget.setVisible(False)
        self.statusBar().showMessage(
            f"アライメント確定: dx={dx:+d}  dy={dy:+d}  — 「▶ DIC実行」を押すと計算します")

    def _on_align_confirm_skip(self):
        """補正なし（確定） → オフセット 0 で確定してmap pageに戻る。"""
        self._align_dx, self._align_dy = 0, 0
        self._map_stack.setCurrentIndex(0)
        self._align_confirm_widget.setVisible(False)
        self.statusBar().showMessage("アライメント補正なし（確定）— 「▶ DIC実行」を押すと計算します")

    def _on_reanalyze_request(self):
        """「最新フレームを再解析」ボタン → 同じ画像をアライメント調整して再計算。"""
        if not self._worker or not self._map_frames:
            return
        if self.btn_dic_run.isEnabled():
            self.statusBar().showMessage("新しい画像が待機中です。先にDIC実行してください")
            return
        self._pending_reanalyze = True
        self._align_dx, self._align_dy = 0, 0
        self.btn_dic_run.setEnabled(True)
        if (self.chk_manual_align.isChecked()
                and self._reanalyze_ref_path and self._reanalyze_def_path):
            try:
                prev_g = dic_core.imread_safe(self._reanalyze_ref_path)
                def_g  = dic_core.imread_safe(self._reanalyze_def_path)
            except Exception as e:
                self._append_log(f"[ERROR] 再解析画像読み込みエラー: {e}")
                self._pending_reanalyze = False
                self.btn_dic_run.setEnabled(False)
                return
            self.align_canvas.set_images(prev_g, def_g)
            self.lbl_align_offset.setText("dx=0  dy=0")
            self._map_stack.setCurrentIndex(1)
            self._align_confirm_widget.setVisible(True)
            self.statusBar().showMessage(
                "再解析 — アライメントを調整して「▶ DIC実行」を押してください")
        else:
            self._map_stack.setCurrentIndex(0)
            self.statusBar().showMessage("再解析 — 「▶ DIC実行」を押してください")

    def _on_dic_run(self):
        """「▶ DIC実行」ボタン → 確定済みオフセットでDIC計算開始。"""
        if not self._worker:
            return
        self.btn_dic_run.setEnabled(False)
        if self._pending_reanalyze:
            self._pending_reanalyze = False
            self._worker.trigger_reanalyze(self._align_dx, self._align_dy)
        else:
            self._worker.trigger_dic(self._align_dx, self._align_dy)

    def _on_quality_check(self):
        """「品質評価」ボタン → SSSIG・スペックルサイズをダイアログ表示、MIGをテーブルに記録。"""
        if not self._latest_img_path:
            self.statusBar().showMessage("品質評価: 評価対象画像がありません")
            return
        try:
            img = dic_core.imread_safe(self._latest_img_path)
        except Exception as e:
            self.statusBar().showMessage(f"品質評価: 画像読み込みエラー — {e}")
            return
        self._show_quality(img, self._roi_xyxy, record_mig=True)

    def _show_quality(self, img: np.ndarray, roi_xyxy, record_mig: bool = False):
        """画像配列に対して品質評価ダイアログを表示する（マップモード・リアルタイムモード共用）。"""
        subset_size = self.sp_subset.value()
        step        = self.sp_step.value()
        search_x    = self.sp_search_x.value()
        search_y    = self.sp_search_y.value()

        # MIG（高速）
        mig = dic_core.calc_mig(img)

        # グリッド生成
        try:
            X, Y = dic_core.create_grid(
                img.shape[:2], subset_size,
                max(search_x, search_y), step,
                roi_xyxy,
            )
        except Exception as e:
            self.statusBar().showMessage(f"品質評価: グリッド生成エラー — {e}")
            return

        # SSSIG マップ（グリッド点ごとの勾配二乗和）
        sssig = dic_core.calc_sssig_map(img, X, Y, subset_size)

        # スペックルサイズ 3×3（ACF ハーフ幅）
        speckle_sizes = dic_core.calc_speckle_sizes_grid(
            img, roi_xyxy=roi_xyxy, n_div=3)

        # ダイアログ表示
        dlg = QualityDialog(
            mig, sssig, X, Y, img.shape,
            speckle_sizes, roi_xyxy, subset_size, self)
        dlg.show()
        dlg.raise_()

        # MIG をフレームテーブルの最新行に記録（マップモードのみ）
        if record_mig:
            self._update_mig_in_table(mig)

        self.statusBar().showMessage(
            f"品質評価完了 — MIG = {mig:.4f}  "
            f"スペックルサイズ（中央）= {float(speckle_sizes[1, 1]):.1f} px")

    def _update_mig_in_table(self, mig: float):
        """最新フレーム行の MIG セル（列4）を更新する。"""
        r = self.frame_table.rowCount() - 1
        if r < 0:
            return
        if r < len(self._frame_rows):
            self._frame_rows[r]["mig"] = mig
        self.frame_table.blockSignals(True)
        it = QTableWidgetItem(f"{mig:.4f}")
        it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.frame_table.setItem(r, 4, it)
        self.frame_table.blockSignals(False)

    def _on_manual_align_toggled(self, checked: bool):
        if not checked:
            self._map_stack.setCurrentIndex(0)
            self._align_confirm_widget.setVisible(False)

    def _on_roi_btn_toggled(self, checked: bool):
        self.map_canvas.set_mode("roi" if checked else "none")

    def _on_map_roi(self, x1: int, y1: int, x2: int, y2: int):
        self._roi_xyxy = (x1, y1, x2, y2)
        if self._worker:
            self._worker.cfg["roi_xyxy"] = self._roi_xyxy  # 実行中ワーカーにも即時反映
        self.lbl_roi.setText(f"ROI: ({x1},{y1})–({x2},{y2})")
        self.btn_roi_set.setChecked(False)
        # ROI矩形をオーバーレイ表示
        self.map_canvas.clear_overlays()
        self.map_canvas.add_overlay({"type": "rect",
                                     "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                                     "color": "#ffaa00"})

    def _on_roi_clear(self):
        self._roi_xyxy = None
        if self._worker:
            self._worker.cfg["roi_xyxy"] = None  # 実行中ワーカーにも即時反映
        self.lbl_roi.setText("ROI: 全域")
        self.map_canvas.clear_overlays()
        if self._ref_pixmap:

            self.map_canvas.set_image(self._ref_pixmap)

    def _on_align_offset_changed(self, dx: int, dy: int):
        self.lbl_align_offset.setText(f"dx={dx:+d}  dy={dy:+d}")

    def _update_area_label(self):
        area = self.sp_width.value() * self.sp_thick.value()
        self.lbl_area.setText(f"断面積: {area:.3f} mm²")

    def _browse_folder(self):
        d = QFileDialog.getExistingDirectory(self, "画像フォルダを選択")
        if d:
            self.edit_folder.setText(d)
            if self.tabs.currentIndex() == 1:  # マップモードのみ自動監視開始
                self._restart_worker()
            else:  # 2点間モードはフォルダ読込のみ（Startは手動）
                self._load_existing_images(pathlib.Path(d))

    def _load_existing_images(self, folder: pathlib.Path):
        """フォルダ選択時に先頭1枚を参照として表示。"""
        sort_mode = "mtime" if self.cb_sort.currentIndex() == 0 else "name"
        imgs = list_images_sorted(folder, sort_mode)
        if not imgs:
            return
        try:
            gray = dic_core.imread_safe(str(imgs[0]))
            self._ref_gray = gray
            self._ref_pixmap = ndarray_to_qpixmap(gray)
            self.ref_canvas.set_image(self._ref_pixmap)

            self.map_canvas.set_image(self._ref_pixmap)
            self.statusBar().showMessage(f"参照画像読み込み済: {imgs[0].name}")
        except Exception as e:
            self.statusBar().showMessage(f"画像読み込みエラー: {e}")

    def _restart_worker(self):
        """既存ワーカーを停止し、現在の設定で監視を再開する。"""
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(3000)
            self._worker = None
        folder = self.edit_folder.text().strip()
        if not folder or not pathlib.Path(folder).is_dir():
            return
        self._load_existing_images(pathlib.Path(folder))
        self._on_start()

    def _start_gauge_add(self):
        if self._ref_pixmap is None:
            self.statusBar().showMessage("先に画像フォルダを選択してください")
            return
        self._gauge_click_step = 1
        self._gauge_draft = {}
        self.ref_canvas.set_mode("click")
        self.lbl_click_hint.setText("点1をクリックしてください")

    def _on_canvas_click(self, ix: int, iy: int):
        if self._gauge_click_step == 1:
            self._gauge_draft["cx1"] = ix
            self._gauge_draft["cy1"] = iy
            self._gauge_click_step = 2
            self.lbl_click_hint.setText(f"点1: ({ix},{iy})  →  点2をクリックしてください")
        elif self._gauge_click_step == 2:
            self._gauge_draft["cx2"] = ix
            self._gauge_draft["cy2"] = iy
            self._gauge_click_step = 0
            self.ref_canvas.set_mode("none")
            self.lbl_click_hint.setText("")
            self._add_gauge_from_draft()

    def _add_gauge_from_draft(self):
        gi = len(self._gauges)
        color = self._colors[gi % len(self._colors)]
        comp = STRAIN_KEYS[self.cb_yaxis.currentIndex()]
        g = {**self._gauge_draft, "comp": comp,
             "name": f"ゲージ{gi+1}", "color": color}
        self._gauges.append(g)
        self._redraw_gauge_overlays()
        self._refresh_gauge_table()
        self._update_table_headers()
        self._rebuild_graph()
        self._refresh_subset_from_ref()

    def _refresh_subset_from_ref(self):
        """参照画像から全ゲージのサブセットを切り出してサブセットウィンドウを更新。"""
        if self._ref_gray is None:
            return
        img = self._ref_gray
        sp = self.sp_subset.value()
        half = sp // 2

        def _crop(cx, cy):
            y0 = max(0, cy - half); y1 = min(img.shape[0], cy + half + 1)
            x0 = max(0, cx - half); x1 = min(img.shape[1], cx + half + 1)
            p = img[y0:y1, x0:x1]
            # img は imread_safe が返す uint8 [0,255] なのでそのまま使う
            return np.ascontiguousarray(p).copy()

        gauge_results = []
        for g in self._gauges:
            gauge_results.append({
                "patch1": _crop(g["cx1"], g["cy1"]),
                "patch2": _crop(g["cx2"], g["cy2"]),
                "zncc": 1.0,
            })
        self._subset_win.update_patches(self._gauges, gauge_results)
        self._subset_win.show()
        self.btn_subset_win.setChecked(True)

    def _redraw_gauge_overlays(self):
        self.ref_canvas.clear_overlays()
        if self._roi_xyxy:
            x1, y1, x2, y2 = self._roi_xyxy
            self.ref_canvas.add_overlay({"type": "rect", "x1": x1, "y1": y1,
                                         "x2": x2, "y2": y2, "color": "#ffaa00"})
        for i, g in enumerate(self._gauges):
            color = g["color"]
            self.ref_canvas.add_overlay(
                {"type": "point", "x": g["cx1"], "y": g["cy1"],
                 "color": color, "label": f"{g['name']}-1"})
            self.ref_canvas.add_overlay(
                {"type": "point", "x": g["cx2"], "y": g["cy2"],
                 "color": color, "label": f"{g['name']}-2"})
            self.ref_canvas.add_overlay(
                {"type": "line", "x1": g["cx1"], "y1": g["cy1"],
                 "x2": g["cx2"], "y2": g["cy2"], "color": color})

    def _refresh_gauge_table(self):
        self.gauge_table.blockSignals(True)
        self.gauge_table.setRowCount(len(self._gauges))
        for i, g in enumerate(self._gauges):
            self.gauge_table.setItem(i, 0, QTableWidgetItem(g["name"]))
            self.gauge_table.setItem(i, 1, QTableWidgetItem(f"({g['cx1']},{g['cy1']})"))
            self.gauge_table.setItem(i, 2, QTableWidgetItem(f"({g['cx2']},{g['cy2']})"))
            cb = QComboBox()
            cb.addItems(STRAIN_KEYS)
            cb.setCurrentText(g.get("comp", "exx"))
            cb.currentTextChanged.connect(
                lambda text, idx=i: self._on_gauge_comp_change(idx, text))
            self.gauge_table.setCellWidget(i, 3, cb)
            btn_del = QPushButton("✕")
            btn_del.setFixedWidth(28)
            btn_del.clicked.connect(lambda _, idx=i: self._delete_gauge(idx))
            self.gauge_table.setCellWidget(i, 4, btn_del)
        self.gauge_table.blockSignals(False)

    def _on_gauge_comp_change(self, idx: int, comp: str):
        if idx < len(self._gauges):
            self._gauges[idx]["comp"] = comp
            self._update_table_headers()
            self._rebuild_graph()

    def _delete_gauge(self, idx: int):
        if idx < len(self._gauges):
            self._gauges.pop(idx)
            self._redraw_gauge_overlays()
            self._refresh_gauge_table()
            self._update_table_headers()
            self._rebuild_graph()

    def _rebuild_graph(self):
        self.graph.clear_all()
        for key in list(self.graph._lines.keys()):
            self.graph._lines[key].remove()
        self.graph._lines.clear()
        for g in self._gauges:
            key = g["name"]
            if key not in self.graph._lines:
                self.graph.add_series(key, g["name"], g["color"])
        # 既存フレームデータを再プロット
        for row_data in self._frame_rows:
            x = self._row_xval(row_data)
            for gi, g in enumerate(self._gauges):
                key = g["name"]
                y = row_data.get(f"gauge_{gi}", np.nan)
                if np.isfinite(x) and np.isfinite(y):
                    self.graph.update_series(key, x, y)
        ylabel = STRAIN_LABELS.get(self._graph_strain_comp, self._graph_strain_comp)
        self.graph.set_labels(self._xaxis_label(), ylabel)

    def _xaxis_label(self) -> str:
        return ["経過時間 [s]", "荷重 [N]", "応力 [MPa]"][self.cb_xaxis.currentIndex()]

    def _row_xval(self, row: dict) -> float:
        mode = self.cb_xaxis.currentIndex()
        if mode == 0:
            return float(row.get("elapsed", np.nan))
        elif mode == 1:
            v = row.get("load", "")
            try: return float(v)
            except: return np.nan
        else:
            v = row.get("stress", "")
            try: return float(v)
            except: return np.nan

    def _on_xaxis_change(self, _):
        self._rebuild_graph()

    def _on_yaxis_change(self, idx):
        self._graph_strain_comp = STRAIN_KEYS[idx]
        self._rebuild_graph()

    def _on_map_comp(self, key: str, checked: bool):
        if checked:
            self._map_comp = key
            self._refresh_map_display()

    def _on_map_slider(self, val: int):
        self._refresh_map_display()

    def _refresh_map_display(self):
        idx = self.map_slider.value()
        if idx < 0 or idx >= len(self._map_frames):
            return
        frame = self._map_frames[idx]
        m = frame.get("map")
        if m is None:
            return
        use_cum = self._rb_map_cum.isChecked()
        cum_key = f"cum_{self._map_comp}"
        if use_cum and cum_key in m:
            data = m.get(cum_key)
            disp_label = "累積"
        else:
            data = m.get(self._map_comp)
            disp_label = "差分"
        if data is None:
            return
        vmin = None if self.chk_auto_scale.isChecked() else self.sp_vmin.value()
        vmax = None if self.chk_auto_scale.isChecked() else self.sp_vmax.value()
        X = m.get("X")
        Y = m.get("Y")
        img_shape = self._ref_gray.shape if self._ref_gray is not None else None
        pix = strain_to_qpixmap(data, vmin, vmax, X=X, Y=Y, img_shape=img_shape)
        self.map_canvas.set_image(pix)
        p = frame.get("path")
        elapsed = frame.get("elapsed", 0.0)
        self.lbl_map_frame.setText(
            f"{idx + 1} / {len(self._map_frames)}   |   "
            f"{p.name if p else ''}   |   "
            f"経過: {elapsed:.1f}s   |   {disp_label}")

    # ===================================================================
    # フレーム受信
    # ===================================================================

    def _on_frame_received(self, result: dict):
        frame_idx    = result["frame"]
        path: pathlib.Path = result["path"]
        save_time: datetime = result["save_time"]
        elapsed      = result["elapsed"]
        is_ref       = result["is_ref"]
        is_reanalyze = result.get("is_reanalyze", False)

        # 品質評価用に最新画像パスを更新（参照・変形どちらでも）
        self._latest_img_path = str(path)
        if self.tabs.currentIndex() == 1:
            self.btn_quality.setEnabled(True)

        # 参照画像を表示 + グラフ原点を打つ
        if is_ref:
            try:
                gray = dic_core.imread_safe(str(path))
                self._ref_gray = gray
                self._ref_pixmap = ndarray_to_qpixmap(gray)
                self.ref_canvas.set_image(self._ref_pixmap)
    
                self.map_canvas.set_image(self._ref_pixmap)   # マップタブにも参照画像を表示
                self._redraw_gauge_overlays()
            except Exception:
                pass
            # 登録済みゲージがあれば原点 (0, 0) を打つ
            for g in self._gauges:
                key = g["name"]
                if key not in self.graph._lines:
                    self.graph.add_series(key, g["name"], g["color"])
                self.graph.update_series(key, 0.0, 0.0)

        # フレームテーブル行データを作成
        row_data = {
            "frame": frame_idx,
            "filename": path.name,
            "save_time": save_time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed": elapsed,
            "load": "",
            "stress": "",
        }
        gauge_results = result.get("gauges", [])
        for gi, gr in enumerate(gauge_results):
            row_data[f"gauge_{gi}"]       = gr.get("strain", np.nan)
            row_data[f"gauge_{gi}_zncc1"] = gr.get("zncc1",  np.nan)
            row_data[f"gauge_{gi}_zncc2"] = gr.get("zncc2",  np.nan)

        if is_reanalyze and self._frame_rows:
            # load/stress/mig のユーザー入力値・計算済み値を引き継いで最終行を上書き
            prev = self._frame_rows[-1]
            row_data["load"]   = prev.get("load",   "")
            row_data["stress"] = prev.get("stress", "")
            row_data["mig"]    = prev.get("mig",    "")
            self._frame_rows[-1] = row_data
        else:
            self._frame_rows.append(row_data)

        # テーブルに追加（再解析時は行を追加せず最終行を上書き）
        self.frame_table.blockSignals(True)
        if is_reanalyze and self.frame_table.rowCount() > 0:
            r = self.frame_table.rowCount() - 1
        else:
            r = self.frame_table.rowCount()
            self.frame_table.insertRow(r)
        editable_start = self._load_col_offset

        def _item(text, editable=False):
            it = QTableWidgetItem(str(text) if text is not None else "")
            if not editable:
                it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
            return it

        self.frame_table.setItem(r, 0, _item(frame_idx))
        self.frame_table.setItem(r, 1, _item(path.name))
        self.frame_table.setItem(r, 2, _item(save_time.strftime("%Y-%m-%d %H:%M:%S")))
        self.frame_table.setItem(r, 3, _item(f"{elapsed:.1f}"))
        mig_val = row_data.get("mig", "")
        self.frame_table.setItem(r, 4, _item(f"{mig_val:.4f}" if isinstance(mig_val, float) else ""))

        for gi in range(len(self._gauges)):
            val = row_data.get(f"gauge_{gi}", np.nan)
            text = f"{val:.5f}" if np.isfinite(val) else ""
            self.frame_table.setItem(r, self._gauge_col_offset + gi * 3,     _item(text))
            z1 = row_data.get(f"gauge_{gi}_zncc1", np.nan)
            self.frame_table.setItem(r, self._gauge_col_offset + gi * 3 + 1, _item(f"{z1:.3f}" if np.isfinite(z1) else ""))
            z2 = row_data.get(f"gauge_{gi}_zncc2", np.nan)
            self.frame_table.setItem(r, self._gauge_col_offset + gi * 3 + 2, _item(f"{z2:.3f}" if np.isfinite(z2) else ""))

        self.frame_table.setItem(r, self._load_col_offset,
                                 _item(row_data.get("load", ""), editable=True))
        self.frame_table.setItem(r, self._load_col_offset + 1,
                                 _item(row_data.get("stress", ""), editable=True))
        for c in range(editable_start, self.frame_table.columnCount()):
            it = self.frame_table.item(r, c)
            if it:
                it.setBackground(QColor("#2a3a2a"))
        self.frame_table.blockSignals(False)
        self.frame_table.scrollToBottom()

        # グラフ更新 + 現在画像・追跡位置を表示（2点間モード）
        if not is_ref:
            x = self._row_xval(row_data)
            for gi, g in enumerate(self._gauges):
                key = g["name"]
                y = row_data.get(f"gauge_{gi}", np.nan)
                if np.isfinite(y):
                    if key not in self.graph._lines:
                        self.graph.add_series(key, g["name"], g["color"])
                    if np.isfinite(x):
                        self.graph.update_series(key, x, y)

            # 現在フレームの画像をキャンバスに表示
            try:
                cur_gray = dic_core.imread_safe(str(path))
                cur_pix = ndarray_to_qpixmap(cur_gray)
                self.ref_canvas.set_image(cur_pix)
            except Exception:
                pass

            # ゲージ点を追跡後の位置に更新して描画
            self.ref_canvas.clear_overlays()
            gauge_results = result.get("gauges", [])
            for gi, g in enumerate(self._gauges):
                color = g["color"]
                gr = gauge_results[gi] if gi < len(gauge_results) else {}
                # 追跡後の位置 = 元の位置 + 累積変位
                tx1 = g["cx1"] + gr.get("cum_u1", gr.get("u1", 0.0))
                ty1 = g["cy1"] + gr.get("cum_v1", gr.get("v1", 0.0))
                tx2 = g["cx2"] + gr.get("cum_u2", gr.get("u2", 0.0))
                ty2 = g["cy2"] + gr.get("cum_v2", gr.get("v2", 0.0))
                self.ref_canvas.add_overlay(
                    {"type": "point", "x": int(tx1), "y": int(ty1),
                     "color": color, "label": f"{g['name']}-1"})
                self.ref_canvas.add_overlay(
                    {"type": "point", "x": int(tx2), "y": int(ty2),
                     "color": color, "label": f"{g['name']}-2"})
                self.ref_canvas.add_overlay(
                    {"type": "line", "x1": int(tx1), "y1": int(ty1),
                     "x2": int(tx2), "y2": int(ty2), "color": color})

            self._update_subset_panel(gauge_results)

        # マップモード用バッファ
        if result.get("map") is not None and not is_ref:
            if result.get("is_reanalyze") and self._map_frames:
                self._map_frames[-1] = result   # 直近フレームを置き換え
            else:
                self._map_frames.append(result)
                n = len(self._map_frames)
                self.map_slider.setRange(0, n - 1)
                self.map_slider.setValue(n - 1)
                self.btn_reanalyze.setEnabled(True)
            self._refresh_map_display()

        self.statusBar().showMessage(
            f"Frame {frame_idx}: {path.name}  |  経過: {elapsed:.1f}s")

    def _on_table_item_changed(self, item: QTableWidgetItem):
        """荷重・応力の手入力を row_data に反映し応力を自動計算。"""
        r = item.row()
        c = item.column()
        if r >= len(self._frame_rows):
            return
        row_data = self._frame_rows[r]
        area = self.sp_width.value() * self.sp_thick.value()

        if c == self._load_col_offset:
            row_data["load"] = item.text()
            try:
                load = float(item.text())
                stress = load / area if area > 0 else np.nan
                row_data["stress"] = f"{stress:.4f}"
                self.frame_table.blockSignals(True)
                it = self.frame_table.item(r, self._load_col_offset + 1)
                if it:
                    it.setText(f"{stress:.4f}")
                self.frame_table.blockSignals(False)
            except ValueError:
                row_data["stress"] = ""
        elif c == self._load_col_offset + 1:
            row_data["stress"] = item.text()

        # X軸が荷重or応力のときグラフを再描画
        if self.cb_xaxis.currentIndex() > 0:
            self._rebuild_graph()

    # ===================================================================
    # Start / Stop
    # ===================================================================

    def _on_start(self):
        folder = self.edit_folder.text().strip()
        if not folder or not pathlib.Path(folder).is_dir():
            self.statusBar().showMessage("有効な画像フォルダを選択してください")
            return

        mode = "twopoint" if self.tabs.currentIndex() == 0 else "map"

        cfg = {
            "folder":       folder,
            "sort_mode":    "mtime" if self.cb_sort.currentIndex() == 0 else "name",
            "ref_mode":     "incremental" if self.rb_ref_incr.isChecked() else "fixed",
            "subset_size":  self.sp_subset.value(),
            "step":         self.sp_step.value(),
            "search_x":     self.sp_search_x.value(),
            "search_y":     self.sp_search_y.value(),
            "zncc_threshold": self.sp_zncc.value(),
            "subpixel":     self.cb_subpixel.currentText(),
            "mode":         mode,
            "gauges":       self._gauges,
            "roi_xyxy":     self._roi_xyxy,
            "n_workers":    self.sp_workers.value(),
            "icgn":         self.chk_icgn.isChecked(),
        }

        self._map_stack.setCurrentIndex(0)
        self.btn_dic_run.setEnabled(False)

        self._worker = DICWorker(cfg, self)
        self._worker.sig_frame.connect(self._on_frame_received)
        self._worker.sig_log.connect(self._append_log)
        self._worker.sig_error.connect(
            lambda msg: self._append_log(f"[ERROR] {msg}"))
        self._worker.sig_new_image.connect(self._on_new_image)
        self._worker.start()

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._append_log(f"監視開始 — モード: {mode}  フォルダ: {folder}")
        self.statusBar().showMessage(f"監視中: {folder}")

    def _on_stop(self):
        if self._worker:
            self._worker.stop()
            self._worker.wait(3000)
            self._worker = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_dic_run.setEnabled(False)
        self._pending_reanalyze = False
        self.statusBar().showMessage("停止しました")

    def _on_reset(self):
        if self._worker:
            self._worker.stop()
            self._worker.wait(3000)
            self._worker = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_dic_run.setEnabled(False)
        self.btn_reanalyze.setEnabled(False)
        self.btn_quality.setEnabled(False)
        self._pending_reanalyze = False
        self._latest_img_path = None
        self._frame_rows.clear()
        self._map_frames.clear()
        self.frame_table.setRowCount(0)
        self.graph.clear_all()
        self.map_slider.setRange(0, 0)
        self.lbl_map_frame.setText("—")
        if self._ref_pixmap:
            self.ref_canvas.set_image(self._ref_pixmap)
        self._redraw_gauge_overlays()
        self.statusBar().showMessage("リセット済み")

    def _append_log(self, msg: str):
        self.log_box.appendPlainText(
            f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

    def closeEvent(self, ev):
        self._save_settings()      # 解析を止める前に保存する（停止処理で値が変わることがある）
        self._rt_panel.shutdown()
        self._on_stop()
        self._subset_win.close()
        super().closeEvent(ev)

    # ===================================================================
    # CSV出力
    # ===================================================================

    def _on_subset_win_toggle(self, checked: bool):
        if checked:
            self._subset_win.show()
        else:
            self._subset_win.hide()

    def _update_subset_panel(self, gauge_results: list):
        self._subset_win.update_patches(self._gauges, gauge_results)

    def _export_csv(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "CSV出力", "livedic_results.csv", "CSV (*.csv)")
        if not path:
            return
        try:
            gauge_cols = []
            for i, g in enumerate(self._gauges):
                gauge_cols.append(f"ゲージ{i+1}_{g.get('comp','exx')}")
                gauge_cols.append(f"ゲージ{i+1}_ZNCC1")
                gauge_cols.append(f"ゲージ{i+1}_ZNCC2")
            header = ["フレーム", "ファイル名", "保存時刻", "経過時間_s"] \
                     + gauge_cols + ["荷重_N", "応力_MPa"]
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(header)
                for row in self._frame_rows:
                    gauge_strs = []
                    for i in range(len(self._gauges)):
                        v = row.get(f"gauge_{i}", "")
                        gauge_strs.append(f"{v:.6f}" if isinstance(v, float) and np.isfinite(v) else "")
                        z1 = row.get(f"gauge_{i}_zncc1", "")
                        gauge_strs.append(f"{z1:.3f}" if isinstance(z1, float) and np.isfinite(z1) else "")
                        z2 = row.get(f"gauge_{i}_zncc2", "")
                        gauge_strs.append(f"{z2:.3f}" if isinstance(z2, float) and np.isfinite(z2) else "")
                    w.writerow([
                        row["frame"], row["filename"],
                        row["save_time"], f"{row['elapsed']:.2f}",
                        *gauge_strs,
                        row.get("load", ""), row.get("stress", ""),
                    ])
            self.statusBar().showMessage(f"CSV出力完了: {path}")
        except Exception as e:
            self.statusBar().showMessage(f"CSV出力エラー: {e}")


# ===========================================================================
# エントリーポイント
# ===========================================================================

def _install_excepthook(get_window):
    """
    PyQt6 は Qt スロット内の未捕捉例外で qFatal → abort() し、closeEvent も atexit も通らずに
    プロセスが消える。そのままだとカメラ取得スレッドが cv2.VideoCapture を release できず、
    400-CAM106 が DirectShow から開けなくなる（USB 抜き差しが必要）。例外はログに出して
    アプリは生かし、終了時のクリーンアップ（rt_source の atexit）が必ず走るようにする。
    """
    def hook(etype, value, tb):
        msg = "".join(traceback.format_exception(etype, value, tb))
        sys.stderr.write(msg)
        win = get_window()
        try:
            if win is not None:
                win._append_log(f"[ERROR] 未捕捉の例外: {etype.__name__}: {value}")
        except Exception:
            pass
    sys.excepthook = hook


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    holder = {"win": None}
    _install_excepthook(lambda: holder["win"])
    win = MainWindow()
    holder["win"] = win
    win.show()
    # どの経路で抜けてもカメラを解放する。DirectShow の停止はメインスレッドのメッセージポンプが
    # 必要なので、イベントループ内（aboutToQuit）と復帰後の両方で processEvents を回しつつ止める
    app.aboutToQuit.connect(lambda: rt_source.stop_all_sources(pump=app.processEvents))
    try:
        rc = app.exec()
    finally:
        rt_source.stop_all_sources(pump=app.processEvents)
    sys.exit(rc)


if __name__ == "__main__":
    multiprocessing.freeze_support()   # PyInstaller exe で子プロセス生成時にGUIが増殖するのを防ぐ
    main()
