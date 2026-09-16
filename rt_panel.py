"""
rt_panel.py  —  LiveDIC リアルタイムマップモード（GUI タブ＋解析ワーカー）

  RTWorker    : フレームソースから最新フレームを取り出して RTTracker で解析する QThread
  RTMapPanel  : 「リアルタイムマップ」タブの UI 一式（ソース選択・ROI・色スケール）

結果の保存（スナップショット .npz と見返しスライダー）は 2026-09-16 に廃止。保存方法は別途検討中。

main.py からは RTMapPanel(host, canvas_factory) で生成する。host は
  sp_subset / sp_step（QSpinBox）, chk_icgn（QCheckBox）, _append_log(str), _show_quality(img, roi)
を持つ MainWindow。ImageCanvas は循環 import を避けるため canvas_factory で受け取る。
"""
from __future__ import annotations
import time
from typing import Optional, List, Tuple

import numpy as np
import cv2
import matplotlib

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QSpinBox, QDoubleSpinBox, QRadioButton, QButtonGroup,
    QSlider, QCheckBox, QSizePolicy, QDialog, QDialogButtonBox, QFormLayout,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QPixmap, QImage

import dic_core
import dic_rt
import rt_source

STRAIN_KEYS = ["exx", "eyy", "exy"]
STRAIN_LABELS = {"exx": "εxx", "eyy": "εyy", "exy": "εxy"}
RATE_KEYS = ["rexx", "reyy", "rexy"]
RATE_LABELS = {"rexx": "ε̇xx", "reyy": "ε̇yy", "rexy": "ε̇xy"}
SCALE_PRESETS = [("0 – 2 %", 0.0, 0.02), ("0 – 5 %", 0.0, 0.05),
                 ("0 – 10 %", 0.0, 0.10), ("0 – 20 %", 0.0, 0.20),
                 ("カスタム", None, None)]
RATE_PRESETS = [("0 – 0.1 %/s", 0.0, 0.001), ("0 – 0.5 %/s", 0.0, 0.005),
                ("0 – 1 %/s", 0.0, 0.01), ("0 – 5 %/s", 0.0, 0.05),
                ("カスタム", None, None)]
CMAPS = ["jet", "turbo", "viridis", "plasma"]
RESOLUTIONS = [(1920, 1080), (3840, 2160), (1280, 720), (640, 480)]


def _lut(name: str) -> np.ndarray:
    # cm.get_cmap は matplotlib 3.9 で削除
    return (matplotlib.colormaps[name](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)


def rgb_to_qpixmap(rgb: np.ndarray) -> QPixmap:
    h, w = rgb.shape[:2]
    arr = np.ascontiguousarray(rgb)
    return QPixmap.fromImage(QImage(arr.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888))


def _fmt_scale(v_pct: float, rate: bool) -> str:
    """スケール値の表示。ひずみは整数 %、速度は小数 2 桁 %/s。"""
    return f"{v_pct:.2f} %/s" if rate else f"{v_pct:.0f} %"


class _ScaleDialog(QDialog):
    """スケール「カスタム」選択時に最小・最大を入力する小ウィンドウ。値は % 単位。"""

    def __init__(self, vmin_pct: float, vmax_pct: float, rate: bool, parent=None):
        super().__init__(parent)
        self.setWindowTitle("カラースケール（カスタム）")
        self.setModal(True)
        unit = " %/s" if rate else " %"
        form = QFormLayout(self)
        if rate:
            self.sp_min = QDoubleSpinBox(); self.sp_max = QDoubleSpinBox()
            for sp in (self.sp_min, self.sp_max):
                sp.setDecimals(2); sp.setSingleStep(0.1); sp.setRange(-1000, 1000)
        else:
            # ひずみは整数 % のみ（小数点以下は不要）
            self.sp_min = QSpinBox(); self.sp_max = QSpinBox()
            for sp in (self.sp_min, self.sp_max):
                sp.setRange(-1000, 1000)
        for sp in (self.sp_min, self.sp_max):
            sp.setSuffix(unit); sp.setMinimumWidth(110)
        self.sp_min.setValue(round(vmin_pct) if not rate else vmin_pct)
        self.sp_max.setValue(round(vmax_pct) if not rate else vmax_pct)
        form.addRow("最小:", self.sp_min)
        form.addRow("最大:", self.sp_max)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self._accept); bb.rejected.connect(self.reject)
        form.addRow(bb)
        self.sp_max.setFocus(); self.sp_max.selectAll()

    def _accept(self):
        if self.sp_max.value() <= self.sp_min.value():
            self.sp_max.setStyleSheet("background:#5a2a2a;")
            return
        self.accept()

    def values_pct(self) -> Tuple[float, float]:
        return float(self.sp_min.value()), float(self.sp_max.value())


# ===========================================================================
# 解析ワーカー
# ===========================================================================

class RTWorker(QThread):
    sig_result = pyqtSignal(dict)
    sig_log = pyqtSignal(str)
    sig_error = pyqtSignal(str)

    def __init__(self, source: rt_source.FrameSource, ref_gray: np.ndarray,
                 tracker_kwargs: dict, parent=None):
        super().__init__(parent)
        self.source = source
        self.ref_gray = ref_gray
        self.kw = tracker_kwargs
        self._stop = False
        self.tracker: Optional[dic_rt.RTTracker] = None

    def stop(self):
        self._stop = True

    def run(self):
        try:
            self.tracker = dic_rt.RTTracker(self.ref_gray, **self.kw)
        except Exception as e:
            self.sig_error.emit(f"トラッカー初期化エラー: {e}")
            return
        tr = self.tracker
        self.sig_log.emit(f"リアルタイム解析開始: {tr.N} 点 ({tr.ny}×{tr.nx})  "
                          f"backend={dic_rt.backend_name(tr.use_gpu)}  threads={tr.n_threads}")
        t_start = time.perf_counter()
        last_id = -1
        n_proc = 0
        n_skip = 0
        fps_t0 = time.perf_counter()
        fps_n = 0
        proc_fps = 0.0

        while not self._stop:
            fr = self.source.latest()
            if fr is None or fr[0] == last_id:
                time.sleep(0.001)
                continue
            fid, ts, gray = fr
            if last_id >= 0:
                n_skip += max(0, fid - last_id - 1)
            last_id = fid
            try:
                res = tr.process(gray, t=ts)
            except Exception as e:
                self.sig_error.emit(f"解析エラー: {e}")
                time.sleep(0.05)
                continue
            n_proc += 1
            fps_n += 1
            now = time.perf_counter()
            if now - fps_t0 >= 1.0:
                proc_fps = fps_n / (now - fps_t0)
                fps_n = 0
                fps_t0 = now
            elapsed = ts - t_start
            out = {"fid": fid, "elapsed": elapsed, "gray": gray, "res": res,
                   "proc_fps": proc_fps, "n_proc": n_proc, "n_skip": n_skip,
                   "src_fps": self.source.fps}
            if res["stats"]["rereferenced"]:
                self.sig_log.emit(f"  自動再参照 #{res['stats']['n_reref']}  "
                                  f"(t={elapsed:.1f}s, 中央値ZNCC={res['stats']['median_zncc']:.3f})")
            self.sig_result.emit(out)
        self.sig_log.emit(f"リアルタイム解析停止: 処理 {n_proc} フレーム / 間引き {n_skip} フレーム / 再参照 {tr.n_reref} 回")


# ===========================================================================
# タブ UI
# ===========================================================================

class RTMapPanel(QWidget):
    def __init__(self, host, canvas_factory, parent=None):
        super().__init__(parent)
        self.host = host
        self._source: Optional[rt_source.FrameSource] = None
        self._worker: Optional[RTWorker] = None
        self._roi: Optional[Tuple[int, int, int, int]] = None
        self._latest: Optional[dict] = None       # 最新の解析結果（描画待ち）
        self._latest_drawn_fid = -1
        self._preview_fid = -1
        self._ref_gray: Optional[np.ndarray] = None
        self._lut = _lut(CMAPS[0])
        self._comp = "exx"
        self._cam_list: List[Tuple[int, str]] = []
        self._scanned = False        # タブを初めて表示したときに一度だけカメラ検索する

        lay = QVBoxLayout(self)
        lay.setSpacing(4)

        # ---- 行1：ソース ----
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("ソース:"))
        self.cb_source = QComboBox()
        self.cb_source.setMinimumWidth(150)
        row1.addWidget(self.cb_source)
        self.btn_scan = QPushButton("カメラ検索")
        self.btn_scan.clicked.connect(self._scan_cameras)
        row1.addWidget(self.btn_scan)
        row1.addWidget(QLabel("解像度:"))
        self.cb_res = QComboBox()
        for w, h in RESOLUTIONS:
            self.cb_res.addItem(f"{w}×{h}")
        row1.addWidget(self.cb_res)
        # ソース／解像度を選ぶと即プレビュー開始（activated は同じ項目を選び直しても発火する）
        self.cb_source.activated.connect(lambda _i: self._start_source())
        self.cb_res.activated.connect(lambda _i: self._start_source())
        row1.addSpacing(16)
        self.chk_gpu = QCheckBox("GPUを使う")
        self.chk_gpu.setChecked(dic_rt.gpu_available())
        self.chk_gpu.setEnabled(dic_rt.gpu_available())
        row1.addWidget(self.chk_gpu)
        self.lbl_backend = QLabel(f"バックエンド: {dic_rt.backend_name()}")
        self.lbl_backend.setStyleSheet("color:#8bc34a;" if dic_rt.gpu_available() else "color:#aaa;")
        if not dic_rt.gpu_available() and dic_rt.gpu_error():
            self.lbl_backend.setToolTip(f"CuPy 利用不可: {dic_rt.gpu_error()}")
        row1.addWidget(self.lbl_backend)
        row1.addStretch()
        lay.addLayout(row1)

        # ---- 行2：解析設定 ----
        row2 = QHBoxLayout()
        self.btn_roi = QPushButton("ROI設定（ドラッグ）")
        self.btn_roi.setCheckable(True)
        self.btn_roi.toggled.connect(lambda c: self.canvas.set_mode("roi" if c else "none"))
        row2.addWidget(self.btn_roi)
        self.btn_roi_clear = QPushButton("ROIクリア")
        self.btn_roi_clear.clicked.connect(self._clear_roi)
        row2.addWidget(self.btn_roi_clear)
        self.lbl_roi = QLabel("ROI: 全域")
        self.lbl_roi.setStyleSheet("color:#888;font-size:9px;")
        row2.addWidget(self.lbl_roi)
        row2.addSpacing(12)
        row2.addWidget(QLabel("探索 ±"))
        self.sp_r1 = QSpinBox(); self.sp_r1.setRange(1, 100); self.sp_r1.setValue(10); self.sp_r1.setFixedWidth(46)
        self.sp_r1.setToolTip("探索半径 [px]。予測位置を中心に全点をこの範囲で探索する")
        self.sp_r2 = QSpinBox(); self.sp_r2.setRange(0, 200); self.sp_r2.setValue(30); self.sp_r2.setFixedWidth(46)
        self.sp_r2.setToolTip("拡張探索の半径 [px]。第1段で相関が取れなかった点だけをこの範囲で再探索する。"
                              "それでも見つからない大きな移動は位相相関で全体ズレを測って復帰する。0 で無効")
        row2.addWidget(self.sp_r1); row2.addWidget(QLabel("拡張 ±")); row2.addWidget(self.sp_r2)
        row2.addSpacing(12)
        row2.addWidget(QLabel("ZNCC閾値:"))
        self.sp_zncc = QDoubleSpinBox(); self.sp_zncc.setRange(0, 1); self.sp_zncc.setSingleStep(0.05)
        self.sp_zncc.setValue(0.5); self.sp_zncc.setDecimals(2); self.sp_zncc.setFixedWidth(56)
        row2.addWidget(self.sp_zncc)
        row2.addWidget(QLabel("再参照ZNCC:"))
        self.sp_reref = QDoubleSpinBox(); self.sp_reref.setRange(0, 1); self.sp_reref.setSingleStep(0.05)
        self.sp_reref.setValue(0.75); self.sp_reref.setDecimals(2); self.sp_reref.setFixedWidth(56)
        self.sp_reref.setToolTip("中央値ZNCCがこの値を下回ると現フレームを新しい参照にする（変位は累積）")
        row2.addWidget(self.sp_reref)
        self.lbl_params = QLabel("")
        self.lbl_params.setStyleSheet("color:#888;font-size:9px;")
        row2.addWidget(self.lbl_params)
        row2.addStretch()
        lay.addLayout(row2)

        # ---- 行3：表示 ----
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("成分:"))
        self._comp_bg = QButtonGroup(self)
        for i, key in enumerate(STRAIN_KEYS):
            rb = QRadioButton(STRAIN_LABELS[key])
            if i == 0: rb.setChecked(True)
            rb.toggled.connect(lambda c, k=key: self._set_comp(k) if c else None)
            self._comp_bg.addButton(rb)
            row3.addWidget(rb)
        row3.addWidget(QLabel("速度:"))
        for key in RATE_KEYS:
            rb = QRadioButton(RATE_LABELS[key])
            rb.setToolTip("ひずみ速度 [%/s]。「速度窓」秒前のひずみ場との差分から計算")
            rb.toggled.connect(lambda c, k=key: self._set_comp(k) if c else None)
            self._comp_bg.addButton(rb)
            row3.addWidget(rb)
        self.sp_rate_win = QDoubleSpinBox(); self.sp_rate_win.setRange(0.1, 60); self.sp_rate_win.setValue(3.0)
        self.sp_rate_win.setDecimals(1); self.sp_rate_win.setSuffix(" s"); self.sp_rate_win.setFixedWidth(60)
        self.sp_rate_win.setToolTip("ひずみ速度の計算窓。短いほど応答が速くノイズが増える")
        row3.addWidget(QLabel("速度窓:")); row3.addWidget(self.sp_rate_win)
        row3.addSpacing(12)
        row3.addWidget(QLabel("描画位置:"))
        self._pos_bg = QButtonGroup(self)
        self.rb_pos_follow = QRadioButton("追従")
        self.rb_pos_follow.setToolTip("各セルを変形後の位置に描く（ライブ映像の材料点と一致。セルは伸びに合わせて広がる）")
        self.rb_pos_ref = QRadioButton("参照位置")
        self.rb_pos_ref.setToolTip("各セルを参照画像上の位置に固定して描く（マップは動かない）")
        self.rb_pos_follow.setChecked(True)
        for rb in (self.rb_pos_follow, self.rb_pos_ref):
            self._pos_bg.addButton(rb); row3.addWidget(rb)
            rb.toggled.connect(lambda c: self._request_redraw() if c else None)
        self.chk_rate_smooth = QCheckBox("速度平滑化")
        self.chk_rate_smooth.setChecked(True)
        self.chk_rate_smooth.setToolTip("速度マップに 3×3 平均フィルタ。サブピクセル誤差由来の縞を抑えるが局所ピークは鈍る")
        row3.addWidget(self.chk_rate_smooth)
        row3.addStretch()
        lay.addLayout(row3)

        # ---- 行3b：スケール ----
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("スケール:"))
        self.cb_scale = QComboBox()
        for name, _, _ in SCALE_PRESETS:
            self.cb_scale.addItem(name)
        self.cb_scale.setCurrentIndex(1)
        # 現在のスケール（割合）と、モードごとに覚えておくカスタム値（% 単位）。
        # 「カスタム」を選ぶと _ScaleDialog で入力し、キャンセルなら直前のプリセットに戻す
        self._vmin, self._vmax = SCALE_PRESETS[1][1], SCALE_PRESETS[1][2]
        self._custom_pct = {False: (0.0, 5.0), True: (0.0, 0.5)}     # key: is_rate
        self._scale_prev_idx = 1
        self.cb_scale.activated.connect(self._on_scale_activated)
        row3.addWidget(self.cb_scale)
        row3.addWidget(QLabel("カラーマップ:"))
        self.cb_cmap = QComboBox(); self.cb_cmap.addItems(CMAPS)
        self.cb_cmap.currentTextChanged.connect(self._on_cmap_changed)
        row3.addWidget(self.cb_cmap)
        row3.addWidget(QLabel("不透明度:"))
        self.sl_alpha = QSlider(Qt.Orientation.Horizontal)
        self.sl_alpha.setRange(0, 100); self.sl_alpha.setValue(65); self.sl_alpha.setFixedWidth(90)
        self.sl_alpha.valueChanged.connect(self._request_redraw)
        row3.addWidget(self.sl_alpha)
        row3.addStretch()
        lay.addLayout(row3)

        # ---- 行4：開始/停止（左寄せ） ----
        row4 = QHBoxLayout()
        self.btn_quality = QPushButton("🔍 品質評価")
        self.btn_quality.clicked.connect(self._quality)
        row4.addWidget(self.btn_quality)
        self.btn_start = QPushButton("▶  参照取得 ＆ 解析開始")
        self.btn_start.setObjectName("btn_start")
        self.btn_start.setMinimumHeight(32)
        self.btn_start.setEnabled(False)
        self.btn_start.clicked.connect(self._start)
        row4.addWidget(self.btn_start)
        self.btn_stop = QPushButton("■  解析停止")
        self.btn_stop.setObjectName("btn_stop")
        self.btn_stop.setMinimumHeight(32)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop)
        row4.addWidget(self.btn_stop)
        row4.addStretch()
        lay.addLayout(row4)

        # ---- ステータス ----
        self.lbl_status = QLabel("待機中 — カメラを検索しています…")
        self.lbl_status.setStyleSheet("color:#4fc3f7;font-size:10px;")
        lay.addWidget(self.lbl_status)

        # ---- キャンバス ＋ カラーバー ----
        self.canvas = canvas_factory()
        self.canvas.signal_roi.connect(self._on_roi)
        lay.addWidget(self.canvas, stretch=1)
        cb_row = QHBoxLayout()
        self.lbl_cb_min = QLabel("0 %")
        self.lbl_cbar = QLabel()
        self.lbl_cbar.setFixedHeight(14)
        # 固定幅 256px の画像をラベル側で伸縮表示する。ラベル幅に合わせて画像を作り直すと
        # 「画像幅 → 推奨サイズ拡大 → ウィンドウ拡大」のループでウィンドウが伸び続ける
        self.lbl_cbar.setScaledContents(True)
        self.lbl_cbar.setMinimumWidth(64)
        self.lbl_cbar.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.lbl_cb_max = QLabel("5 %")
        cb_row.addWidget(self.lbl_cb_min); cb_row.addWidget(self.lbl_cbar, stretch=1); cb_row.addWidget(self.lbl_cb_max)
        lay.addLayout(cb_row)

        # 長文ラベルがウィンドウ最小幅を押し広げないようにする（QLabel は文字列幅を最小幅にする）
        for lbl in (self.lbl_status, self.lbl_params, self.lbl_roi, self.lbl_backend):
            lbl.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            lbl.setMinimumWidth(1)
        self.lbl_backend.setMinimumWidth(120)

        # ---- タイマー ----
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

        self._populate_sources([])
        self._apply_scale_index(self.cb_scale.currentIndex())

    def bind_host(self):
        """host 側の設定パネル（sp_subset / sp_step）が構築された後に main.py から呼ぶ。"""
        self._update_params_label()
        self.host.sp_subset.valueChanged.connect(self._update_params_label)
        self.host.sp_step.valueChanged.connect(self._update_params_label)

    # ------------------------------------------------------------------
    # ソース
    # ------------------------------------------------------------------
    def _populate_sources(self, cams: List[Tuple[int, str]]):
        cur = self.cb_source.currentText()
        self.cb_source.blockSignals(True)
        self.cb_source.clear()
        for i, name in cams:
            label = f"カメラ {i}: {name}" if name else f"カメラ {i}"
            if name and rt_source.is_builtin_camera_name(name):
                label += "（内蔵）"
            self.cb_source.addItem(label, ("cam", i))
        if not cams:
            self.cb_source.addItem("（カメラなし）", None)
        idx = self.cb_source.findText(cur)
        self.cb_source.setCurrentIndex(idx if idx >= 0 else 0)
        self.cb_source.blockSignals(False)

    def _scan_cameras(self):
        self.btn_scan.setEnabled(False)
        self.btn_scan.setText("検索中…")
        self._scan_thread = _ScanThread(self)
        self._scan_thread.done.connect(self._on_scan_done)
        self._scan_thread.start()

    def _on_scan_done(self, cams: list):
        self._cam_list = cams
        self._populate_sources(cams)
        self.btn_scan.setEnabled(True)
        self.btn_scan.setText("カメラ検索")
        self.host._append_log("カメラ検索: " + (", ".join(f"{i}: {n or '?'}" for i, n in cams) if cams else "なし"))
        if cams and self._worker is None:
            # 外付け（内蔵らしくない名前）を優先。ノート PC では内蔵が 0 番に来るため
            pick = rt_source.pick_default_camera(cams)
            for k in range(self.cb_source.count()):
                data = self.cb_source.itemData(k)
                if data and data[1] == pick:
                    self.cb_source.setCurrentIndex(k)
                    break
            self._start_source()

    def showEvent(self, ev):
        super().showEvent(ev)
        if not self._scanned:
            self._scanned = True
            self._scan_cameras()

    def _stop_source(self):
        if self._source is not None:
            self._source.stop()
            self._source = None
        self.btn_start.setEnabled(False)

    def _start_source(self):
        """コンボで選ばれているソースを開いてプレビューを始める（解析中は何もしない）。"""
        if self._worker is not None:
            return
        self._stop_source()
        self._reset_view()
        data = self.cb_source.currentData()
        if data is None:
            return
        kind, idx = data
        w, h = RESOLUTIONS[self.cb_res.currentIndex()]
        if kind == "cam":
            self._source = rt_source.CameraSource(idx, w, h, 30)
        else:   # 回帰テスト用（UI からは選べない）
            self._source = rt_source.SyntheticSource(w, h, 30.0)
        self._source.start()
        self._preview_fid = -1
        self.btn_start.setEnabled(True)
        self.lbl_status.setText("ソースを開いています…")
        self.host._append_log(f"プレビュー開始: {self.cb_source.currentText()} {w}×{h}")

    # ------------------------------------------------------------------
    # ROI / 品質
    # ------------------------------------------------------------------
    def _on_roi(self, x1, y1, x2, y2):
        self._roi = (x1, y1, x2, y2)
        self.lbl_roi.setText(f"ROI: ({x1},{y1})–({x2},{y2})")
        self.btn_roi.setChecked(False)
        self._draw_roi_overlay()
        self._update_params_label()

    def _clear_roi(self):
        self._roi = None
        self.lbl_roi.setText("ROI: 全域")
        self.canvas.clear_overlays()
        self._update_params_label()

    def _draw_roi_overlay(self):
        self.canvas.clear_overlays()
        if self._roi:
            x1, y1, x2, y2 = self._roi
            self.canvas.add_overlay({"type": "rect", "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                                     "color": "#ffaa00"})

    def _update_params_label(self):
        if not hasattr(self.host, "sp_subset"):
            return
        shape = None
        if self._source is not None and self._source.shape is not None:
            shape = self._source.shape
        elif self._ref_gray is not None:
            shape = self._ref_gray.shape
        if shape is None:
            self.lbl_params.setText(f"subset={self.host.sp_subset.value()} step={self.host.sp_step.value()}")
            return
        try:
            X, _ = dic_core.create_grid(shape, self.host.sp_subset.value() | 1, self.sp_r1.value(),
                                        self.host.sp_step.value(), self._roi)
            self.lbl_params.setText(f"subset={self.host.sp_subset.value()} step={self.host.sp_step.value()} "
                                    f"→ {X.size} 点 ({X.shape[0]}×{X.shape[1]})")
        except Exception:
            self.lbl_params.setText("グリッド生成不可")

    def _quality(self):
        img = None
        if self._source is not None and self._source.latest() is not None:
            img = self._source.latest()[2]
        elif self._ref_gray is not None:
            img = self._ref_gray
        if img is None:
            self.lbl_status.setText("品質評価: 画像がありません")
            return
        self.host._show_quality(img, self._roi)

    # ------------------------------------------------------------------
    # 表示設定
    # ------------------------------------------------------------------
    def _is_rate(self) -> bool:
        return self._comp in RATE_KEYS

    def _presets(self):
        return RATE_PRESETS if self._is_rate() else SCALE_PRESETS

    def _set_comp(self, key: str):
        was_rate = self._is_rate()
        self._comp = key
        if was_rate != self._is_rate():
            # ひずみ [%] と速度 [%/s] でプリセット一覧を入れ替える（選択位置は引き継ぐ）
            idx = self.cb_scale.currentIndex()
            self.cb_scale.blockSignals(True)
            self.cb_scale.clear()
            for name, _, _ in self._presets():
                self.cb_scale.addItem(name)
            idx = min(idx, self.cb_scale.count() - 1)
            self.cb_scale.setCurrentIndex(idx)
            self.cb_scale.blockSignals(False)
            self._apply_scale_index(idx)
            return
        self._request_redraw()

    def _custom_index(self) -> int:
        return len(self._presets()) - 1

    def _on_scale_activated(self, idx: int):
        """プルダウンをユーザーが選んだとき。「カスタム」は別ウィンドウで最小・最大を入力する。"""
        if idx == self._custom_index():
            rate = self._is_rate()
            cur = self._custom_pct[rate]
            dlg = _ScaleDialog(cur[0], cur[1], rate, self)
            if dlg.exec() != QDialog.DialogCode.Accepted:
                self.cb_scale.blockSignals(True)
                self.cb_scale.setCurrentIndex(self._scale_prev_idx)
                self.cb_scale.blockSignals(False)
                return
            self._custom_pct[rate] = dlg.values_pct()
        self._apply_scale_index(idx)

    def _apply_scale_index(self, idx: int):
        """プルダウンの idx に対応するスケールを反映する（カスタムは記憶している値を使う）。"""
        rate = self._is_rate()
        if idx == self._custom_index():
            lo, hi = self._custom_pct[rate]
            self._vmin, self._vmax = lo / 100.0, hi / 100.0
            label = f"カスタム ({_fmt_scale(lo, rate)} – {_fmt_scale(hi, rate)})"
        else:
            _, self._vmin, self._vmax = self._presets()[idx]
            label = "カスタム"
        self.cb_scale.setItemText(self._custom_index(), label)
        self._scale_prev_idx = idx
        self._request_redraw()

    def _on_cmap_changed(self, name: str):
        self._lut = _lut(name)
        self._request_redraw()

    def _scale(self) -> Tuple[float, float]:
        """表示スケール（割合）。"""
        return self._vmin, self._vmax

    def _update_colorbar(self):
        vmin, vmax = self._scale()
        grad = np.arange(256, dtype=np.uint8)
        rgb = np.repeat(self._lut[grad][None, :, :], 14, axis=0)
        self.lbl_cbar.setPixmap(rgb_to_qpixmap(rgb))
        rate = self._is_rate()
        self.lbl_cb_min.setText(_fmt_scale(vmin * 100, rate))
        self.lbl_cb_max.setText(_fmt_scale(vmax * 100, rate))

    def _request_redraw(self):
        self._latest_drawn_fid = -1
        self._update_colorbar()

    def _reset_view(self):
        """ソース切替・解析開始時に描画済みフレームの記録をリセットする。"""
        self._latest_drawn_fid = -1
        self._preview_fid = -1

    # ------------------------------------------------------------------
    # 開始 / 停止
    # ------------------------------------------------------------------
    def _tracker_kwargs(self) -> dict:
        stages = [self.sp_r1.value()]
        if self.sp_r2.value() > 0: stages.append(self.sp_r2.value())
        return dict(
            subset_size=self.host.sp_subset.value() | 1,
            step=self.host.sp_step.value(),
            roi_xyxy=self._roi,
            zncc_threshold=self.sp_zncc.value(),
            search_stages=stages,
            reref_zncc=self.sp_reref.value(),
            use_gpu=self.chk_gpu.isChecked() and dic_rt.gpu_available(),
            rate_window=self.sp_rate_win.value(),
            rate_smooth=3 if self.chk_rate_smooth.isChecked() else 1,
            icgn=self.host.chk_icgn.isChecked(),   # 左パネル「IC-GN 精密化」（マップモードと共通）
        )

    def _start(self):
        if self._source is None or self._source.latest() is None:
            self.lbl_status.setText("フレームがまだ来ていません")
            return
        self._reset_view()
        fid, ts, gray = self._source.latest()
        self._ref_gray = gray.copy()
        try:
            self._worker = RTWorker(self._source, self._ref_gray, self._tracker_kwargs(), self)
        except Exception as e:
            self.lbl_status.setText(f"開始エラー: {e}")
            return
        self._worker.sig_result.connect(self._on_result)
        self._worker.sig_log.connect(self.host._append_log)
        self._worker.sig_error.connect(lambda m: self.host._append_log(f"[ERROR] {m}"))
        self._worker.start()
        self._latest = None
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        for wdg in (self.btn_roi, self.btn_roi_clear, self.sp_r1, self.sp_r2,
                    self.sp_zncc, self.sp_reref, self.chk_gpu, self.btn_scan,
                    self.cb_source, self.cb_res, self.sp_rate_win, self.chk_rate_smooth, self.host.chk_icgn):
            wdg.setEnabled(False)
        self.host._append_log(f"参照取得: フレーム {fid}")

    def _stop(self):
        if self._worker is None:
            return
        self._worker.stop()
        self._worker.wait(3000)
        self._worker = None
        self.btn_start.setEnabled(self._source is not None)
        self.btn_stop.setEnabled(False)
        for wdg in (self.btn_roi, self.btn_roi_clear, self.sp_r1, self.sp_r2,
                    self.sp_zncc, self.sp_reref, self.btn_scan,
                    self.cb_source, self.cb_res, self.sp_rate_win, self.chk_rate_smooth, self.host.chk_icgn):
            wdg.setEnabled(True)
        self.chk_gpu.setEnabled(dic_rt.gpu_available())
        self.lbl_status.setText("解析停止")

    def shutdown(self):
        """アプリ終了時に呼ぶ。"""
        self._timer.stop()
        self._stop()
        self._stop_source()

    # ------------------------------------------------------------------
    # 結果受信・描画
    # ------------------------------------------------------------------
    def _on_result(self, out: dict):
        self._latest = out

    def _tick(self):
        if self._worker is not None and self._latest is not None:
            out = self._latest
            if out["fid"] == self._latest_drawn_fid:
                return
            self._latest_drawn_fid = out["fid"]
            self._draw_result(out["gray"], out["res"])
            st = out["res"]["stats"]
            self.lbl_status.setText(
                f"取得 {out['src_fps']:.1f} fps | 解析 {out['proc_fps']:.1f} fps ({st['proc_time'] * 1000:.0f} ms) "
                f"| 間引き {out['n_skip']} | 中央値ZNCC {st['median_zncc']:.3f} | 不良 {st['frac_bad'] * 100:.1f}% "
                f"| 拡張探索 {st['stage_counts'][1:] if len(st['stage_counts']) > 1 else '-'}"
                f"{' (省略 ' + str(st['skipped']) + ')' if st['skipped'] else ''}"
                f"{' 復帰:プローブ' + str(st['probe']) if st.get('probe', 0) > 0 else ''}"
                f"{' 復帰:位相相関' if st.get('probe', 0) < 0 else ''} "
                f"{' | IC-GN ' + format(st['icgn_frac'] * 100, '.0f') + '% ' + format(st['icgn_iter'], '.1f') + '反復 ' + format(st['icgn_time'] * 1000, '.0f') + 'ms' if 'icgn_frac' in st else ''}"
                f"| 再参照 {st['n_reref']} 回 | 経過 {out['elapsed']:.1f} s"
                f"{' | 動画終了' if getattr(self._source, 'finished', False) else ''}")
        elif self._source is not None:
            fr = self._source.latest()
            if self._source.error:
                self.lbl_status.setText(f"[ERROR] {self._source.error}")
                return
            if fr is None or fr[0] == self._preview_fid:
                return
            self._preview_fid = fr[0]
            if self._preview_fid % 2:      # プレビューは約15fpsで十分
                return
            self.canvas.set_image(rgb_to_qpixmap(cv2.cvtColor(fr[2], cv2.COLOR_GRAY2RGB)))
            if self._source.shape is not None and self.lbl_params.text().startswith("subset=") \
                    and "点" not in self.lbl_params.text():
                self._update_params_label()
            self.lbl_status.setText(f"プレビュー中: {fr[2].shape[1]}×{fr[2].shape[0]}  {self._source.fps:.1f} fps"
                                    f"  — ROI を設定して「参照取得＆解析開始」")

    def _draw_result(self, gray: np.ndarray, res: dict):
        data = res.get(self._comp)
        vmin, vmax = self._scale()
        alpha = int(self.sl_alpha.value() * 2.55)
        rgba = None
        if data is not None:
            follow = self.rb_pos_follow.isChecked() and res.get("U") is not None
            rgba = dic_rt.strain_to_rgba(data, vmin, vmax, res["X"], res["Y"], gray.shape, self._lut, alpha,
                                         U=res.get("U_draw", res["U"]) if follow else None,
                                         V=res.get("V_draw", res["V"]) if follow else None)
        rgb = dic_rt.compose_overlay(gray, rgba)
        self.canvas.set_image(rgb_to_qpixmap(rgb))


class _ScanThread(QThread):
    done = pyqtSignal(list)

    def run(self):
        # 列挙も open テストもカメラがハングしていると固まるので、必ずこのスレッドで行う
        try:
            self.done.emit(rt_source.list_cameras_named())
        except Exception:
            self.done.emit([])
