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
import pathlib
from typing import Optional, List, Tuple

import numpy as np
import cv2
import matplotlib

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QSpinBox, QDoubleSpinBox, QRadioButton, QButtonGroup,
    QSlider, QCheckBox, QSizePolicy, QDialog, QDialogButtonBox, QFormLayout, QMessageBox,
    QLineEdit, QFileDialog,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QRect
from PyQt6.QtGui import QPixmap, QImage, QPainter, QKeySequence, QShortcut

import dic_core
import dic_rt
import rt_source
import rt_record

STRAIN_KEYS = ["exx", "eyy", "exy"]
STRAIN_LABELS = {"exx": "εxx", "eyy": "εyy", "exy": "εxy"}
RATE_KEYS = ["rexx", "reyy", "rexy"]
RATE_LABELS = {"rexx": "ε̇xx", "reyy": "ε̇yy", "rexy": "ε̇xy"}
SCALE_PRESETS = [("0 – 2 %", 0.0, 0.02), ("0 – 5 %", 0.0, 0.05),
                 ("0 – 10 %", 0.0, 0.10), ("0 – 20 %", 0.0, 0.20),
                 ("0 – 40 %", 0.0, 0.40),
                 ("カスタム", None, None)]
RATE_PRESETS = [("0 – 0.1 %/s", 0.0, 0.001), ("0 – 0.5 %/s", 0.0, 0.005),
                ("0 – 1 %/s", 0.0, 0.01), ("0 – 5 %/s", 0.0, 0.05),
                ("カスタム", None, None)]
FOCUS_TEXT = {False: "🔓 フォーカス自動", True: "🔒 フォーカス固定中"}
SUBSET_COLOR = "#ff3b30"          # 追従できている点
SUBSET_LOST_COLOR = "#ffd54f"     # 見失った点
SUBSET_RGB = (255, 59, 48)
SUBSET_LOST_RGB = (255, 213, 79)
# サブセット枠を描く点の割合。1/2 は市松、1/4 は 1 つおき、1/8 はその市松（いずれも等方的に散る）
SUBSET_DENSITY = [("枠: 全点", 1), ("枠: 1/2", 2), ("枠: 1/4", 4), ("枠: 1/8", 8)]
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


ROTATIONS = [("横（そのまま）", None),
             ("縦（時計回り 90°）", cv2.ROTATE_90_CLOCKWISE),
             ("縦（反時計回り 90°）", cv2.ROTATE_90_COUNTERCLOCKWISE)]   # (表示名, cv2.rotate のコード)


class _FitImage(QWidget):
    """アスペクト比を保ってウィジェットいっぱいに画像を描く表示専用ウィジェット（補間なし）。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pix: Optional[QPixmap] = None
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(160, 120)
        self.setStyleSheet("background:#000;")

    def set_pixmap(self, pix: QPixmap):
        self._pix = pix
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.fillRect(self.rect(), Qt.GlobalColor.black)
        if self._pix is None or self._pix.isNull():
            return
        # 科学データの表示なので補間・美化はしない（FastTransformation 相当）
        scaled = self._pix.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                  Qt.TransformationMode.FastTransformation)
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        p.drawPixmap(QRect(x, y, scaled.width(), scaled.height()), scaled)


class RTDisplayWindow(QWidget):
    """
    カメラ画像＋ひずみマップを別ウィンドウで表示する（第 2 モニタ・プロジェクタ用）。
    タブ側のキャンバスは操作用（ROI 等）にそのまま残し、このウィンドウは同じ RGB を
    向き（横／縦 90° 回転）を変えて映すだけ。F11 または ダブルクリックで全画面、Esc で解除。
    """
    closed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle("LiveDIC — リアルタイムマップ表示")
        self.resize(720, 1000)
        self._rot_code: Optional[int] = None
        self._last_rgb: Optional[np.ndarray] = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4); lay.setSpacing(3)
        bar = QHBoxLayout()
        bar.addWidget(QLabel("向き:"))
        self.cb_rot = QComboBox()
        for name, _ in ROTATIONS:
            self.cb_rot.addItem(name)
        self.cb_rot.currentIndexChanged.connect(self._on_rot)
        bar.addWidget(self.cb_rot)
        self.btn_full = QPushButton("全画面 (F11)")
        self.btn_full.setCheckable(True)
        self.btn_full.toggled.connect(self._set_fullscreen)
        bar.addWidget(self.btn_full)
        bar.addStretch()
        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet("color:#4fc3f7;font-size:10px;")
        self.lbl_status.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        bar.addWidget(self.lbl_status, stretch=2)
        lay.addLayout(bar)
        self.view = _FitImage()
        lay.addWidget(self.view, stretch=1)
        cb = QHBoxLayout()
        self.lbl_cb_min = QLabel("")
        self.lbl_cbar = QLabel(); self.lbl_cbar.setFixedHeight(14); self.lbl_cbar.setScaledContents(True)
        self.lbl_cbar.setMinimumWidth(64)
        self.lbl_cbar.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.lbl_cb_max = QLabel("")
        cb.addWidget(self.lbl_cb_min); cb.addWidget(self.lbl_cbar, stretch=1); cb.addWidget(self.lbl_cb_max)
        lay.addLayout(cb)
        QShortcut(QKeySequence("F11"), self, activated=lambda: self.btn_full.toggle())
        QShortcut(QKeySequence("Escape"), self, activated=lambda: self.btn_full.setChecked(False))

    # ---- 外部 API ----
    def set_rotation_index(self, idx: int):
        self.cb_rot.setCurrentIndex(idx)

    def set_frame(self, rgb: np.ndarray):
        self._last_rgb = rgb
        # cv2.rotate は SIMD で 1080p RGB を約 1 ms（np.rot90 + 連続化コピーは約 8 ms）
        img = cv2.rotate(rgb, self._rot_code) if self._rot_code is not None else rgb
        self.view.set_pixmap(rgb_to_qpixmap(img))

    def set_colorbar(self, bar_rgb: np.ndarray, text_min: str, text_max: str):
        self.lbl_cbar.setPixmap(rgb_to_qpixmap(bar_rgb))
        self.lbl_cb_min.setText(text_min); self.lbl_cb_max.setText(text_max)

    def set_status(self, text: str):
        self.lbl_status.setText(text)

    # ---- 内部 ----
    def _on_rot(self, idx: int):
        self._rot_code = ROTATIONS[idx][1]
        if self._last_rgb is not None:
            self.set_frame(self._last_rgb)

    def _set_fullscreen(self, on: bool):
        if on:
            self.showFullScreen()
        else:
            self.showNormal()

    def mouseDoubleClickEvent(self, ev):
        self.btn_full.toggle()

    def closeEvent(self, ev):
        self.btn_full.setChecked(False)
        self.closed.emit()
        super().closeEvent(ev)


class _SaveDialog(QDialog):
    """
    解析停止後に「残す / 一部だけ残す / 捨てる」を選ぶ。

    記録するかどうかの判断をここまで遅らせるのが狙い（開始前に決めると、調整のたびに録って
    容量が膨らむか、録り忘れて本番を逃すかのどちらかになる）。
    """

    def __init__(self, rec, parent=None):
        super().__init__(parent)
        self.setWindowTitle("記録を残しますか？")
        self.setModal(True)
        self.rec = rec
        self.choice = None                      # ("all"| "marker"| "last"| None), 値
        dur = (rec.t_last - rec.t_first) if (rec.t_first is not None and rec.t_last is not None) else 0.0
        lay = QVBoxLayout(self)
        head = QLabel(f"{rec.n_frames:,} フレーム / {int(dur // 60):02d}:{int(dur % 60):02d} / "
                      f"{rec.disk_bytes() / 1e6:,.0f} MB"
                      + (f" / マーカー {len(rec.markers)} 個" if rec.markers else ""))
        head.setStyleSheet("font-size:12px;")
        lay.addWidget(head)

        self._bg = QButtonGroup(self)
        self.rb_all = QRadioButton("すべて残す")
        self.rb_all.setChecked(True)
        self._bg.addButton(self.rb_all); lay.addWidget(self.rb_all)

        row_m = QHBoxLayout()
        self.rb_mark = QRadioButton("マーカーの前後")
        self.rb_mark.setEnabled(bool(rec.markers))
        self._bg.addButton(self.rb_mark); row_m.addWidget(self.rb_mark)
        self.sp_mark = QSpinBox(); self.sp_mark.setRange(1, 600); self.sp_mark.setValue(30)
        self.sp_mark.setSuffix(" 秒"); self.sp_mark.setEnabled(False)
        row_m.addWidget(self.sp_mark); row_m.addStretch(); lay.addLayout(row_m)

        row_l = QHBoxLayout()
        self.rb_last = QRadioButton("最後の")
        self._bg.addButton(self.rb_last); row_l.addWidget(self.rb_last)
        self.sp_last = QDoubleSpinBox(); self.sp_last.setRange(0.1, 600); self.sp_last.setValue(2.0)
        self.sp_last.setDecimals(1); self.sp_last.setSuffix(" 分"); self.sp_last.setEnabled(False)
        row_l.addWidget(self.sp_last); row_l.addStretch(); lay.addLayout(row_l)

        self.rb_mark.toggled.connect(self.sp_mark.setEnabled)
        self.rb_last.toggled.connect(self.sp_last.setEnabled)

        self.lbl_dest = QLabel(f"保存先: {rec.base}")
        self.lbl_dest.setStyleSheet("color:#888;font-size:9px;")
        self.lbl_dest.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.lbl_dest.setMinimumWidth(1)
        lay.addWidget(self.lbl_dest)

        bb = QDialogButtonBox()
        self.b_keep = bb.addButton("保存", QDialogButtonBox.ButtonRole.AcceptRole)
        self.b_drop = bb.addButton("破棄", QDialogButtonBox.ButtonRole.DestructiveRole)
        self.b_drop.setStyleSheet("color:#e57373;")
        self.b_keep.clicked.connect(self._accept_keep)
        self.b_drop.clicked.connect(self._confirm_discard)
        lay.addWidget(bb)
        self.b_keep.setDefault(True); self.b_keep.setFocus()

    def _accept_keep(self):
        if self.rb_mark.isChecked():
            self.choice = ("marker", float(self.sp_mark.value()))
        elif self.rb_last.isChecked():
            self.choice = ("last", float(self.sp_last.value()) * 60.0)
        else:
            self.choice = ("all", None)
        self.accept()

    def _confirm_discard(self):
        # 破棄は取り消せないので一度だけ確認する
        box = QMessageBox(QMessageBox.Icon.Warning, "記録の破棄",
                          f"{self.rec.n_frames:,} フレームの記録を削除します。元に戻せません。",
                          QMessageBox.StandardButton.Cancel, self)
        yes = box.addButton("破棄する", QMessageBox.ButtonRole.DestructiveRole)
        box.exec()
        if box.clickedButton() is yes:
            self.choice = None
            self.accept()

    def closeEvent(self, ev):
        # ウィンドウを閉じただけでは消さない（判断を保留＝そのまま残す）
        if self.choice is None and self.result() != QDialog.DialogCode.Accepted:
            self.choice = ("all", None)
        super().closeEvent(ev)


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

    sig_recorder = pyqtSignal(object)      # 記録開始を UI に伝える（グリッドが確定してから作る）

    def __init__(self, source: rt_source.FrameSource, ref_gray: np.ndarray,
                 tracker_kwargs: dict, rec_opts: Optional[dict] = None, parent=None):
        super().__init__(parent)
        self.source = source
        self.ref_gray = ref_gray
        self.kw = tracker_kwargs
        self.rec_opts = rec_opts        # None なら記録しない
        self.recorder: Optional[rt_record.Recorder] = None
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
        # 記録は開始と同時に無条件で始める（残すかどうかは停止時に決める）
        if self.rec_opts is not None:
            try:
                meta = dict(self.rec_opts.get("meta", {}))
                meta.update({"backend": dic_rt.backend_name(tr.use_gpu), "subset": tr.subset,
                             "step": tr.step, "search_stages": list(tr.stages),
                             "zncc_threshold": tr.zncc_thr, "reref_zncc": tr.reref_zncc,
                             "icgn": bool(tr.icgn)})
                self.recorder = rt_record.Recorder(
                    self.rec_opts["base_dir"], self.ref_gray, tr.X, tr.Y, meta,
                    fps=self.rec_opts.get("fps", 30.0),
                    video_budget_gb=self.rec_opts.get("video_budget_gb", 10.0),
                    video=self.rec_opts.get("video", True))
                self.sig_log.emit(f"記録開始（一時）: {self.recorder.dir.name}")
            except Exception as e:
                self.recorder = None
                self.sig_error.emit(f"記録を開始できません: {e}")
        self.sig_recorder.emit(self.recorder)
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
            if self.recorder is not None:
                try:
                    self.recorder.add_frame(elapsed, res, self.source.fps, proc_fps)
                except Exception as e:
                    self.sig_error.emit(f"記録エラー: {e}")
                    self.recorder = None
            self.sig_result.emit(out)
        if self.recorder is not None:
            self.recorder.close()      # ファイルを閉じるだけ。残すかは UI 側で決める
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
        self._store = None                                # 設定の保存先（main.py が登録する）
        self._focus_seq = None                            # フォーカス操作の結果待ち
        self._rec: Optional[rt_record.Recorder] = None    # 解析中の一時記録
        self._rec_t = 0.0                                 # 記録中表示の更新間引き用
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
        # フォーカス固定は解析中でも押せる（取得スレッドがコマンドを適用するので中断されない）
        self.btn_focus = QPushButton(FOCUS_TEXT[False])
        self.btn_focus.setObjectName("btn_focus")
        self.btn_focus.setCheckable(True)
        self.btn_focus.setMinimumWidth(150)      # 文言が変わっても幅が動かないように
        self.btn_focus.setToolTip("今のピントで固定する（オートフォーカスを切る）。"
                                  "解析中でも押せる。もう一度押すとオートフォーカスに戻る")
        self.btn_focus.toggled.connect(self._toggle_focus_lock)
        row1.addWidget(self.btn_focus)
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
        # サブセット・ステップがスペックルに対して適切かをプレビュー上で確かめるための表示
        self.chk_subset = QCheckBox("サブセット表示")
        self.chk_subset.setToolTip("解析点とサブセットの実寸を重ねて表示する。"
                                   "スペックルがサブセット内に十分入っているか確認する")
        self.chk_subset.toggled.connect(self._update_overlays)
        row2.addWidget(self.chk_subset)
        self.cb_subset_den = QComboBox()
        for name, _ in SUBSET_DENSITY:
            self.cb_subset_den.addItem(name)
        self.cb_subset_den.setToolTip("枠を描く点の割合。サブセットがステップより大きいと枠が重なって"
                                      "画像が見えなくなるので、間引いて実寸を見る")
        self.cb_subset_den.currentIndexChanged.connect(lambda _i: self._update_overlays())
        row2.addWidget(self.cb_subset_den)
        row2.addSpacing(12)
        row2.addWidget(QLabel("探索 ±"))
        self.sp_r1 = QSpinBox(); self.sp_r1.setRange(1, 100); self.sp_r1.setValue(10); self.sp_r1.setFixedWidth(46)
        self.sp_r1.setToolTip("探索半径 [px]。予測位置を中心に全点をこの範囲で探索する")
        self.sp_r1.valueChanged.connect(lambda _v: (self._update_params_label(), self._update_overlays()))
        # GPU では第 2 段は FFT 相関で ±50 全点でも 16 ms なので既定を広げる（CPU は計算量上限で省略されるため 30 のまま）
        self.sp_r2 = QSpinBox(); self.sp_r2.setRange(0, 200); self.sp_r2.setFixedWidth(46)
        self.sp_r2.setValue(50 if dic_rt.gpu_available() else 30)
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
        # ひずみマップ自体の ON/OFF（サブセット表示と同じ扱い）。OFF にすると素の映像の上に
        # 十字だけが出るので、追従の様子だけを見せられる
        self.chk_map = QCheckBox("マップ表示")
        self.chk_map.setChecked(True)
        self.chk_map.setToolTip("ひずみマップの重ね描きを切り替える。OFF ならカメラ映像だけになる")
        self.chk_map.toggled.connect(self._request_redraw)
        row3.addWidget(self.chk_map)
        row3.addSpacing(8)
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
        row3.addSpacing(12)
        # 別ウィンドウ表示（第 2 モニタ・プロジェクタ用）。向きはウィンドウ側でも変えられる
        self.btn_ext = QPushButton("別ウィンドウで表示")
        self.btn_ext.setCheckable(True)
        self.btn_ext.setToolTip("カメラ画像＋ひずみマップを別ウィンドウに表示する（F11 で全画面、向きは縦 90° 回転も可）")
        self.btn_ext.toggled.connect(self._toggle_ext_window)
        row3.addWidget(self.btn_ext)
        self.cb_rot = QComboBox()
        for name, _ in ROTATIONS:
            self.cb_rot.addItem(name)
        self.cb_rot.setToolTip("別ウィンドウでの表示の向き。引張試験機の縦向き試験片を縦長モニタに映すときは 90° 回転")
        self.cb_rot.setEnabled(False)
        row3.addWidget(self.cb_rot)
        self._ext: Optional[RTDisplayWindow] = None
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
        # clicked は checked(=False) をスロットに渡すので、そのまま繋ぐと _stop(ask=False) になり
        # 停止時の保存ダイアログが出ないまま保存されてしまう。引数を捨ててから呼ぶ
        self.btn_stop.clicked.connect(lambda: self._stop())
        row4.addWidget(self.btn_stop)
        # 記録は開始と同時に無条件で始まり、残すかどうかは停止時に決める（rt_record の設計）。
        # 試験中にできるのは「ここ」の印を付けることだけ
        self.btn_mark = QPushButton("★ マーカー (M)")
        self.btn_mark.setToolTip("「ここ」の印を記録に残す。停止後にマーカー前後だけ保存できる")
        self.btn_mark.setEnabled(False)
        self.btn_mark.clicked.connect(self._mark)
        row4.addWidget(self.btn_mark)
        QShortcut(QKeySequence("M"), self, activated=self._mark)
        self.lbl_rec = QLabel("")
        self.lbl_rec.setStyleSheet("color:#e57373;font-size:10px;")
        row4.addWidget(self.lbl_rec)
        row4.addStretch()
        lay.addLayout(row4)

        # ---- 行5：記録設定 ----
        row5 = QHBoxLayout()
        self.chk_record = QCheckBox("記録する")
        self.chk_record.setChecked(True)
        self.chk_record.setToolTip("解析開始と同時に一時フォルダへ記録し、停止時に残すか決める")
        row5.addWidget(self.chk_record)
        self.chk_rec_video = QCheckBox("映像も")
        self.chk_rec_video.setChecked(True)
        self.chk_rec_video.setToolTip("表示そのままの映像を mp4 で記録（約 1.2 MB/s）。数値データは常に全フレーム記録する")
        row5.addWidget(self.chk_rec_video)
        row5.addWidget(QLabel("映像の上限:"))
        self.sp_budget = QDoubleSpinBox(); self.sp_budget.setRange(0.5, 500); self.sp_budget.setValue(10.0)
        self.sp_budget.setDecimals(1); self.sp_budget.setSuffix(" GB"); self.sp_budget.setFixedWidth(80)
        self.sp_budget.setToolTip("一時記録中の映像がこれを超えたら古い方から捨てる（直近は必ず残る）")
        row5.addWidget(self.sp_budget)
        row5.addWidget(QLabel("保存先:"))
        self.edit_dir = QLineEdit()
        self.edit_dir.setPlaceholderText(str(pathlib.Path.home() / "LiveDIC_RT"))
        self.edit_dir.setMinimumWidth(1)
        row5.addWidget(self.edit_dir, stretch=1)
        btn_dir = QPushButton("…"); btn_dir.setFixedWidth(28)
        btn_dir.clicked.connect(self._browse_dir)
        row5.addWidget(btn_dir)
        self.chk_record.toggled.connect(lambda on: [w.setEnabled(on) for w in
                                                    (self.chk_rec_video, self.sp_budget, self.edit_dir, btn_dir)])
        lay.addLayout(row5)

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
        # 描画は新しい結果があるときだけ行うので、周期を短くしても負荷は結果レート以上には増えない。
        # 33 ms だと解析結果（~27 fps）との位相ずれで空振りが出て画面更新が 20 fps に落ちた
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

        self._populate_sources([])
        self._apply_scale_index(self.cb_scale.currentIndex())

    def register_settings(self, store):
        """次回起動時に復元する項目を登録する（main.py から呼ぶ）。"""
        self._store = store
        store.add_many(
            rt_source=self.cb_source, rt_res=self.cb_res, rt_gpu=self.chk_gpu,
            rt_r1=self.sp_r1, rt_r2=self.sp_r2, rt_zncc=self.sp_zncc, rt_reref=self.sp_reref,
            rt_subset_view=self.chk_subset, rt_subset_den=self.cb_subset_den, rt_map_view=self.chk_map,
            rt_rate_win=self.sp_rate_win, rt_rate_smooth=self.chk_rate_smooth,
            rt_pos_follow=self.rb_pos_follow,
            rt_cmap=self.cb_cmap, rt_alpha=self.sl_alpha, rt_rot=self.cb_rot,
            rt_record=self.chk_record, rt_rec_video=self.chk_rec_video,
            rt_budget=self.sp_budget, rt_dir=self.edit_dir,
        )
        # ウィジェットでない状態。ROI は画像サイズが変わると無効なので解像度ごとに覚える
        store.add_custom("rt_comp", lambda: self._comp, self._restore_comp)
        store.add_custom("rt_scale", self._scale_state, self._restore_scale)
        store.add_custom("rt_roi", lambda: {"res": self.cb_res.currentText(),
                                            "roi": list(self._roi) if self._roi else None},
                         self._restore_roi)

    def _restore_comp(self, key: str):
        for rb in self._comp_bg.buttons():
            if rb.text() == (STRAIN_LABELS.get(key) or RATE_LABELS.get(key)):
                rb.setChecked(True)      # toggled → _set_comp が走る
                return

    def _scale_state(self) -> dict:
        # プリセットを増減しても意味が変わらないよう、位置ではなく名前で覚える
        return {"name": SCALE_PRESETS[min(self.cb_scale.currentIndex(), len(SCALE_PRESETS) - 1)][0]
                if not self._is_rate() else RATE_PRESETS[self.cb_scale.currentIndex()][0],
                "custom": {str(int(k)): list(v) for k, v in self._custom_pct.items()}}

    def _restore_scale(self, s: dict):
        for k, v in (s.get("custom") or {}).items():
            self._custom_pct[bool(int(k))] = (float(v[0]), float(v[1]))
        names = [n for n, _, _ in self._presets()]
        idx = names.index(s["name"]) if s.get("name") in names else 1
        self.cb_scale.setCurrentIndex(idx)
        self._apply_scale_index(idx)

    def _restore_roi(self, s: dict):
        # 解像度が違うときの ROI は座標の意味が変わるので捨てる
        if s and s.get("roi") and s.get("res") == self.cb_res.currentText():
            self._on_roi(*[int(v) for v in s["roi"]])

    def bind_host(self):
        """host 側の設定パネル（sp_subset / sp_step）が構築された後に main.py から呼ぶ。"""
        self.host.sp_subset.valueChanged.connect(lambda _v: self._update_overlays())
        self.host.sp_step.valueChanged.connect(lambda _v: self._update_overlays())
        self.host._append_log(f"リアルタイムマップ: バックエンド {dic_rt.backend_name()}")
        if not dic_rt.gpu_available() and dic_rt.gpu_error():
            # GPU があるのに使えない場合は理由を必ず見せる（ツールチップだけでは気づかれない）
            self.host._append_log(f"[WARN] GPU 利用不可のため CPU で動作します: {dic_rt.gpu_error()}")
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
            # 前回使ったカメラがあればそれを優先。無ければ外付け（内蔵らしくない名前）を選ぶ
            # ノート PC では内蔵が 0 番に来るので「最初のカメラ」だと実験用が選ばれない
            saved = self._store.get("rt_source") if self._store is not None else None
            k = self.cb_source.findText(saved) if saved else -1
            if k < 0:
                pick = rt_source.pick_default_camera(cams)
                k = next((i for i in range(self.cb_source.count())
                          if (self.cb_source.itemData(i) or (None, None))[1] == pick), 0)
            self.cb_source.setCurrentIndex(k)
            self._start_source()

    def showEvent(self, ev):
        super().showEvent(ev)
        if not self._scanned:
            self._scanned = True
            QTimer.singleShot(0, self.check_stale_recordings)
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
        # 開き直したカメラはオートフォーカスから始まるのでボタンも戻す
        self._set_focus_button(False)
        self._focus_seq = None
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
        self._update_params_label()
        self._update_overlays()

    def _draw_roi_overlay(self):
        self._update_overlays()

    def _density_mask(self, shape) -> np.ndarray:
        """枠を描く点のマスク。いずれも ROI 全体に等方的に散るように選ぶ。"""
        ny, nx = shape
        iy, ix = np.mgrid[0:ny, 0:nx]
        den = SUBSET_DENSITY[self.cb_subset_den.currentIndex()][1]
        if den == 2:
            return (ix + iy) % 2 == 0                               # 市松：ちょうど半分
        if den == 4:
            return (ix % 2 == 0) & (iy % 2 == 0)                    # 1 つおき
        if den == 8:
            return (ix % 2 == 0) & (iy % 2 == 0) & ((ix // 2 + iy // 2) % 2 == 0)
        return np.ones((ny, nx), bool)

    def _mark_arm(self, S: int) -> float:
        """
        十字の腕の長さ [画像 px]。

        サブセットの実寸（S/2）を出したいが、ステップが小さい設定では隣の十字と重なって
        一面の網目になる（subset 31 / step 10 で実際にそうなった）。描く点どうしの最小間隔の
        4 割を上限にして、どの設定でも十字が離れて見えるようにする。
        """
        step = max(1, self.host.sp_step.value())
        den = SUBSET_DENSITY[self.cb_subset_den.currentIndex()][1]
        gap = {1: step, 2: step * 1.414, 4: step * 2.0, 8: step * 2.828}[den]
        return max(2.0, min(S / 2.0, 0.4 * gap))

    def _subset_overlay(self, PX: np.ndarray, PY: np.ndarray, S: int,
                        good: Optional[np.ndarray] = None) -> dict:
        """
        キャンバス用のオーバーレイ 1 件。**表示倍率で描かせる**ためにデータだけ渡す。
        画像に焼き込んでから縮小すると 1px の線が潰れて網目になるため。
        """
        m = self._density_mask(PX.shape)
        fin = np.isfinite(PX) & np.isfinite(PY)
        g = fin if good is None else (fin & good)

        def pack(sel):
            return [(int(x), int(y)) for x, y in zip(PX[sel].ravel(), PY[sel].ravel())]
        return {"type": "subsets", "arm": self._mark_arm(S), "color": SUBSET_COLOR,
                "points": pack(g), "marks": pack(m & g),
                "lost_points": pack(fin & ~g), "lost_marks": pack(m & fin & ~g),
                "lost_color": SUBSET_LOST_COLOR}

    def _draw_marks_on(self, rgb: np.ndarray, PX: np.ndarray, PY: np.ndarray,
                       S: int, good: Optional[np.ndarray] = None) -> None:
        """
        サブセットの十字を画像そのものに焼き込む（in-place）。

        **別ウィンドウと録画用のみ**。どちらも原寸に近い大きさで見るので線が潰れない。
        タブ内のキャンバスは縮小表示なので `_subset_overlay` を使う。
        """
        h, w = rgb.shape[:2]
        m = self._density_mask(PX.shape)
        fin = np.isfinite(PX) & np.isfinite(PY)
        g = fin if good is None else (fin & good)
        half = max(1, int(round(self._mark_arm(S))))
        # 縮小表示されても線が消えないよう、画像サイズに比例した太さにする
        # （1080p で 2px。0.5 倍に縮小しても 1px 残る。4K なら 4px）
        thick = max(2, int(round(min(h, w) / 540.0)))
        for sel, col in ((m & g, SUBSET_RGB), (m & fin & ~g, SUBSET_LOST_RGB)):
            if not sel.any():
                continue
            x = np.clip(PX[sel], -1e4, 1e4).astype(np.int32)
            y = np.clip(PY[sel], -1e4, 1e4).astype(np.int32)
            segs = np.empty((2 * x.size, 2, 2), np.int32)
            segs[0::2, 0, 0] = x - half; segs[0::2, 0, 1] = y        # 横棒
            segs[0::2, 1, 0] = x + half; segs[0::2, 1, 1] = y
            segs[1::2, 0, 0] = x; segs[1::2, 0, 1] = y - half        # 縦棒
            segs[1::2, 1, 0] = x; segs[1::2, 1, 1] = y + half
            cv2.polylines(rgb, segs, False, col, thick, cv2.LINE_8)
        # 解析点そのものは全点を 1px の点で（十字を間引いても点の密度が分かる）
        px = PX[g]; py = PY[g]
        if px.size:
            xi = np.clip(px.astype(np.int32), 0, w - 1)
            yi = np.clip(py.astype(np.int32), 0, h - 1)
            rgb[yi, xi] = SUBSET_RGB

    def _preview_grid(self, shape):
        """プレビュー時に十字を描くグリッド（解析前なので参照位置そのもの）。"""
        try:
            X, Y = dic_core.create_grid(shape, self.host.sp_subset.value() | 1,
                                        self.sp_r1.value(), self.host.sp_step.value(), self._roi)
        except Exception:
            return None, None
        if X.size == 0:
            return None, None
        return X.astype(np.float64), Y.astype(np.float64)

    def _reset_canvas_overlays(self):
        """キャンバスのオーバーレイを ROI 枠だけに戻す（十字はこの後に積む）。"""
        self.canvas.clear_overlays()
        if self._roi:
            x1, y1, x2, y2 = self._roi
            self.canvas.add_overlay({"type": "rect", "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                                     "color": "#ffaa00"})

    def _update_overlays(self):
        """設定が変わったときに呼ぶ。次の描画で十字を引き直させる。"""
        self._reset_canvas_overlays()
        self._latest_drawn_fid = -1
        self._preview_fid = -1

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

    def _set_focus_button(self, locked: bool):
        """ボタンの見た目を状態に合わせる（文言＋色。色は QSS の #btn_focus:checked）。"""
        self.btn_focus.blockSignals(True)
        self.btn_focus.setChecked(locked)
        self.btn_focus.blockSignals(False)
        self.btn_focus.setText(FOCUS_TEXT[locked])

    def _toggle_focus_lock(self, on: bool):
        src = self._source
        if not isinstance(src, rt_source.CameraSource):
            self.host._append_log("フォーカス固定: カメラが開いていません")
            self._set_focus_button(False)
            return
        self.btn_focus.setText("… 切替中")
        self._focus_seq = src.focus_seq()      # 結果が返ったら _tick でログに出す
        src.lock_focus() if on else src.unlock_focus()

    def _poll_focus(self):
        """取得スレッドがフォーカス操作を終えていたら結果を知らせる。"""
        src = self._source
        if not isinstance(src, rt_source.CameraSource) or self._focus_seq is None:
            return
        if src.focus_seq() == self._focus_seq:
            return
        self._focus_seq = None
        st = src.focus_state
        self.host._append_log(f"フォーカス: {st.get('msg', '')}")
        # 効かなかったら解除状態に戻す（固定できていないのに固定表示は紛らわしい）
        self._set_focus_button(bool(st.get("supported")) and bool(st.get("locked")))

    def _browse_dir(self):
        d = QFileDialog.getExistingDirectory(self, "記録の保存先を選択", self.edit_dir.text().strip())
        if d:
            self.edit_dir.setText(d)

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
            # ひずみ [%] と速度 [%/s] でプリセット一覧を入れ替える（選択位置は引き継ぐ）。
            # プリセットの数が違うので、カスタムだったかどうかで分けてから丸める
            was_custom = self.cb_scale.currentIndex() == len(
                (RATE_PRESETS if was_rate else SCALE_PRESETS)) - 1
            idx = self.cb_scale.currentIndex()
            self.cb_scale.blockSignals(True)
            self.cb_scale.clear()
            for name, _, _ in self._presets():
                self.cb_scale.addItem(name)
            idx = self._custom_index() if was_custom else min(idx, self._custom_index() - 1)
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
        tmin, tmax = _fmt_scale(vmin * 100, rate), _fmt_scale(vmax * 100, rate)
        self.lbl_cb_min.setText(tmin)
        self.lbl_cb_max.setText(tmax)
        if self._ext is not None:
            self._ext.set_colorbar(rgb, tmin, tmax)

    def _request_redraw(self):
        self._latest_drawn_fid = -1
        self._update_colorbar()

    def _reset_view(self):
        """ソース切替・解析開始時に描画済みフレームの記録をリセットする。"""
        self._latest_drawn_fid = -1
        self._preview_fid = -1

    # ------------------------------------------------------------------
    # 別ウィンドウ表示
    # ------------------------------------------------------------------
    def _toggle_ext_window(self, on: bool):
        if on:
            if self._ext is None:
                self._ext = RTDisplayWindow(self)
                self._ext.closed.connect(lambda: self.btn_ext.setChecked(False))
                self._ext.cb_rot.currentIndexChanged.connect(
                    lambda i: (self.cb_rot.blockSignals(True), self.cb_rot.setCurrentIndex(i),
                               self.cb_rot.blockSignals(False)))
                self.cb_rot.currentIndexChanged.connect(self._ext.set_rotation_index)
            self._ext.set_rotation_index(self.cb_rot.currentIndex())
            self._update_colorbar()
            self._ext.set_status(self.lbl_status.text())
            self._ext.show(); self._ext.raise_()
            self.cb_rot.setEnabled(True)
            self._latest_drawn_fid = -1; self._preview_fid = -1     # 直近フレームを送り直す
        else:
            if self._ext is not None and self._ext.isVisible():
                self._ext.hide()
            self.cb_rot.setEnabled(False)

    def _show_rgb(self, rgb: np.ndarray, t: Optional[float] = None, marks: Optional[tuple] = None):
        """
        タブ内キャンバスに表示し、別ウィンドウと記録にも送る。

        タブ内は縮小表示なので十字をキャンバスのオーバーレイ（表示倍率で描画）に任せ、
        別ウィンドウと録画へは十字を焼き込んだ画像を渡す。
        """
        self.canvas.set_image(rgb_to_qpixmap(rgb))
        need_ext = self._ext is not None and self._ext.isVisible()
        need_rec = self._rec is not None and t is not None
        if not (need_ext or need_rec):
            return
        out = rgb
        if marks is not None:
            out = rgb.copy()
            self._draw_marks_on(out, *marks)
        if need_ext:
            self._ext.set_frame(out)
        if need_rec:
            self._rec.add_display(out, t)      # 書き込みは別スレッド（呼び出しは軽い）

    def _set_status(self, text: str):
        self.lbl_status.setText(text)
        if self._ext is not None and self._ext.isVisible():
            self._ext.set_status(text)

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
            self._worker = RTWorker(self._source, self._ref_gray, self._tracker_kwargs(),
                                    self._rec_opts(), self)
        except Exception as e:
            self.lbl_status.setText(f"開始エラー: {e}")
            return
        self._worker.sig_result.connect(self._on_result)
        self._worker.sig_log.connect(self.host._append_log)
        self._worker.sig_error.connect(lambda m: self.host._append_log(f"[ERROR] {m}"))
        self._worker.sig_recorder.connect(self._on_recorder)
        self._worker.start()
        self._latest = None
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_mark.setEnabled(True)
        self._update_overlays()       # 次の描画で追従位置の十字に切り替える
        for wdg in (self.btn_roi, self.btn_roi_clear, self.sp_r1, self.sp_r2,
                    self.sp_zncc, self.sp_reref, self.chk_gpu, self.btn_scan,
                    self.cb_source, self.cb_res, self.sp_rate_win, self.chk_rate_smooth, self.host.chk_icgn):
            wdg.setEnabled(False)
        self.host._append_log(f"参照取得: フレーム {fid}")
        # 試験中にクラッシュしても設定が残るよう、開始時にも保存しておく
        if hasattr(self.host, "_save_settings"):
            self.host._save_settings()

    def _stop(self, ask: bool = True):
        if self._worker is None:
            return
        self._worker.stop()
        self._worker.wait(5000)
        self._worker = None
        self.btn_start.setEnabled(self._source is not None)
        self.btn_stop.setEnabled(False)
        self.btn_mark.setEnabled(False)
        for wdg in (self.btn_roi, self.btn_roi_clear, self.sp_r1, self.sp_r2,
                    self.sp_zncc, self.sp_reref, self.btn_scan,
                    self.cb_source, self.cb_res, self.sp_rate_win, self.chk_rate_smooth, self.host.chk_icgn):
            wdg.setEnabled(True)
        self.chk_gpu.setEnabled(dic_rt.gpu_available())
        self.lbl_status.setText("解析停止")
        self.lbl_rec.setText("")
        self._update_overlays()
        self._finish_recording(ask)

    # ------------------------------------------------------------------
    # 記録（開始時は無条件、停止時に残すか決める）
    # ------------------------------------------------------------------
    def _rec_opts(self) -> Optional[dict]:
        if not self.chk_record.isChecked():
            return None
        base = self.edit_dir.text().strip() or str(pathlib.Path.home() / "LiveDIC_RT")
        self.edit_dir.setText(base)
        src_fps = self._source.fps if (self._source is not None and self._source.fps > 1) else 30.0
        return {"base_dir": base, "fps": src_fps, "video": self.chk_rec_video.isChecked(),
                "video_budget_gb": float(self.sp_budget.value()),
                "meta": {"camera": self.cb_source.currentText(),
                         "resolution": list(RESOLUTIONS[self.cb_res.currentIndex()]),
                         "roi": list(self._roi) if self._roi else None}}

    def _on_recorder(self, rec):
        self._rec = rec

    def _mark(self):
        if self._rec is not None and self._worker is not None:
            self._rec.mark()
            self.host._append_log(f"★ マーカー {len(self._rec.markers)}（t={self._rec.t_last:.1f}s）")

    def _finish_recording(self, ask: bool):
        rec, self._rec = self._rec, None
        if rec is None:
            return
        rec.close()
        if rec.n_frames == 0:
            rec.discard(); return
        if not ask:
            rec.keep(); return
        dlg = _SaveDialog(rec, self)
        dlg.exec()
        try:
            if dlg.choice is None:
                rec.discard()
                self.host._append_log("記録を破棄しました")
                return
            kind, val = dlg.choice
            t0 = t1 = None
            if kind == "marker" and rec.markers:
                t0 = max(0.0, min(m[1] for m in rec.markers) - val)
                t1 = max(m[1] for m in rec.markers) + val
            elif kind == "last":
                t1 = rec.t_last
                t0 = max(0.0, rec.t_last - val)
            dest = rec.keep(t0, t1)
            self.host._append_log(f"記録を保存: {dest}（{rt_record.session_bytes(dest)/1e6:,.0f} MB）")
        except Exception as e:
            self.host._append_log(f"[ERROR] 記録の保存に失敗: {e}（フォルダは {rec.dir} に残っています）")

    def check_stale_recordings(self):
        """起動時に、前回クラッシュして残った記録を見つけたら残すか聞く（自動では消さない）。"""
        base = self.edit_dir.text().strip() or str(pathlib.Path.home() / "LiveDIC_RT")
        for p in rt_record.find_stale(base):
            mb = rt_record.session_bytes(p) / 1e6
            box = QMessageBox(QMessageBox.Icon.Question, "前回の記録が残っています",
                              f"{p.name}\n{mb:,.0f} MB\n\n前回の解析が正常に終了しなかったようです。"
                              f"\nこの記録を残しますか？", parent=self)
            b_keep = box.addButton("残す", QMessageBox.ButtonRole.AcceptRole)
            box.addButton("削除", QMessageBox.ButtonRole.DestructiveRole)
            box.exec()
            dest = rt_record.recover(str(p), box.clickedButton() is b_keep)
            self.host._append_log(f"前回の記録: {p.name} → " + (f"{dest}" if dest else "削除"))

    def shutdown(self):
        """アプリ終了時に呼ぶ。"""
        self._timer.stop()
        # 終了時は確認ダイアログを出さずに残す（閉じる操作で記録を失わせない）
        self._stop(ask=False)
        self._stop_source()
        if self._ext is not None:
            self._ext.close()

    # ------------------------------------------------------------------
    # 結果受信・描画
    # ------------------------------------------------------------------
    def _on_result(self, out: dict):
        self._latest = out

    def _tick(self):
        self._poll_focus()
        if self._worker is not None and self._latest is not None:
            out = self._latest
            if out["fid"] == self._latest_drawn_fid:
                return
            self._latest_drawn_fid = out["fid"]
            self._draw_result(out["gray"], out["res"], out["elapsed"])
            st = out["res"]["stats"]
            if self._rec is not None and out["elapsed"] - self._rec_t >= 1.0:
                self._rec_t = out["elapsed"]
                self.lbl_rec.setText("● " + self._rec.status())
            self._set_status(
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
            rgb = cv2.cvtColor(fr[2], cv2.COLOR_GRAY2RGB)
            marks = None
            if self.chk_subset.isChecked():
                gx, gy = self._preview_grid(fr[2].shape)
                if gx is not None:
                    marks = (gx, gy, self.host.sp_subset.value() | 1, None)
                    self._reset_canvas_overlays()
                    self.canvas.add_overlay(self._subset_overlay(*marks))
            self._show_rgb(rgb, None, marks)
            if self._source.shape is not None and self.lbl_params.text().startswith("subset=") \
                    and "点" not in self.lbl_params.text():
                self._update_params_label()
            self._set_status(f"プレビュー中: {fr[2].shape[1]}×{fr[2].shape[0]}  {self._source.fps:.1f} fps"
                             f"  — ROI を設定して「参照取得＆解析開始」")

    def _draw_result(self, gray: np.ndarray, res: dict, t: Optional[float] = None):
        data = res.get(self._comp)
        vmin, vmax = self._scale()
        alpha = int(self.sl_alpha.value() * 2.55)
        rgba = None
        if data is not None and self.chk_map.isChecked():
            follow = self.rb_pos_follow.isChecked() and res.get("U") is not None
            rgba = dic_rt.strain_to_rgba(data, vmin, vmax, res["X"], res["Y"], gray.shape, self._lut, alpha,
                                         U=res.get("U_draw", res["U"]) if follow else None,
                                         V=res.get("V_draw", res["V"]) if follow else None)
        rgb = dic_rt.compose_overlay(gray, rgba)
        marks = None
        if self.chk_subset.isChecked():
            # トラッキングの様子を見るためのものなので、参照位置ではなく**追従後の位置**に描く。
            # 見失った点は直近の追従位置（U_draw）に別色で残すので、どこで外れたかが分かる
            U = res.get("U_draw", res["U"]); V = res.get("V_draw", res["V"])
            S = self.host.sp_subset.value() | 1
            good = np.isfinite(res["U"]) & np.isfinite(res["V"])
            marks = (res["X"] + U, res["Y"] + V, S, good)
            self._reset_canvas_overlays()
            self.canvas.add_overlay(self._subset_overlay(*marks))
        self._show_rgb(rgb, t, marks)


class _ScanThread(QThread):
    done = pyqtSignal(list)

    def run(self):
        # 列挙も open テストもカメラがハングしていると固まるので、必ずこのスレッドで行う
        try:
            self.done.emit(rt_source.list_cameras_named())
        except Exception:
            self.done.emit([])
