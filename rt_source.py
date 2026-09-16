"""
rt_source.py  —  LiveDIC リアルタイムマップモード用フレームソース

  CameraSource     : USBカメラ（cv2.VideoCapture）を別スレッドで連続取得し、最新フレームだけ保持
  VideoFileSource  : 動画ファイルを実時間ペース（ファイルのfps）で再生してカメラの代わりにする
  SyntheticSource  : スペックル画像に時間変化する変形を与える合成ソース（カメラ無しの動作確認・デモ用）

いずれも latest() で (frame_id, timestamp, gray_u8) を返す。取得は常に最大レートで行い、
消費側（DIC）が追いつかなければ古いフレームは自然に捨てられる（間引き）。
"""
from __future__ import annotations
import os
import sys
import re
import time
import atexit
import threading
import weakref
from typing import Optional, Tuple, List
import numpy as np
import cv2
from scipy.special import erf

try:
    # DirectShow のデバイス名を OpenCV の CAP_DSHOW と同じ順で列挙する（Windows のみ、任意依存）
    if os.environ.get("LIVEDIC_NO_DSHOW_NAMES"):
        raise ImportError("disabled by LIVEDIC_NO_DSHOW_NAMES")
    import comtypes as _comtypes
    from pygrabber.dshow_graph import FilterGraph as _DShowFilterGraph
except Exception:       # 未インストール / 非 Windows / comtypes 不在
    _comtypes = None
    _DShowFilterGraph = None

Frame = Tuple[int, float, np.ndarray]   # (frame_id, timestamp, gray uint8)

# 取得スレッドが動いたままプロセスが終わると、cv2.VideoCapture が release されず
# DirectShow 側でデバイスが掴まれたままになり、次に開けなくなる（400-CAM106 で実測。
# USB の抜き差しで復帰）。生きているソースを弱参照で記録し、インタプリタ終了時に必ず止める。
_LIVE_SOURCES: "weakref.WeakSet[FrameSource]" = weakref.WeakSet()


def stop_all_sources(pump=None, timeout: float = 10.0):
    """
    生きている全ソースを止める。GUI からは pump=QApplication.processEvents を渡すこと。

    DirectShow の停止（cv2.VideoCapture.release）は一部が COM のメインスレッド（STA）経由で
    処理されるため、Qt のイベントループが終わった後（app.exec() 復帰後・atexit）に止めると
    release() 自体は返ってくるのにデバイスへ停止が届かず、以後そのカメラが開けなくなる
    （400-CAM106 で複数回再現、USB 抜き差しで復帰）。停止待ちの間に pump でメッセージを
    回し続ければ届く。
    """
    srcs = list(_LIVE_SOURCES)
    if not srcs:
        return
    for src in srcs:
        src._stop.set()
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline and any(s.running() for s in srcs):
        if pump is not None:
            try:
                pump()
            except Exception:
                pass
        time.sleep(0.01)
    for src in srcs:
        try:
            src.stop()
        except Exception:
            pass
    if pump is not None and any(isinstance(s, CameraSource) for s in srcs):
        # 解放後もしばらくメッセージを回す（停止完了の通知がデバイス／ドライバまで届くのを待つ）
        t_end = time.perf_counter() + 0.6
        while time.perf_counter() < t_end:
            try:
                pump()
            except Exception:
                pass
            time.sleep(0.01)


def _stop_all_sources():
    stop_all_sources(pump=None)


atexit.register(_stop_all_sources)


class FrameSource:
    """フレームソース共通インタフェース。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._latest: Optional[Frame] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._fps_count = 0
        self._fps_t0 = time.perf_counter()
        self.fps = 0.0
        self.error: Optional[str] = None
        self.shape: Optional[Tuple[int, int]] = None

    def start(self):
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        _LIVE_SOURCES.add(self)

    def stop(self):
        self._stop.set()
        th = self._thread
        if th is not None:
            th.join(timeout=3.0)
            if th.is_alive():
                # ここで諦めると cv2.VideoCapture が release されず、DirectShow がデバイスを
                # 掴んだままになる（次回 open 不能、USB 抜き差しが必要）。もう少し待つ
                sys.stderr.write(f"[rt_source] {type(self).__name__}: 取得スレッドが 3 s で止まりません。待機を延長\n")
                th.join(timeout=7.0)
                if th.is_alive():
                    sys.stderr.write(f"[rt_source] {type(self).__name__}: 取得スレッドが停止せず。カメラが解放されない可能性\n")
            self._thread = None
        _LIVE_SOURCES.discard(self)

    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def latest(self) -> Optional[Frame]:
        with self._lock:
            return self._latest

    def _publish(self, fid: int, gray: np.ndarray):
        now = time.perf_counter()
        with self._lock:
            self._latest = (fid, now, gray)
        self._fps_count += 1
        if now - self._fps_t0 >= 1.0:
            self.fps = self._fps_count / (now - self._fps_t0)
            self._fps_count = 0
            self._fps_t0 = now

    def _loop(self):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# USB カメラ
# ---------------------------------------------------------------------------

def _default_backend() -> int:
    # MSMF はこの環境でオープンに 12 秒以上かかった。DSHOW は 0.5 秒。
    return cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY


def list_cameras(max_index: int = 6) -> List[int]:
    """開けるカメラ番号を列挙する（1台あたり数百 ms かかるので GUI からは別スレッドで呼ぶ）。"""
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, _default_backend())
        ok = cap.isOpened()
        if ok:
            ok, _ = cap.read()
        cap.release()
        if ok:
            found.append(i)
    return found


def camera_names() -> List[str]:
    """
    DirectShow のデバイス名（OpenCV CAP_DSHOW のインデックス順）。取れない環境では []。

    呼び出しスレッドで COM を初期化してから列挙する（QThread など COM 未初期化のスレッドでは
    失敗して [] になるため）。**メインスレッドから呼ばないこと**: カメラがハングしていると
    DirectShow の列挙自体が数十秒〜無期限に固まり、GUI ごと止まる（400-CAM106 で実測）。
    """
    if _DShowFilterGraph is None or not sys.platform.startswith("win"):
        return []
    inited = False
    try:
        try:
            _comtypes.CoInitialize()
            inited = True
        except Exception:      # 既に別モードで初期化済み等。そのまま試す
            pass
        return [str(n) for n in _DShowFilterGraph().get_input_devices()]
    except Exception:
        return []
    finally:
        if inited:
            try:
                _comtypes.CoUninitialize()
            except Exception:
                pass


def list_cameras_named(max_index: int = 6, names: Optional[List[str]] = None) -> List[Tuple[int, str]]:
    """
    カメラの (番号, 名前) 一覧。GUI からは別スレッドで呼ぶ（列挙も open テストも固まり得る）。

    名前が取れたときは **デバイスを開かずに** DirectShow の列挙だけで一覧を作る。
    400-CAM106 はストリームの開始／停止を繰り返すと確率的（実測 2〜3 割/回）にファームウェアが
    ハングし、USB 抜き差しでしか復帰しない（pnputil /restart-device では消えたまま戻らない）ため、
    「検索で開いて 1 フレーム読んで閉じる → 本番でもう一度開く」の 2 回開きを避ける。
    名前が取れない環境（pygrabber 無し）だけ従来の open テストにフォールバック。
    """
    if names is None:
        names = camera_names()
    if names:
        return [(i, n) for i, n in enumerate(names)]
    return [(i, "") for i in list_cameras(max_index)]


# ノート PC 内蔵カメラらしい名前。これに当たらないカメラ（外付け）を既定で選ぶ
_BUILTIN_CAMERA_RE = re.compile(
    r"integrated|built-?in|internal|ir camera|infrared|facetime|surface|"
    r"^f?hd webcam$|^usb ?2\.0 (hd )?uvc webcam$",      # 各社ノート PC の典型名
    re.IGNORECASE)


def is_builtin_camera_name(name: str) -> bool:
    return bool(name) and _BUILTIN_CAMERA_RE.search(name.strip()) is not None


def pick_default_camera(cams: List[Tuple[int, str]]) -> Optional[int]:
    """
    既定で選ぶカメラ番号。外付け（内蔵らしくない名前）を優先し、なければ最後に見つかった番号。
    ノート PC では内蔵カメラが 0 番で最初に列挙されるため、単に 0 を選ぶと実験用カメラが選ばれない。
    """
    if not cams:
        return None
    external = [i for i, n in cams if n and not is_builtin_camera_name(n)]
    if external:
        return external[0]
    return cams[-1][0]


class CameraSource(FrameSource):
    def __init__(self, index: int = 0, width: int = 1920, height: int = 1080,
                 fps: int = 30, fourcc: str = "MJPG"):
        super().__init__()
        self.index = index
        self.req_w, self.req_h, self.req_fps, self.fourcc = width, height, fps, fourcc
        self.actual: Tuple[int, int, float] = (0, 0, 0.0)

    def _loop(self):
        cap = cv2.VideoCapture(self.index, _default_backend())
        if not cap.isOpened():
            self.error = f"カメラ {self.index} を開けません"
            return
        try:
            # 設定順序が重要。400-CAM106 は FOURCC→解像度 の順だと 1080p が非圧縮 YUY2（2.5fps）に
            # 落ちるが、fps→解像度→FOURCC の順なら MJPG 30fps が選ばれる（2026-09-16 実測）
            cap.set(cv2.CAP_PROP_FPS, self.req_fps)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.req_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.req_h)
            if self.fourcc:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
            self.actual = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                           int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                           float(cap.get(cv2.CAP_PROP_FPS)))
            fid = 0
            fail = 0
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    fail += 1
                    if fail > 30:
                        self.error = "カメラからフレームを取得できません"
                        break
                    time.sleep(0.01)
                    continue
                fail = 0
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
                self.shape = gray.shape
                self._publish(fid, gray)
                fid += 1
        finally:
            t0 = time.perf_counter()
            cap.release()
            dt = time.perf_counter() - t0
            if dt > 1.0:
                sys.stderr.write(f"[rt_source] カメラ {self.index} の release に {dt:.1f} s\n")


# ---------------------------------------------------------------------------
# 動画ファイルソース（撮影済み動画・デモ動画をカメラ代わりに再生）
# ---------------------------------------------------------------------------

class VideoFileSource(FrameSource):
    """
    動画ファイルをファイルの fps で実時間再生する。
    loop=True ならプレビュー用に末尾で先頭へ戻る。restart() で先頭から再生し直す
    （解析開始時に呼び、ループも解除して 1 回だけ再生する）。
    """

    def __init__(self, path: str, loop: bool = True):
        super().__init__()
        self.path = str(path)
        self.loop = loop
        self.file_fps = 0.0
        self.n_frames = 0
        self.finished = False
        self._restart = threading.Event()

    def restart(self, loop: Optional[bool] = None):
        if loop is not None:
            self.loop = loop
        self.finished = False
        self._restart.set()

    def _loop(self):
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            self.error = f"動画を開けません: {self.path}"
            return
        try:
            self.file_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
            self.n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            period = 1.0 / self.file_fps
            fid = 0
            next_t = time.perf_counter()
            while not self._stop.is_set():
                if self._restart.is_set():
                    self._restart.clear()
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    next_t = time.perf_counter()
                ok, frame = cap.read()
                if not ok or frame is None:
                    if self.loop:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        next_t = time.perf_counter()
                        continue
                    self.finished = True
                    # 末尾で停止：restart() か stop() を待つ
                    time.sleep(0.02)
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
                self.shape = gray.shape
                self._publish(fid, gray)
                fid += 1
                next_t += period
                delay = next_t - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_t = time.perf_counter()
        finally:
            cap.release()


# ---------------------------------------------------------------------------
# 合成変形ソース（デモ・検証用）
# ---------------------------------------------------------------------------

def make_speckle(h: int, w: int, seed: int = 0, sigma: float = 1.6) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), sigma)
    base = (base - base.min()) / (base.max() - base.min())
    return np.clip((base - 0.45) * 6 * 255 + 128, 0, 255).astype(np.uint8)


class SyntheticSource(FrameSource):
    """
    横長試験片を模した合成ソース。左端固定・右へ引張。
    一様ひずみ（rate [1/s]）＋中央やや右に局所ひずみ帯（くびれ）が成長する。
    ground truth の変位関数 u_of(X) は displacement_fn() で取得できる。
    """

    def __init__(self, width: int = 1920, height: int = 1080, fps: float = 30.0,
                 strain_rate: float = 0.002, band_gain: float = 3.0,
                 band_x: Optional[float] = None, band_w: float = 60.0,
                 noise: float = 2.0, seed: int = 0):
        super().__init__()
        self.w, self.h, self.target_fps = width, height, fps
        self.strain_rate = strain_rate
        self.band_gain = band_gain
        self.band_x = float(width * 0.62) if band_x is None else float(band_x)
        self.band_w = band_w
        self.noise = noise
        self.ref = make_speckle(height, width, seed)
        # 試験片形状：中央帯だけ模様、外は暗い背景
        y1, y2 = int(height * 0.33), int(height * 0.67)
        self.ref[:y1] = 30
        self.ref[y2:] = 30
        self.shape = (height, width)
        self.t_start: Optional[float] = None
        self._xs1d = np.arange(width, dtype=np.float64)
        self._ys = np.arange(height, dtype=np.float32)[:, None].repeat(width, 1)
        self.x_fix = float(width * 0.05)   # 固定端
        # ノイズはバンクを作っておきランダムにずらして使う（毎フレーム生成は重い）
        rng = np.random.default_rng(seed + 1)
        self._noise = rng.normal(0, 1.0, (height + 64, width + 64)).astype(np.float32) if noise > 0 else None

    def eps_at(self, t: float) -> float:
        return self.strain_rate * t

    def displacement_fn(self, t: float):
        eps = self.eps_at(t)
        amp = eps * self.band_gain
        bx, bw, xf = self.band_x, self.band_w, self.x_fix

        def u_of(Xg):
            return eps * (Xg - xf) + amp * bw * 0.5 * (1 + erf((Xg - bx) / bw))
        return u_of

    def render(self, t: float) -> np.ndarray:
        # 変位は X のみの関数なので逆写像は 1 次元で解いて行方向に複製する
        u_of = self.displacement_fn(t)
        xs = self._xs1d
        Xg = xs.copy()
        for _ in range(8):
            Xg = xs - u_of(Xg)
        map_x = np.ascontiguousarray(np.broadcast_to(Xg.astype(np.float32)[None, :], (self.h, self.w)))
        # 線形補間は小数位置でだけ画像がボケるため 1px 周期の擬似ロッキングを持ち込む。Lanczos で生成する
        img = cv2.remap(self.ref, map_x, self._ys, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
        if self._noise is not None:
            oy, ox = np.random.randint(0, 64, 2)
            n = self._noise[oy:oy + self.h, ox:ox + self.w] * self.noise
            img = np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)
        return img

    def _loop(self):
        self.t_start = time.perf_counter()
        fid = 0
        period = 1.0 / self.target_fps
        next_t = time.perf_counter()
        while not self._stop.is_set():
            t = time.perf_counter() - self.t_start
            self._publish(fid, self.render(t))
            fid += 1
            next_t += period
            delay = next_t - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.perf_counter()
