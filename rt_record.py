"""
rt_record.py  —  リアルタイムマップの記録（常時一時記録 → 停止時に残すか決める）

設計の要点（2026-11-17）:
  「録画するか」を**開始前ではなく停止後に決める**。解析を始めたら無条件に一時フォルダへ
  記録し、停止したときに初めて「残す / 一部だけ残す / 捨てる」を選ぶ。こうすると
    ・調整や試し解析のたびに録ってしまう（容量が膨らむ）
    ・録り忘れて本番を逃す（引張試験はやり直せない）
  の両方が起きない。残す＝フォルダ名を変えるだけ、捨てる＝消すだけなので停止時は一瞬。

データの重さで扱いを分ける:
  数値（変位場）… ROI 1,953 点で 0.5 MB/s と軽いので**判断せず全フレーム**書く
  映像          … 1.2 MB/s あるので一時領域に上限を設け、超えたら古いセグメントから捨てる
                  （直近は必ず残るので、破断の瞬間は常にバッファ内にある）

ファイル構成（<base>/_recording_YYYYmmdd_HHMMSS/ → 保存時に <base>/YYYYmmdd_HHMMSS/ へ改名）:
  meta.json      パラメータ・グリッド形状・カメラ・開始時刻・バックエンド
  grid.npz       グリッド座標 X, Y（不変なので 1 回だけ）
  reference.png  参照画像（可逆）
  disp.bin       毎フレーム: t(float64) + U(N float32) + V(N float32) + zncc(N uint8) を追記
                 ※ 追記型にしてあるので、試験中にクラッシュしてもそこまでが読める
  frames.csv     毎フレームの要約（荷重・応力の列は手入力用に空で用意）
  overlay_*.mp4  表示そのままの映像。既定 120 秒ごとのセグメント

ひずみ・ひずみ速度は dic_core.calc_strain_field(X, Y, U, V, step) で再現できるので保存しない
（11 配列 210 KB/frame → 3 配列 17 KB/frame）。読み出しは read_session() を使う。
"""
from __future__ import annotations
import csv
import json
import queue
import shutil
import pathlib
import threading
from datetime import datetime
from typing import Optional, Tuple, List, Dict

import numpy as np
import cv2

REC_PREFIX = "_recording_"
CSV_COLUMNS = ["frame", "time_s", "src_fps", "proc_fps", "median_zncc", "frac_bad",
               "exx_mean_pct", "exx_max_pct", "eyy_mean_pct", "rate_max_pct_s",
               "n_reref", "rereferenced", "marker", "load_N", "stress_MPa"]


def _nanstat(a: np.ndarray, fn) -> float:
    m = np.isfinite(a)
    return float(fn(a[m])) if m.any() else float("nan")


# ---------------------------------------------------------------------------
# 映像の書き出し（別スレッド）
# ---------------------------------------------------------------------------

class _VideoWriter(threading.Thread):
    """
    表示 RGB をセグメント分割して書く。GUI スレッドから呼ぶと 1 フレーム 10 ms 取られるので
    キュー経由で別スレッドに逃がす。キューが詰まったら捨てる（表示より解析を優先）。
    """

    def __init__(self, dir_: pathlib.Path, fps: float, segment_sec: float, budget_bytes: int):
        super().__init__(daemon=True)
        self.dir = dir_
        self.fps = float(fps) if fps and fps > 1 else 30.0
        self.segment_sec = float(segment_sec)
        self.budget = int(budget_bytes)
        self.q: "queue.Queue" = queue.Queue(maxsize=8)
        self._stop = threading.Event()
        self.segments: List[Tuple[pathlib.Path, float, float]] = []   # (path, t_start, t_end)
        self.n_dropped = 0
        self.n_written = 0
        self._vw: Optional[cv2.VideoWriter] = None
        self._seg_t0 = 0.0
        self._seg_path: Optional[pathlib.Path] = None
        self._size = 0
        self._lock = threading.Lock()

    # ---- 呼び出し側 ----
    def submit(self, rgb: np.ndarray, t: float):
        try:
            self.q.put_nowait((rgb, t))
        except queue.Full:
            self.n_dropped += 1

    def stop(self):
        self._stop.set()
        try:
            self.q.put_nowait(None)
        except queue.Full:
            pass
        self.join(timeout=10.0)

    def total_bytes(self) -> int:
        with self._lock:
            return self._size + (self._seg_path.stat().st_size if self._seg_path and self._seg_path.exists() else 0)

    def segment_list(self) -> List[Tuple[pathlib.Path, float, float]]:
        with self._lock:
            return list(self.segments)

    # ---- スレッド本体 ----
    def run(self):
        while True:
            item = self.q.get()
            if item is None or self._stop.is_set():
                break
            rgb, t = item
            try:
                self._write(rgb, t)
            except Exception:
                pass            # 記録の失敗で解析を止めない
        self._close_segment(None)

    def _open_segment(self, shape, t: float):
        h, w = shape[:2]
        p = self.dir / f"overlay_{len(self.segments):03d}.mp4"
        vw = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h), True)
        if not vw.isOpened():       # mp4v が無い環境は MJPG avi に落とす
            p = self.dir / f"overlay_{len(self.segments):03d}.avi"
            vw = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"MJPG"), self.fps, (w, h), True)
        self._vw, self._seg_path, self._seg_t0 = (vw if vw.isOpened() else None), p, t

    def _close_segment(self, t: Optional[float]):
        if self._vw is None:
            return
        self._vw.release(); self._vw = None
        if self._seg_path and self._seg_path.exists():
            with self._lock:
                self.segments.append((self._seg_path, self._seg_t0, t if t is not None else self._seg_t0))
                self._size += self._seg_path.stat().st_size
        self._seg_path = None
        self._enforce_budget()

    def _enforce_budget(self):
        """上限を超えたら古いセグメントから捨てる（直近が常に残るようにする）。"""
        with self._lock:
            while self._size > self.budget and len(self.segments) > 1:
                p, _, _ = self.segments.pop(0)
                try:
                    self._size -= p.stat().st_size
                    p.unlink()
                except Exception:
                    pass

    def _write(self, rgb: np.ndarray, t: float):
        if self._vw is None:
            self._open_segment(rgb.shape, t)
        elif t - self._seg_t0 >= self.segment_sec:
            self._close_segment(t)
            self._open_segment(rgb.shape, t)
        if self._vw is not None:
            self._vw.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            self.n_written += 1


# ---------------------------------------------------------------------------
# 記録本体
# ---------------------------------------------------------------------------

class Recorder:
    """解析開始と同時に一時フォルダへ記録する。停止後に keep() か discard() を呼ぶ。"""

    def __init__(self, base_dir: pathlib.Path, ref_gray: np.ndarray, X: np.ndarray, Y: np.ndarray,
                 meta: Dict, fps: float = 30.0, segment_sec: float = 120.0,
                 video_budget_gb: float = 10.0, video: bool = True):
        self.base = pathlib.Path(base_dir)
        self.stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dir = self.base / (REC_PREFIX + self.stamp)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.N = int(X.size)
        self.n_frames = 0
        self.markers: List[Tuple[int, float]] = []      # (frame, t)
        self._pending_marker = False
        self.t_first: Optional[float] = None
        self.t_last: Optional[float] = None
        self.closed = False

        np.savez(str(self.dir / "grid.npz"), X=X, Y=Y)
        cv2.imencode(".png", ref_gray)[1].tofile(str(self.dir / "reference.png"))
        meta = dict(meta)
        meta.update({"version": 1, "start": datetime.now().isoformat(),
                     "grid": [int(X.shape[0]), int(X.shape[1])], "n_points": self.N,
                     "record_format": "disp.bin: per frame = t(float64) + U(N float32) + V(N float32) + zncc(N uint8)"})
        (self.dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        self._fb = open(self.dir / "disp.bin", "wb", buffering=1024 * 1024)
        self._fc = open(self.dir / "frames.csv", "w", newline="", encoding="utf-8-sig")
        self._csv = csv.writer(self._fc); self._csv.writerow(CSV_COLUMNS)
        self._video = _VideoWriter(self.dir, fps, segment_sec, int(video_budget_gb * 1e9)) if video else None
        if self._video:
            self._video.start()

    # ---- 記録 ----
    def mark(self):
        """試験中の「ここ」マーカー。次のフレームに記録する。"""
        self._pending_marker = True

    def add_frame(self, t: float, res: dict, src_fps: float, proc_fps: float):
        """解析ワーカーから毎フレーム呼ぶ（数値のみ。軽いので判断せず全部書く）。"""
        if self.closed:
            return
        st = res["stats"]
        U = np.ascontiguousarray(res["U"], dtype=np.float32).ravel()
        V = np.ascontiguousarray(res["V"], dtype=np.float32).ravel()
        z = np.clip(np.nan_to_num(res["zncc"], nan=-1.0) * 255.0, 0, 255).astype(np.uint8).ravel()
        self._fb.write(np.float64(t).tobytes()); self._fb.write(U.tobytes())
        self._fb.write(V.tobytes()); self._fb.write(z.tobytes())
        marker = self._pending_marker
        if marker:
            self.markers.append((self.n_frames, t)); self._pending_marker = False
        exx, eyy = res["exx"], res["eyy"]
        rate = res.get("rexx")
        self._csv.writerow([
            self.n_frames, f"{t:.3f}", f"{src_fps:.1f}", f"{proc_fps:.1f}",
            f"{st['median_zncc']:.4f}", f"{st['frac_bad']:.4f}",
            f"{_nanstat(exx, np.mean) * 100:.4f}", f"{_nanstat(exx, np.max) * 100:.4f}",
            f"{_nanstat(eyy, np.mean) * 100:.4f}",
            f"{_nanstat(rate, np.max) * 100:.4f}" if rate is not None else "",
            st["n_reref"], int(bool(st["rereferenced"])), int(marker), "", "",
        ])
        if self.t_first is None:
            self.t_first = t
        self.t_last = t
        self.n_frames += 1
        if self.n_frames % 30 == 0:
            self._fb.flush(); self._fc.flush()

    def add_display(self, rgb: np.ndarray, t: float):
        """描画スレッドから表示 RGB を渡す（別スレッドで書くので呼び出しは軽い）。"""
        if self._video is not None and not self.closed:
            self._video.submit(rgb, t)

    # ---- 状態表示 ----
    def status(self) -> str:
        dur = (self.t_last - self.t_first) if (self.t_first is not None and self.t_last is not None) else 0.0
        mb = (self.disk_bytes()) / 1e6
        m = f" ★{len(self.markers)}" if self.markers else ""
        return f"記録中 {int(dur // 60):02d}:{int(dur % 60):02d} / {mb:,.0f} MB{m}"

    def disk_bytes(self) -> int:
        n = 8 + self.N * 9
        v = self._video.total_bytes() if self._video else 0
        return self.n_frames * n + v

    # ---- 終了処理 ----
    def close(self):
        """解析停止時に呼ぶ。ファイルを閉じるだけで、残すかどうかはまだ決めない。"""
        if self.closed:
            return
        self.closed = True
        if self._video is not None:
            self._video.stop()
        for f in (self._fb, self._fc):
            try:
                f.flush(); f.close()
            except Exception:
                pass

    def discard(self):
        """捨てる。"""
        self.close()
        shutil.rmtree(self.dir, ignore_errors=True)

    def keep(self, t_from: Optional[float] = None, t_to: Optional[float] = None) -> pathlib.Path:
        """
        残す。t_from/t_to を与えるとその時間範囲だけに切り詰める（範囲外の映像セグメントは削除、
        disp.bin と frames.csv は該当フレームだけ書き直す）。戻り値は確定後のフォルダ。
        """
        self.close()
        if t_from is not None or t_to is not None:
            self._trim(t_from if t_from is not None else -1e18,
                       t_to if t_to is not None else 1e18)
        dest = self.base / self.stamp
        i = 1
        while dest.exists():
            dest = self.base / f"{self.stamp}_{i}"; i += 1
        self.dir.rename(dest)      # 同一ボリュームなので一瞬
        return dest

    def _trim(self, t0: float, t1: float):
        rec = 8 + self.N * 9
        path = self.dir / "disp.bin"
        try:
            raw = np.fromfile(str(path), dtype=np.uint8)
            n = raw.size // rec
            raw = raw[:n * rec].reshape(n, rec)
            ts = raw[:, :8].copy().view(np.float64).ravel()
            keep = np.nonzero((ts >= t0) & (ts <= t1))[0]
            if keep.size:
                raw[keep].tofile(str(path))
            # frames.csv も同じ範囲に
            csv_path = self.dir / "frames.csv"
            rows = list(csv.reader(open(csv_path, encoding="utf-8-sig")))
            head, body = rows[0], rows[1:]
            body = [r for r in body if r and t0 <= float(r[1]) <= t1]
            with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f); w.writerow(head); w.writerows(body)
        except Exception:
            pass
        # 範囲と重ならない映像セグメントを削除
        if self._video is not None:
            for p, s, e in self._video.segment_list():
                if (e < t0 or s > t1) and p.exists():
                    try:
                        p.unlink()
                    except Exception:
                        pass


# ---------------------------------------------------------------------------
# 読み出し
# ---------------------------------------------------------------------------

def read_session(dir_: str) -> Dict:
    """
    保存したセッションを読む。戻り値:
      meta, X, Y, t (T,), U (T,ny,nx), V, zncc (0〜1, 不良点は負)
    ひずみは dic_core.calc_strain_field(X, Y, U[i], V[i], step) で再現できる。
    """
    d = pathlib.Path(dir_)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    g = np.load(str(d / "grid.npz"))
    X, Y = g["X"], g["Y"]
    ny, nx = X.shape
    N = ny * nx
    rec = 8 + N * 9
    raw = np.fromfile(str(d / "disp.bin"), dtype=np.uint8)
    T = raw.size // rec                      # 途中で切れていても読める
    raw = raw[:T * rec].reshape(T, rec)
    t = raw[:, :8].copy().view(np.float64).ravel()
    U = raw[:, 8:8 + N * 4].copy().view(np.float32).reshape(T, ny, nx)
    V = raw[:, 8 + N * 4:8 + N * 8].copy().view(np.float32).reshape(T, ny, nx)
    z = raw[:, 8 + N * 8:].astype(np.float32).reshape(T, ny, nx) / 255.0
    return {"meta": meta, "X": X, "Y": Y, "t": t, "U": U, "V": V, "zncc": z}


def find_stale(base_dir: str) -> List[pathlib.Path]:
    """
    前回クラッシュして残った記録中フォルダを探す（起動時に呼ぶ）。

    **自動では消さない。** 追記型にしてあるのはクラッシュしても残すためで、
    勝手に消したら意味がない。残すか捨てるかは停止時と同じくユーザーが決める。
    """
    base = pathlib.Path(base_dir)
    if not base.is_dir():
        return []
    return sorted(p for p in base.glob(REC_PREFIX + "*") if p.is_dir())


def recover(dir_: str, keep: bool) -> Optional[pathlib.Path]:
    """クラッシュで残ったフォルダを確定（keep=True）または削除する。"""
    p = pathlib.Path(dir_)
    if not p.is_dir():
        return None
    if not keep:
        shutil.rmtree(p, ignore_errors=True)
        return None
    dest = p.with_name(p.name[len(REC_PREFIX):] + "_recovered")
    i = 1
    while dest.exists():
        dest = p.with_name(p.name[len(REC_PREFIX):] + f"_recovered_{i}"); i += 1
    p.rename(dest)
    return dest


def session_bytes(dir_: str) -> int:
    return sum(f.stat().st_size for f in pathlib.Path(dir_).rglob("*") if f.is_file())
