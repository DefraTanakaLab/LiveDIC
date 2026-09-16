"""
dic_rt.py  —  LiveDIC リアルタイムマップモード用コア

連続フレーム（USBカメラ 30fps 等）を対象に、
  ・全グリッド点を一括処理するバッチZNCC（Python呼び出しオーバーヘッドを排除）
  ・前フレームの変位を初期予測値にした「狭い探索＋必要時だけ自動拡張」
  ・固定参照＋相関低下時の自動再参照（変位は累積して元グリッド基準を維持）
を行う。

配列モジュール xp を CuPy / NumPy で切り替えることで GPU / CPU を同一コードで動かす。
起動時に CuPy と CUDA デバイスが使えれば GPU、なければ自動で CPU にフォールバック。
"""
from __future__ import annotations
import os
import time
import warnings
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
import scipy.fft
from typing import Optional, Tuple, Dict, Sequence

import dic_core
import dic_icgn

# ---------------------------------------------------------------------------
# バックエンド判定（CuPy が import でき、CUDA デバイスが 1 台以上あれば GPU）
# ---------------------------------------------------------------------------
_cp = None
_GPU_NAME = ""
_GPU_ERROR = ""
try:
    import cupy as _cupy
    if _cupy.cuda.runtime.getDeviceCount() > 0:
        _props = _cupy.cuda.runtime.getDeviceProperties(0)
        _name = _props.get("name", b"")
        _GPU_NAME = _name.decode() if isinstance(_name, bytes) else str(_name)
        # 実際にカーネルが動くか確認（sm 非対応などで import は通るが実行不能な場合を検出）
        _cupy.arange(4).sum()
        _cp = _cupy
except Exception as _e:  # ImportError, CUDARuntimeError など
    _cp = None
    _GPU_ERROR = str(_e)


def gpu_available() -> bool:
    return _cp is not None


def backend_name(use_gpu: Optional[bool] = None) -> str:
    """表示用のバックエンド名。use_gpu=None は自動判定結果。"""
    if (use_gpu is None and _cp is not None) or (use_gpu and _cp is not None):
        return f"GPU ({_GPU_NAME})"
    return "CPU (numpy)"


def gpu_error() -> str:
    return _GPU_ERROR


def _to_numpy(a):
    return _cp.asnumpy(a) if (_cp is not None and isinstance(a, _cp.ndarray)) else np.asarray(a)


# ---------------------------------------------------------------------------
# リアルタイムトラッカー
# ---------------------------------------------------------------------------

class RTTracker:
    """
    固定参照＋予測探索＋自動再参照のフルフィールドトラッカー。

    座標系:
      gx0, gy0     … 最初の参照画像上のグリッド点（不変）
      cx, cy       … 現在の参照画像上のサブセット中心（整数。再参照のたびに更新）
      u_base       … cx - gx0（再参照で累積した整数変位）
      u_pred       … 現在の参照に対する直近の変位（次フレームの初期予測値）
      U_total      … u_base + u（元グリッド基準の総変位。ひずみはこれから計算）
    """

    def __init__(
        self,
        ref_img: np.ndarray,
        subset_size: int,
        step: int,
        roi_xyxy: Optional[Tuple[int, int, int, int]] = None,
        zncc_threshold: float = 0.2,
        search_stages: Sequence[int] = (10, 30),
        retry_zncc: float = 0.6,
        reref_zncc: float = 0.75,
        reref_frac: float = 0.3,
        reref_min_interval: int = 5,
        use_gpu: Optional[bool] = None,
        mac_budget: Optional[float] = None,
        n_threads: Optional[int] = None,
        median_filter: bool = False,
        rate_window: float = 3.0,
        rate_smooth: int = 3,
        icgn: bool = False,
        icgn_max_iter: int = 3,
    ):
        if use_gpu is None:
            use_gpu = _cp is not None
        if use_gpu and _cp is None:
            use_gpu = False
        self.xp = _cp if use_gpu else np
        self.use_gpu = use_gpu

        self.subset = int(subset_size) | 1     # 奇数化
        self.half = self.subset // 2
        self.step = int(step)
        self.zncc_thr = float(zncc_threshold)
        self.stages = [int(r) for r in search_stages if int(r) > 0]
        if not self.stages:
            self.stages = [3]
        self.retry_zncc = float(retry_zncc)
        self.reref_zncc = float(reref_zncc)
        self.reref_frac = float(reref_frac)
        self.reref_min_interval = int(reref_min_interval)
        # 拡張探索の計算量上限（積和回数）。超える段階はそのフレームでは省略する。
        self.mac_budget = float(mac_budget) if mac_budget else (4e9 if use_gpu else 1.5e9)

        self.img_shape = ref_img.shape[:2]
        H, W = self.img_shape
        X, Y = dic_core.create_grid(self.img_shape, self.subset, self.stages[0], self.step, roi_xyxy)
        self.X, self.Y = X, Y
        self.ny, self.nx = X.shape
        self.N = X.size
        self.gx0 = X.ravel().astype(np.int64)
        self.gy0 = Y.ravel().astype(np.int64)
        self.cx = self.gx0.copy()
        self.cy = self.gy0.copy()
        self.u_base = np.zeros(self.N, np.float64)
        self.v_base = np.zeros(self.N, np.float64)
        self.u_pred = np.zeros(self.N, np.float32)
        self.v_pred = np.zeros(self.N, np.float32)
        # 速度外挿用: 直近の良好点の (時刻, 総変位) と推定速度 [px/s]
        self.t_last = np.full(self.N, np.nan)
        self.vel_u = np.zeros(self.N, np.float32)
        self.vel_v = np.zeros(self.N, np.float32)
        self.last_good = np.zeros(self.N, bool)
        self.ever_good = np.zeros(self.N, bool)   # 一度でも追従できた点（描画用の変位埋めに使う）
        self.velocity_gain = 0.9     # 外挿の重み（1 で完全外挿。過大予測を避けて少し抑える）
        # 位相相関フォールバック用: 前フレームの縮小 ROI（探索範囲を超える大ジャンプの復帰に使う）
        self._prev_small: Optional[np.ndarray] = None
        self._small_scale = 4
        self.u_total_last = np.zeros(self.N, np.float64)   # 直近の有効な総変位
        self.v_total_last = np.zeros(self.N, np.float64)
        self.valid = np.ones(self.N, bool)                  # 永久無効点（画像外・無地）

        self.pad = self.half + max(self.stages) + 1
        # IC-GN 精密化（整数探索＋放物線の結果を初期値に、1 次形状関数で反復）
        self.icgn = bool(icgn)
        self.icgn_max_iter = int(icgn_max_iter)
        self._icgn: Optional[dic_icgn.ICGNRefiner] = None
        self.p_aff = np.zeros((self.N, 6), np.float32)   # 点ごとの直近ワープ（次フレームの初期値）
        self.median_filter = bool(median_filter)
        # ひずみ速度: 約 rate_window 秒前のひずみ場との差分から求める（隣接フレーム差分はノイズが大きすぎる）
        self.rate_window = float(rate_window)
        self.rate_smooth = int(rate_smooth)     # 速度マップの k×k 平均平滑化（1 で無効）
        self._strain_hist: deque = deque()     # (t, exx, eyy, exy)
        # CPU 経路のスレッド数（cv2.matchTemplate は GIL を解放する。4 本で頭打ちの実測）
        if n_threads is None:
            n_threads = max(1, min(4, (os.cpu_count() or 2) - 1))
        self.n_threads = int(n_threads)
        self._pool = ThreadPoolExecutor(self.n_threads) if self.n_threads > 1 else None
        self._band_cache: Dict = {}
        self._ref_fft_cache: Dict = {}   # GPU 経路: 参照サブセットの FFT（FFT サイズごと。再参照で破棄）
        self.frame_count = 0
        self.n_reref = 0
        self.frames_since_reref = 0
        self.last_reref_frame = -1

        self._set_reference(ref_img)

    # ------------------------------------------------------------------
    # 参照設定
    # ------------------------------------------------------------------
    def _prep_image(self, img_u8: np.ndarray):
        """uint8 グレー画像をパディングして xp float32 [0,1] にする。"""
        if img_u8.ndim == 3:
            img_u8 = cv2.cvtColor(img_u8, cv2.COLOR_BGR2GRAY)
        p = self.pad
        padded = cv2.copyMakeBorder(img_u8, p, p, p, p, cv2.BORDER_CONSTANT, value=0)
        return self.xp.asarray(padded, dtype=self.xp.float32) / self.xp.float32(255.0)

    def _set_reference(self, img_u8: np.ndarray):
        xp = self.xp
        S, half, p = self.subset, self.half, self.pad
        img = self._prep_image(img_u8)
        H, W = self.img_shape
        # 中心がサブセット＋第1段探索の余裕を持って画像内にある点のみ有効
        m = half + self.stages[0]
        inside = ((self.cx - m >= 0) & (self.cx + m < W) &
                  (self.cy - m >= 0) & (self.cy + m < H))
        self.valid &= inside

        cx = xp.asarray(self.cx + p)
        cy = xp.asarray(self.cy + p)
        ar = xp.arange(S)
        rows = (cy - half)[:, None] + ar[None, :]         # (N,S)
        cols = (cx - half)[:, None] + ar[None, :]
        patches = img[rows[:, :, None], cols[:, None, :]]  # (N,S,S)
        mean = patches.mean(axis=(1, 2), keepdims=True)
        self.ref_zm = (patches - mean).astype(xp.float32)
        norm = xp.sqrt((self.ref_zm ** 2).sum(axis=(1, 2)))
        self.ref_norm = norm.astype(xp.float32)
        flat = _to_numpy(norm) < 1e-3     # 無地（コントラストなし）
        self.valid &= ~flat
        self._ref_fft_cache.clear()
        self.frames_since_reref = 0
        if self.icgn:
            img_gray = img_u8 if img_u8.ndim == 2 else cv2.cvtColor(img_u8, cv2.COLOR_BGR2GRAY)
            self._icgn = dic_icgn.ICGNRefiner(img_gray, self.cx, self.cy, S, xp=xp,
                                              max_iter=self.icgn_max_iter, margin=self.stages[0] + 8)
            self.p_aff[:] = 0.0

    # ------------------------------------------------------------------
    # バッチZNCC（1段階分）
    # ------------------------------------------------------------------
    def _zncc_maps_cpu(self, img: np.ndarray, idx: np.ndarray,
                       x0: np.ndarray, y0: np.ndarray, r: int) -> np.ndarray:
        """CPU: 1点ごとに cv2.matchTemplate(TM_CCOEFF_NORMED)。cv2 は GIL を解放するのでスレッド並列が効く。"""
        S = self.subset
        K = 2 * r + 1
        Wn = S + 2 * r
        M = len(idx)
        out = np.empty((M, K, K), np.float32)
        ref = self.ref_zm

        def work(lo: int, hi: int):
            for k in range(lo, hi):
                i = idx[k]
                out[k] = cv2.matchTemplate(
                    img[y0[k]:y0[k] + Wn, x0[k]:x0[k] + Wn], ref[i], cv2.TM_CCOEFF_NORMED)

        nt = self.n_threads
        if nt <= 1 or M < 64:
            work(0, M)
        else:
            b = np.linspace(0, M, nt + 1).astype(int)
            list(self._pool.map(lambda k: work(b[k], b[k + 1]), range(nt)))
        return out

    def _zncc_maps_gpu(self, img, idx: np.ndarray, x0: np.ndarray, y0: np.ndarray, r: int):
        """GPU(CuPy): 窓を一括 gather → 積分は帯行列 matmul、分子はバッチ FFT 相互相関。

        分子を全オフセットの einsum ループ（K×K 回のカーネル起動）で求めると
        1080p・1,547 点・K=21 で 44 ms かかり CPU より遅かった（RTX 5070 Ti 実測）。
        cuFFT のバッチ相関なら同条件で 0.9 ms、差は相対 1e-6 級。
        """
        xp = self.xp
        S = self.subset
        K = 2 * r + 1
        Wn = S + 2 * r
        M = len(idx)
        ar = xp.arange(Wn)
        rows = xp.asarray(y0)[:, None] + ar[None, :]
        cols = xp.asarray(x0)[:, None] + ar[None, :]
        win = img[rows[:, :, None], cols[:, None, :]]      # (M,Wn,Wn)
        nf = scipy.fft.next_fast_len(Wn)
        ref_fft_all = self._ref_fft(nf)
        if M != self.N:
            idx_x = xp.asarray(idx)
            ref_fft = ref_fft_all[idx_x]
            ref_norm = self.ref_norm[idx_x]
        else:
            ref_fft, ref_norm = ref_fft_all, self.ref_norm

        # 分母：S×S の箱和を帯行列で (M,K,K) に一括計算
        B = self._band_matrix(K, Wn)
        sum_g = xp.matmul(xp.matmul(B, win), B.T)
        sum_g2 = xp.matmul(xp.matmul(B, win * win), B.T)
        n = float(S * S)
        var_g = xp.maximum(sum_g2 - sum_g * sum_g / n, 0.0)

        # 分子：Σ g·(f - f̄)（ref_zm は零平均なので ḡ の項は消える）
        # 反転した参照との畳み込み＝相互相関。circular サイズ nf ≥ Wn なら
        # 有効部分（線形インデックス S-1 .. Wn-1）には折り返しが混入しない
        win_fft = xp.fft.rfft2(win, s=(nf, nf), axes=(1, 2))
        corr = xp.fft.irfft2(win_fft * ref_fft, s=(nf, nf), axes=(1, 2))
        num = corr[:, S - 1:S - 1 + K, S - 1:S - 1 + K].astype(xp.float32)
        return num / (xp.sqrt(var_g) * ref_norm[:, None, None] + 1e-6)

    def _ref_fft(self, nf: int):
        """参照サブセット（零平均・上下左右反転）の rfft2。再参照まで不変なのでサイズごとにキャッシュ。"""
        if nf not in self._ref_fft_cache:
            xp = self.xp
            self._ref_fft_cache[nf] = xp.fft.rfft2(self.ref_zm[:, ::-1, ::-1], s=(nf, nf), axes=(1, 2))
        return self._ref_fft_cache[nf]

    def _band_matrix(self, K: int, Wn: int):
        key = (K, Wn)
        if key not in self._band_cache:
            B = np.zeros((K, Wn), np.float32)
            for j in range(K):
                B[j, j:j + self.subset] = 1.0
            self._band_cache[key] = self.xp.asarray(B)
        return self._band_cache[key]

    def _search(self, img, idx: np.ndarray, iu: np.ndarray, iv: np.ndarray, r: int):
        """
        idx で指定した点を、予測 (iu,iv) を中心に ±r で探索。
        戻り値（numpy）: u, v, zncc, border（ピークが探索窓の縁にあったか）
        """
        xp = self.xp
        S, half, p = self.subset, self.half, self.pad
        K = 2 * r + 1
        Wn = S + 2 * r
        M = len(idx)
        Hp, Wp = img.shape

        cx = self.cx[idx] + p
        cy = self.cy[idx] + p
        x0 = np.clip(cx + iu - half - r, 0, Wp - Wn).astype(np.int64)
        y0 = np.clip(cy + iv - half - r, 0, Hp - Wn).astype(np.int64)

        if self.use_gpu:
            zncc = self._zncc_maps_gpu(img, idx, x0, y0, r)
        else:
            zncc = self._zncc_maps_cpu(img, idx, x0, y0, r)
        x0x = xp.asarray(x0)
        y0x = xp.asarray(y0)

        # ---- ピーク＋放物線サブピクセル ----
        flat = zncc.reshape(M, -1)
        pk = xp.argmax(flat, axis=1)
        arM = xp.arange(M)
        peak = flat[arM, pk]
        bi = pk // K
        bj = pk % K
        border = (bi == 0) | (bi == K - 1) | (bj == 0) | (bj == K - 1)
        biC = xp.clip(bi, 1, max(K - 2, 1))
        bjC = xp.clip(bj, 1, max(K - 2, 1))
        if K >= 3:
            cm = zncc[arM, biC, bjC - 1]; c0 = zncc[arM, biC, bjC]; cpl = zncc[arM, biC, bjC + 1]
            d = cm - 2 * c0 + cpl
            du = xp.where(xp.abs(d) > 1e-12, 0.5 * (cm - cpl) / xp.where(xp.abs(d) > 1e-12, d, 1.0), 0.0)
            rm = zncc[arM, biC - 1, bjC]; rpl = zncc[arM, biC + 1, bjC]
            d2 = rm - 2 * c0 + rpl
            dv = xp.where(xp.abs(d2) > 1e-12, 0.5 * (rm - rpl) / xp.where(xp.abs(d2) > 1e-12, d2, 1.0), 0.0)
            du = xp.clip(du, -1.0, 1.0); dv = xp.clip(dv, -1.0, 1.0)
            du = xp.where(border, 0.0, du); dv = xp.where(border, 0.0, dv)
        else:
            du = xp.zeros(M, xp.float32); dv = xp.zeros(M, xp.float32)

        # 変位 = 候補中心（パディング座標） - 現在の中心（パディング座標）
        u = (x0x + bj + half).astype(xp.float32) - xp.asarray(cx, dtype=xp.float32) + du
        v = (y0x + bi + half).astype(xp.float32) - xp.asarray(cy, dtype=xp.float32) + dv
        return _to_numpy(u), _to_numpy(v), _to_numpy(peak), _to_numpy(border)

    # ------------------------------------------------------------------
    # 1フレーム処理
    # ------------------------------------------------------------------
    def process(self, def_img_u8: np.ndarray, t: Optional[float] = None) -> Dict:
        """1フレーム解析。t はフレームの取得時刻 [s]（ひずみ速度の計算に使う。省略時は現在時刻）。"""
        t0 = time.perf_counter()
        if t is None:
            t = t0
        img = self._prep_image(def_img_u8)
        N = self.N
        S = self.subset

        u = np.full(N, np.nan, np.float32)
        v = np.full(N, np.nan, np.float32)
        z = np.full(N, -1.0, np.float32)
        pu, pv = self._predict(t)
        iu = np.round(pu).astype(np.int64)
        iv = np.round(pv).astype(np.int64)

        need = self.valid.copy()
        stage_counts = []
        skipped = 0
        for si, r in enumerate(self.stages):
            idx = np.nonzero(need)[0]
            if idx.size == 0:
                stage_counts.append(0)
                continue
            K = 2 * r + 1
            if si > 0 and idx.size * K * K * S * S > self.mac_budget:
                skipped = int(idx.size)
                stage_counts.append(0)
                break
            us, vs, zs, bs = self._search(img, idx, iu[idx], iv[idx], r)
            better = zs > z[idx]
            sel = idx[better]
            u[sel] = us[better]; v[sel] = vs[better]; z[sel] = zs[better]
            # 相関が閾値未満、またはピークが窓の縁 → 次段で拡張探索
            need[idx] = (z[idx] < self.retry_zncc) | bs
            stage_counts.append(int(idx.size))

        # ---- 復帰処理：広域探索が計算量上限で省略されたとき ----
        # 省略された点から一様に少数（プローブ）を選んで最終段の半径で探索し、
        # 見つかった「予測からのズレ」を近傍に伝播して残りを第2段の半径で再探索する。
        # 一気に大きく動いた（引張機のジャンプ等）フレームでも 1 フレーム内で追従を取り戻す。
        n_probe_used = 0
        if skipped > 0 and len(self.stages) >= 2:
            idx = np.nonzero(need)[0]
            r_wide = self.stages[-1]
            K = 2 * r_wide + 1
            n_probe = int(max(16, min(idx.size, self.mac_budget // (K * K * S * S))))
            probe = idx[np.linspace(0, idx.size - 1, n_probe).astype(int)]
            us, vs, zs, bs = self._search(img, probe, iu[probe], iv[probe], r_wide)
            better = zs > z[probe]
            sel = probe[better]
            u[sel] = us[better]; v[sel] = vs[better]; z[sel] = zs[better]
            need[probe] = (z[probe] < self.retry_zncc) | bs
            n_probe_used = int(n_probe)
            hit = probe[(zs >= self.retry_zncc) & ~bs]
            rest = idx[~np.isin(idx, probe)]
            if hit.size < 4 and self._prev_small is not None:
                # 最終段の半径でも見つからない → 位相相関で全体のズレを測り、直近変位＋ズレを中心に再探索
                gs = self._global_shift(def_img_u8)
                if gs is not None:
                    sx, sy = gs
                    iu3 = np.round(self.u_pred[idx] + sx).astype(np.int64)
                    iv3 = np.round(self.v_pred[idx] + sy).astype(np.int64)
                    us, vs, zs, bs = self._search(img, idx, iu3, iv3, self.stages[1])
                    better = zs > z[idx]
                    sel = idx[better]
                    u[sel] = us[better]; v[sel] = vs[better]; z[sel] = zs[better]
                    need[idx] = (z[idx] < self.retry_zncc) | bs
                    n_probe_used = -int(abs(sx) + abs(sy))   # 負値で「位相相関で復帰」を表す
                    rest = np.zeros(0, np.int64)
            if hit.size >= 4 and rest.size > 0:
                du = np.full(N, np.nan, np.float32); dv = np.full(N, np.nan, np.float32)
                du[hit] = u[hit] - iu[hit]; dv[hit] = v[hit] - iv[hit]
                mdu = _nanmedian_window(du.reshape(self.ny, self.nx), 9).ravel()
                mdv = _nanmedian_window(dv.reshape(self.ny, self.nx), 9).ravel()
                gdu = float(np.nanmedian(du[hit])); gdv = float(np.nanmedian(dv[hit]))
                mdu = np.where(np.isfinite(mdu), mdu, gdu); mdv = np.where(np.isfinite(mdv), mdv, gdv)
                iu2 = iu[rest] + np.round(mdu[rest]).astype(np.int64)
                iv2 = iv[rest] + np.round(mdv[rest]).astype(np.int64)
                r2 = self.stages[1]
                us, vs, zs, bs = self._search(img, rest, iu2, iv2, r2)
                better = zs > z[rest]
                sel = rest[better]
                u[sel] = us[better]; v[sel] = vs[better]; z[sel] = zs[better]
                need[rest] = (z[rest] < self.retry_zncc) | bs

        # マッチしたサブセットが実画像の外（パディング領域）にかかる点は視野外として無効
        H, W = self.img_shape
        with np.errstate(invalid="ignore"):
            mx = self.cx + u
            my = self.cy + v
            outside = ((mx - self.half < 0) | (mx + self.half >= W) |
                       (my - self.half < 0) | (my + self.half >= H))
        z[outside & np.isfinite(u)] = -1.0
        good = self.valid & (z >= self.zncc_thr)

        # ---- IC-GN 精密化（良好点のみ。前フレームのワープを初期値にするので通常 1〜2 反復）----
        icgn_stats = {}
        if self._icgn is not None and good.any():
            H, W = self.img_shape
            m = self.half + self.stages[0] + 8
            mx_g = self.cx[good] + u[good]; my_g = self.cy[good] + v[good]
            bbox = (int(mx_g.min()) - m, int(my_g.min()) - m, int(mx_g.max()) + m + 1, int(my_g.max()) + m + 1)
            t_i = time.perf_counter()
            self._icgn.set_deformed(def_img_u8, bbox)
            p0 = self.p_aff.copy()
            p0[:, 0] = np.where(good, u, 0.0); p0[:, 3] = np.where(good, v, 0.0)
            p_new, z2, conv, n_it = self._icgn.refine(p0, mask=good)
            # 収束した点、または未収束でも相関が悪化していない点（最適解へ近づいている）を採用
            acc = good & np.isfinite(z2) & (conv | (z2 >= z))
            u[acc] = p_new[acc, 0]; v[acc] = p_new[acc, 3]; z[acc] = z2[acc]
            self.p_aff[acc] = p_new[acc]
            icgn_stats = {"icgn_frac": float(acc.sum() / max(1, good.sum())),
                          "icgn_iter": float(n_it[good].mean()) if good.any() else 0.0,
                          "icgn_time": time.perf_counter() - t_i}
        # 速度更新（前回も良好だった点だけ。dt が極端に短いときはノイズになるので更新しない）
        both = good & self.last_good & np.isfinite(self.t_last)
        dt_p = t - self.t_last
        upd = both & (dt_p > 1e-3)
        if upd.any():
            d_u = self.u_base[upd] + u[upd] - self.u_total_last[upd]
            d_v = self.v_base[upd] + v[upd] - self.v_total_last[upd]
            # 1 フレームで第2段の探索半径を超えるほど動いたのは「ジャンプ」であり速度ではない。
            # 学習すると次フレームでさらに同量を外挿して見失うので、その点の速度はリセットする
            r_jump = float(self.stages[1] if len(self.stages) > 1 else self.stages[0])
            jump = (np.abs(d_u) > r_jump) | (np.abs(d_v) > r_jump)
            new_vu = np.where(jump, 0.0, d_u / dt_p[upd])
            new_vv = np.where(jump, 0.0, d_v / dt_p[upd])
            # 指数平滑で暴れを抑える（ジャンプ点は即リセット）
            self.vel_u[upd] = np.where(jump, 0.0, 0.5 * self.vel_u[upd] + 0.5 * new_vu)
            self.vel_v[upd] = np.where(jump, 0.0, 0.5 * self.vel_v[upd] + 0.5 * new_vv)
        self.u_pred[good] = u[good]
        self.v_pred[good] = v[good]
        self.u_total_last[good] = self.u_base[good] + u[good]
        self.v_total_last[good] = self.v_base[good] + v[good]
        self.t_last[good] = t
        self.last_good = good.copy()
        self.ever_good |= good

        U = np.full(N, np.nan, np.float32)
        V = np.full(N, np.nan, np.float32)
        U[good] = self.u_total_last[good]
        V[good] = self.v_total_last[good]
        Z = z.copy(); Z[~self.valid] = np.nan

        n_valid = int(self.valid.sum())
        n_good = int(good.sum())
        frac_bad = 1.0 - n_good / n_valid if n_valid else 1.0
        med_z = float(np.median(z[good])) if n_good else 0.0

        self.frame_count += 1
        self.frames_since_reref += 1

        # ---- 自動再参照 ----
        # 良好点が半分未満のとき（無地・ピンぼけ・視野外）は再参照しても改善しないので見送る
        rerefed = False
        if (self.frames_since_reref >= self.reref_min_interval and n_valid > 0 and
                frac_bad <= 0.5 and
                (med_z < self.reref_zncc or frac_bad > self.reref_frac)):
            self._rereference(def_img_u8)
            rerefed = True

        self._prev_small = self._make_small(def_img_u8)

        U2 = U.reshape(self.ny, self.nx)
        V2 = V.reshape(self.ny, self.nx)
        if self.median_filter:
            U2 = nanmedian3(U2)
            V2 = nanmedian3(V2)
        strain = dic_core.calc_strain_field(self.X, self.Y, U2, V2, self.step)
        rate = self._strain_rate(t, strain)
        dt = time.perf_counter() - t0
        return {
            "X": self.X, "Y": self.Y,
            "U": U2, "V": V2, "zncc": Z.reshape(self.ny, self.nx),
            # 描画（追従モード）用: 見失った点も直近の有効な総変位で埋めた変位場。
            # 一度も追従できていない点は 0 のままなので、描画側で近傍から補う
            "U_draw": np.where(self.valid & self.ever_good, self.u_total_last, np.nan).reshape(self.ny, self.nx).astype(np.float32),
            "V_draw": np.where(self.valid & self.ever_good, self.v_total_last, np.nan).reshape(self.ny, self.nx).astype(np.float32),
            **strain, **rate,
            "stats": {
                "proc_time": dt,
                "median_zncc": med_z,
                "frac_bad": frac_bad,
                "n_points": N, "n_valid": n_valid, "n_good": n_good,
                "stage_counts": stage_counts,
                "skipped": skipped,
                "probe": n_probe_used,
                "rereferenced": rerefed,
                "n_reref": self.n_reref,
                "frame": self.frame_count,
                **icgn_stats,
            },
        }

    # ------------------------------------------------------------------
    # 位相相関による全体ズレ（探索範囲を超える大ジャンプからの復帰用）
    # ------------------------------------------------------------------
    def _make_small(self, img_u8: np.ndarray) -> np.ndarray:
        H, W = self.img_shape
        m = self.half + 4
        x0 = max(0, int(self.cx.min()) - m); x1 = min(W, int(self.cx.max()) + m + 1)
        y0 = max(0, int(self.cy.min()) - m); y1 = min(H, int(self.cy.max()) + m + 1)
        g = img_u8 if img_u8.ndim == 2 else cv2.cvtColor(img_u8, cv2.COLOR_BGR2GRAY)
        crop = g[y0:y1, x0:x1]
        k = self._small_scale
        small = cv2.resize(crop, (max(8, crop.shape[1] // k), max(8, crop.shape[0] // k)),
                           interpolation=cv2.INTER_AREA)
        return small.astype(np.float32)

    def _global_shift(self, img_u8: np.ndarray) -> Optional[Tuple[float, float]]:
        """前フレーム（縮小）との位相相関で全体シフト [px] を返す。応答が弱ければ None。"""
        cur = self._make_small(img_u8)
        prev = self._prev_small
        if prev is None or prev.shape != cur.shape:
            return None
        win = cv2.createHanningWindow((cur.shape[1], cur.shape[0]), cv2.CV_32F)
        (dx, dy), resp = cv2.phaseCorrelate(prev, cur, win)
        if resp < 0.1:
            return None
        return float(dx * self._small_scale), float(dy * self._small_scale)

    # ------------------------------------------------------------------
    # 探索中心の予測（速度外挿＋近傍伝播）
    # ------------------------------------------------------------------
    def _predict(self, t: float):
        """
        次の探索中心（現在の参照に対する変位）を返す。
          良好点: 直近の変位 + 推定速度 × 経過時間（速度外挿）
          不良点: 周囲 5×5 の良好点の予測値の中央値（近傍伝播）。無ければ自身の直近値
        """
        pu = self.u_pred.astype(np.float32).copy()
        pv = self.v_pred.astype(np.float32).copy()
        dt = t - self.t_last
        ok = self.last_good & np.isfinite(dt) & (dt > 0) & (dt < 5.0)
        if ok.any():
            g = self.velocity_gain
            pu[ok] += g * self.vel_u[ok] * dt[ok]
            pv[ok] += g * self.vel_v[ok] * dt[ok]
        bad = self.valid & ~self.last_good
        if bad.any() and self.last_good.any():
            gu = np.where(self.last_good, pu, np.nan).reshape(self.ny, self.nx)
            gv = np.where(self.last_good, pv, np.nan).reshape(self.ny, self.nx)
            mu = _nanmedian_window(gu, 5).ravel()
            mv = _nanmedian_window(gv, 5).ravel()
            fill = bad & np.isfinite(mu) & np.isfinite(mv)
            pu[fill] = mu[fill]
            pv[fill] = mv[fill]
        return pu, pv

    # ------------------------------------------------------------------
    # ひずみ速度
    # ------------------------------------------------------------------
    def _strain_rate(self, t: float, strain: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """rate_window 秒前（に最も近い過去）のひずみ場との差分 / 経過時間 [1/s]。履歴不足時は NaN。"""
        hist = self._strain_hist
        hist.append((t, strain["exx"], strain["eyy"], strain["exy"]))
        # 窓より古いものは、窓の境界をまたぐ 1 件だけ残して捨てる
        while len(hist) >= 2 and hist[1][0] <= t - self.rate_window:
            hist.popleft()
        t_old, exx_o, eyy_o, exy_o = hist[0]
        dt = t - t_old
        if dt < 0.5 * self.rate_window or dt <= 0:
            nan = np.full_like(strain["exx"], np.nan)
            return {"rexx": nan, "reyy": nan, "rexy": nan}
        k = self.rate_smooth
        return {"rexx": nan_smooth((strain["exx"] - exx_o) / dt, k),
                "reyy": nan_smooth((strain["eyy"] - eyy_o) / dt, k),
                "rexy": nan_smooth((strain["exy"] - exy_o) / dt, k)}

    # ------------------------------------------------------------------
    # 再参照
    # ------------------------------------------------------------------
    def _rereference(self, img_u8: np.ndarray):
        """現フレームを新しい参照にする。総変位は u_base に整数部を移して継続。"""
        new_cx = np.round(self.gx0 + self.u_total_last).astype(np.int64)
        new_cy = np.round(self.gy0 + self.v_total_last).astype(np.int64)
        self.cx = new_cx
        self.cy = new_cy
        self.u_base = (new_cx - self.gx0).astype(np.float64)
        self.v_base = (new_cy - self.gy0).astype(np.float64)
        self.u_pred = (self.u_total_last - self.u_base).astype(np.float32)
        self.v_pred = (self.v_total_last - self.v_base).astype(np.float32)
        self._set_reference(img_u8)
        self.n_reref += 1
        self.last_reref_frame = self.frame_count


def _nanmedian_window(a: np.ndarray, k: int) -> np.ndarray:
    """NaN を無視する k×k メディアン（自分自身を含む）。全 NaN の窓は NaN。"""
    ny, nx = a.shape
    r = k // 2
    p = np.pad(a, r, mode="constant", constant_values=np.nan)
    stack = np.stack([p[dy:dy + ny, dx:dx + nx] for dy in range(k) for dx in range(k)], axis=-1)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(stack, axis=-1)


def nanmedian3(a: np.ndarray) -> np.ndarray:
    """NaN を無視する 3×3 メディアンフィルタ（誤マッチの孤立点を抑える）。NaN 点は NaN のまま。"""
    ny, nx = a.shape
    if ny < 3 or nx < 3:
        return a
    p = np.pad(a, 1, mode="constant", constant_values=np.nan)
    stack = np.stack([p[dy:dy + ny, dx:dx + nx] for dy in range(3) for dx in range(3)], axis=-1)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(stack, axis=-1)
    med[~np.isfinite(a)] = np.nan
    return med.astype(a.dtype)


def nan_smooth(a: np.ndarray, k: int = 3) -> np.ndarray:
    """NaN を無視する k×k 平均フィルタ（正規化畳み込み）。NaN 点は NaN のまま。"""
    if k <= 1:
        return a
    finite = np.isfinite(a)
    v = np.where(finite, a, 0.0).astype(np.float32)
    w = finite.astype(np.float32)
    sv = cv2.blur(v, (k, k), borderType=cv2.BORDER_CONSTANT)
    sw = cv2.blur(w, (k, k), borderType=cv2.BORDER_CONSTANT)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = sv / sw
    out[~finite | (sw <= 0)] = np.nan
    return out.astype(a.dtype)


# ---------------------------------------------------------------------------
# 表示用：ひずみ配列 → 元画像サイズの RGBA オーバーレイ（ベクトル化）
# ---------------------------------------------------------------------------

def strain_to_rgba(
    data: np.ndarray, vmin: float, vmax: float,
    X: np.ndarray, Y: np.ndarray, img_shape: Tuple[int, int],
    lut: np.ndarray, alpha: int = 160,
    U: Optional[np.ndarray] = None, V: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    グリッド値をセル（step×step）に最近傍拡大して元画像サイズの RGBA(uint8) を返す。
    lut: (256,3) uint8 のカラーマップ。NaN は透明。
    U, V を渡すと、参照位置に描いたマップを変位で変形後の位置へ写す（材料に追従した描画。
    セルは伸びに合わせて広がる）。渡さなければ参照時点の位置に固定して描く。
    """
    H, W = img_shape[:2]
    ny, nx = data.shape
    xs = X[0, :].astype(int); ys = Y[:, 0].astype(int)
    step_x = int(xs[1] - xs[0]) if nx > 1 else 20
    step_y = int(ys[1] - ys[0]) if ny > 1 else 20
    finite = np.isfinite(data)
    t = np.zeros_like(data, dtype=np.float32)
    if vmax > vmin:
        t[finite] = (data[finite] - vmin) / (vmax - vmin)
    idx = np.clip(np.nan_to_num(t * 255.0), 0, 255).astype(np.uint8)
    small = np.zeros((ny, nx, 4), np.uint8)
    small[..., :3] = lut[idx]
    small[..., 3] = np.where(finite, alpha, 0).astype(np.uint8)
    big = cv2.resize(small, (nx * step_x, ny * step_y), interpolation=cv2.INTER_NEAREST)
    out = np.zeros((H, W, 4), np.uint8)
    x0 = int(xs[0] - step_x // 2); y0 = int(ys[0] - step_y // 2)
    bx0, by0 = max(0, x0), max(0, y0)
    bx1, by1 = min(W, x0 + big.shape[1]), min(H, y0 + big.shape[0])
    if bx1 > bx0 and by1 > by0:
        out[by0:by1, bx0:bx1] = big[by0 - y0:by1 - y0, bx0 - x0:bx1 - x0]
    if U is None or V is None:
        return out
    return warp_by_displacement(out, U, V, X, Y)


def warp_by_displacement(rgba_ref: np.ndarray, U: np.ndarray, V: np.ndarray,
                         X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    参照位置に描いた RGBA を、グリッドの変位 (U,V) で変形後の位置へ写す。
    出力画素 (x,y) の色 = 入力の (x - U(x,y), y - V(x,y))。変位場はグリッドから
    画像全体へ線形補間で広げる（変形後の位置で参照変位を引く近似。表示用途では十分）。
    """
    H, W = rgba_ref.shape[:2]
    ny, nx = U.shape
    xs = X[0, :].astype(int); ys = Y[:, 0].astype(int)
    step_x = int(xs[1] - xs[0]) if nx > 1 else 20
    step_y = int(ys[1] - ys[0]) if ny > 1 else 20
    Uf = np.nan_to_num(U, nan=0.0).astype(np.float32)
    Vf = np.nan_to_num(V, nan=0.0).astype(np.float32)
    # NaN 点は周囲の値で埋めておく（透明セルの縁が引きつるのを防ぐ）
    fin = np.isfinite(U)
    if fin.any() and not fin.all():
        # 全体中央値で埋めると周囲との段差で描画が糸状に引き伸ばされる。近傍の中央値で段階的に埋める
        Uw = np.where(fin, U, np.nan).astype(np.float32); Vw = np.where(fin, V, np.nan).astype(np.float32)
        for k in (3, 5, 9):
            miss = ~np.isfinite(Uw)
            if not miss.any():
                break
            mu = _nanmedian_window(Uw, k); mv = _nanmedian_window(Vw, k)
            Uw = np.where(miss & np.isfinite(mu), mu, Uw); Vw = np.where(miss & np.isfinite(mv), mv, Vw)
        Uf = np.where(np.isfinite(Uw), Uw, float(np.nanmedian(U))).astype(np.float32)
        Vf = np.where(np.isfinite(Vw), Vw, float(np.nanmedian(V))).astype(np.float32)
    # グリッド → 画像全体の稠密変位場（グリッド外は最寄りの値）
    big_u = cv2.resize(Uf, (nx * step_x, ny * step_y), interpolation=cv2.INTER_LINEAR)
    big_v = cv2.resize(Vf, (nx * step_x, ny * step_y), interpolation=cv2.INTER_LINEAR)
    x0 = int(xs[0] - step_x // 2); y0 = int(ys[0] - step_y // 2)
    pad_l, pad_t = max(0, x0), max(0, y0)
    pad_r = max(0, W - (x0 + big_u.shape[1])); pad_b = max(0, H - (y0 + big_u.shape[0]))
    dense_u = cv2.copyMakeBorder(big_u, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REPLICATE)[:H, :W]
    dense_v = cv2.copyMakeBorder(big_v, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REPLICATE)[:H, :W]
    if dense_u.shape != (H, W):   # グリッドが画像外へはみ出す場合の保険
        dense_u = cv2.resize(dense_u, (W, H)); dense_v = cv2.resize(dense_v, (W, H))
    # 出力が非透明になり得る範囲（参照セル範囲 ± 変位の最大）だけ remap する（1080p 全面だと 60ms）
    umin, umax = float(np.nanmin(Uf)), float(np.nanmax(Uf))
    vmin_, vmax_ = float(np.nanmin(Vf)), float(np.nanmax(Vf))
    ox0 = max(0, int(np.floor(x0 + min(0.0, umin))) - 2)
    ox1 = min(W, int(np.ceil(x0 + big_u.shape[1] + max(0.0, umax))) + 2)
    oy0 = max(0, int(np.floor(y0 + min(0.0, vmin_))) - 2)
    oy1 = min(H, int(np.ceil(y0 + big_u.shape[0] + max(0.0, vmax_))) + 2)
    out = np.zeros_like(rgba_ref)
    if ox1 <= ox0 or oy1 <= oy0:
        return out
    yy, xx = np.mgrid[oy0:oy1, ox0:ox1].astype(np.float32)
    map_x = xx - dense_u[oy0:oy1, ox0:ox1]
    map_y = yy - dense_v[oy0:oy1, ox0:ox1]
    out[oy0:oy1, ox0:ox1] = cv2.remap(rgba_ref, map_x, map_y, cv2.INTER_NEAREST,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    return out


def compose_overlay(gray_u8: np.ndarray, rgba: Optional[np.ndarray]) -> np.ndarray:
    """グレー画像に RGBA オーバーレイをアルファ合成して RGB(uint8) を返す。"""
    rgb = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2RGB)
    if rgba is None:
        return rgb
    a = rgba[..., 3:4].astype(np.float32) / 255.0
    out = rgb.astype(np.float32) * (1 - a) + rgba[..., :3].astype(np.float32) * a
    return out.astype(np.uint8)
