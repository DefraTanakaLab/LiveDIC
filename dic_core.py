"""
dic_core.py  —  LiveDIC コア計算モジュール
PineOak dic_sem_strain_v58 のベクトル化ZNCCをベースに、
LiveDIC向けにシンプル1段階探索に整理。
"""
from __future__ import annotations
import sys
import numpy as np
import cv2
from typing import Optional, Tuple, Dict

# PyInstaller でフリーズされた exe の場合は joblib 並列を無効化
_FROZEN = getattr(sys, 'frozen', False)

try:
    from joblib import Parallel, delayed
    _JOBLIB = True
except ImportError:
    _JOBLIB = False

try:
    from scipy.interpolate import RectBivariateSpline
    _SCIPY = True
except ImportError:
    RectBivariateSpline = None  # type: ignore
    _SCIPY = False

# ---------------------------------------------------------------------------
# 画像読み込み
# ---------------------------------------------------------------------------

def imread_safe(path: str) -> np.ndarray:
    """日本語・スペース含むパス対応グレースケール読み込み（uint8）。"""
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"画像を読み込めません: {path}")
    return img


def read_gray(path: str) -> np.ndarray:
    """グレースケール float32 で返す。8/16bit 整数は [0,1] に正規化。"""
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"画像を読み込めません: {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY if img.shape[2] == 3
                           else cv2.COLOR_BGRA2GRAY)
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    if img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0
    return img.astype(np.float32)


# ---------------------------------------------------------------------------
# グリッド生成
# ---------------------------------------------------------------------------

def create_grid(
    shape: Tuple[int, int],
    subset_size: int,
    search_range: int,
    step: int,
    roi_xyxy: Optional[Tuple[int, int, int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """サブセット中心座標の格子を生成。roi_xyxy=(x1,y1,x2,y2) で範囲を絞れる。"""
    h, w = shape
    margin = subset_size // 2 + search_range + 2
    x_min, x_max = margin, w - margin - 1
    y_min, y_max = margin, h - margin - 1
    if roi_xyxy is not None:
        rx1, ry1, rx2, ry2 = [int(v) for v in roi_xyxy]
        if rx2 < rx1: rx1, rx2 = rx2, rx1
        if ry2 < ry1: ry1, ry2 = ry2, ry1
        x_min, x_max = max(x_min, rx1), min(x_max, rx2)
        y_min, y_max = max(y_min, ry1), min(y_max, ry2)
    xs = np.arange(x_min, x_max + 1, step, dtype=np.int32)
    ys = np.arange(y_min, y_max + 1, step, dtype=np.int32)
    if xs.size == 0 or ys.size == 0:
        raise ValueError("グリッドを生成できません。ROI・サブセット・ステップを見直してください。")
    return np.meshgrid(xs, ys)  # X, Y どちらも shape=(ny, nx)


# ---------------------------------------------------------------------------
# ZNCC（ベクトル化・高速版）
# ---------------------------------------------------------------------------

def _zncc_integer(
    ref: np.ndarray,
    deformed: np.ndarray,
    cx: int, cy: int,
    subset_size: int,
    search_x: int,
    search_y: int,
    init_u: int = 0,
    init_v: int = 0,
) -> Tuple[int, int, np.ndarray]:
    """
    cv2.matchTemplate (TM_CCOEFF_NORMED) による整数画素 ZNCC 探索。
    search_x: X方向探索範囲 [±px]
    search_y: Y方向探索範囲 [±px]
    戻り値: (best_u, best_v, zncc_map)  zncc_map shape=(2*sy+1, 2*sx+1)
    """
    half = subset_size // 2
    zncc_map = np.zeros((2 * search_y + 1, 2 * search_x + 1), dtype=np.float32)
    best_u, best_v = init_u, init_v

    patch_ref = ref[cy - half:cy + half + 1, cx - half:cx + half + 1]
    if patch_ref.shape != (subset_size, subset_size):
        return best_u, best_v, zncc_map

    # 探索領域の座標（init_u/v を中心に search_x/y だけ広げた矩形）
    sx0 = cx + init_u - search_x - half
    sx1 = cx + init_u + search_x + half + 1
    sy0 = cy + init_v - search_y - half
    sy1 = cy + init_v + search_y + half + 1

    # 画像境界でクリップ
    sx0c = max(0, sx0);  sy0c = max(0, sy0)
    sx1c = min(deformed.shape[1], sx1)
    sy1c = min(deformed.shape[0], sy1)

    region = deformed[sy0c:sy1c, sx0c:sx1c]
    if region.shape[0] < subset_size or region.shape[1] < subset_size:
        return best_u, best_v, zncc_map

    res = cv2.matchTemplate(
        region.astype(np.float32),
        patch_ref.astype(np.float32),
        cv2.TM_CCOEFF_NORMED,
    )
    # res[j, i] → zncc_map[j_off+j, i_off+i]
    j_off = sy0c - sy0
    i_off = sx0c - sx0
    rh, rw = res.shape
    zncc_map[j_off:j_off + rh, i_off:i_off + rw] = res

    best_vi, best_ui = np.unravel_index(np.argmax(zncc_map), zncc_map.shape)
    best_u = init_u + (int(best_ui) - search_x)
    best_v = init_v + (int(best_vi) - search_y)
    return best_u, best_v, zncc_map


# ---------------------------------------------------------------------------
# サブピクセル補間
# ---------------------------------------------------------------------------

def _subpixel_1d_gaussian(cm: float, c0: float, cp: float) -> float:
    if cm <= 0 or c0 <= 0 or cp <= 0:
        return 0.0
    lm, l0, lp = np.log(cm), np.log(c0), np.log(cp)
    d = lm - 2 * l0 + lp
    return float(np.clip(0.5 * (lm - lp) / d, -1.0, 1.0)) if abs(d) > 1e-12 else 0.0


def subpixel_refinement(
    zncc_map: np.ndarray,
    best_u: int, best_v: int,
    search_x: int, search_y: int,
    method: str = "parabolic",
    init_u: int = 0, init_v: int = 0,
) -> Tuple[float, float]:
    """zncc_map のピーク周辺からサブピクセルオフセットを推定。"""
    vi = best_v - round(init_v) + search_y
    ui = best_u - round(init_u) + search_x
    sz_v, sz_u = zncc_map.shape
    if not (0 < vi < sz_v - 1 and 0 < ui < sz_u - 1):
        return float(best_u), float(best_v)

    patch = zncc_map[vi - 1:vi + 2, ui - 1:ui + 2]

    if method == "gaussian":
        du = _subpixel_1d_gaussian(patch[1, 0], patch[1, 1], patch[1, 2])
        dv = _subpixel_1d_gaussian(patch[0, 1], patch[1, 1], patch[2, 1])
        return best_u + du, best_v + dv

    if method == "spline":
        try:
            if not _SCIPY:
                raise ImportError
            w = 2
            wy1, wy2 = max(0, vi - w), min(sz_v - 1, vi + w)
            wx1, wx2 = max(0, ui - w), min(sz_u - 1, ui + w)
            if (wy2 - wy1 >= 2) and (wx2 - wx1 >= 2):
                local = zncc_map[wy1:wy2 + 1, wx1:wx2 + 1].astype(np.float64)
                ys = np.arange(wy1, wy2 + 1, dtype=np.float64)
                xs = np.arange(wx1, wx2 + 1, dtype=np.float64)
                spl = RectBivariateSpline(ys, xs, local,
                                          kx=min(3, len(ys) - 1),
                                          ky=min(3, len(xs) - 1))
                ys_f = np.linspace(wy1, wy2, 50)
                xs_f = np.linspace(wx1, wx2, 50)
                vals = spl(ys_f, xs_f)
                sy, sx = np.unravel_index(np.argmax(vals), vals.shape)
                du = float(np.clip(xs_f[sx] - ui, -1.0, 1.0))
                dv = float(np.clip(ys_f[sy] - vi, -1.0, 1.0))
                return best_u + du, best_v + dv
        except Exception:
            pass

    # parabolic（デフォルト）
    def _para(cm, c0, cp):
        d = cm - 2 * c0 + cp
        return float(np.clip(0.5 * (cm - cp) / d, -1.0, 1.0)) if abs(d) > 1e-12 else 0.0

    du = _para(patch[1, 0], patch[1, 1], patch[1, 2])
    dv = _para(patch[0, 1], patch[1, 1], patch[2, 1])
    return best_u + du, best_v + dv


# ---------------------------------------------------------------------------
# 1点DIC（2点間モード用・高速）
# ---------------------------------------------------------------------------

def compute_single_point(
    ref: np.ndarray,
    deformed: np.ndarray,
    cx: int, cy: int,
    subset_size: int,
    search_x: int,
    search_y: int,
    method: str = "parabolic",
    init_u: float = 0.0,
    init_v: float = 0.0,
) -> Tuple[float, float, float]:
    """
    1点のみDIC計算。2点間モードの高速パスとして使用。
    戻り値: (u, v, zncc_peak)
    """
    best_u, best_v, zncc_map = _zncc_integer(
        ref, deformed, cx, cy, subset_size, search_x, search_y,
        int(round(init_u)), int(round(init_v)))
    u, v = subpixel_refinement(zncc_map, best_u, best_v, search_x, search_y,
                                method, int(round(init_u)), int(round(init_v)))
    return u, v, float(zncc_map.max())


# ---------------------------------------------------------------------------
# 行単位DIC（joblib並列用）
# ---------------------------------------------------------------------------

def _process_row(
    ref: np.ndarray,
    deformed: np.ndarray,
    xs: np.ndarray,
    cy: int,
    subset_size: int,
    search_x: int,
    search_y: int,
    method: str,
    init_us: list,
    init_vs: list,
) -> list:
    results = []
    for cx, iu, iv in zip(xs, init_us, init_vs):
        best_u, best_v, zncc_map = _zncc_integer(
            ref, deformed, int(cx), int(cy), subset_size, search_x, search_y,
            int(round(iu)), int(round(iv)))
        u, v = subpixel_refinement(zncc_map, best_u, best_v, search_x, search_y,
                                   method, int(round(iu)), int(round(iv)))
        results.append((int(cx), int(cy), u, v, float(zncc_map.max())))
    return results


# ---------------------------------------------------------------------------
# フルマップDIC（マップモード用）
# ---------------------------------------------------------------------------

def run_dic_full(
    ref: np.ndarray,
    deformed: np.ndarray,
    subset_size: int,
    step: int,
    search_x: int,
    search_y: int,
    method: str = "parabolic",
    zncc_threshold: float = 0.2,
    roi_xyxy: Optional[Tuple[int, int, int, int]] = None,
    n_workers: int = -1,
    init_u_grid: Optional[np.ndarray] = None,
    init_v_grid: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    フルフィールドDIC。joblib で行単位並列処理。
    戻り値: {'X','Y','U','V','zncc', 'exx','eyy','exy'}
    """
    X, Y = create_grid(ref.shape, subset_size, max(search_x, search_y), step, roi_xyxy)
    ny, nx = X.shape
    xs = X[0, :]
    ys = Y[:, 0]

    row_tasks = []
    for j, cy in enumerate(ys):
        if init_u_grid is not None:
            iu_row = [float(init_u_grid[j, i]) for i in range(nx)]
            iv_row = [float(init_v_grid[j, i]) for i in range(nx)]
        else:
            iu_row = [0.0] * nx
            iv_row = [0.0] * nx
        row_tasks.append((xs, int(cy), iu_row, iv_row))

    if _JOBLIB and n_workers != 1 and not _FROZEN:
        rows = Parallel(n_jobs=n_workers, backend='loky')(
            delayed(_process_row)(ref, deformed, t[0], t[1],
                                  subset_size, search_x, search_y, method, t[2], t[3])
            for t in row_tasks
        )
    else:
        rows = [_process_row(ref, deformed, t[0], t[1],
                              subset_size, search_x, search_y, method, t[2], t[3])
                for t in row_tasks]

    flat = [pt for row in rows for pt in row]
    U = np.array([p[2] for p in flat], dtype=np.float32).reshape(ny, nx)
    V = np.array([p[3] for p in flat], dtype=np.float32).reshape(ny, nx)
    Z = np.array([p[4] for p in flat], dtype=np.float32).reshape(ny, nx)

    if zncc_threshold > 0:
        mask = Z < zncc_threshold
        U[mask] = np.nan
        V[mask] = np.nan

    strain = calc_strain_field(X, Y, U, V, step)
    return {"X": X, "Y": Y, "U": U, "V": V, "zncc": Z, **strain}


# ---------------------------------------------------------------------------
# ひずみ計算
# ---------------------------------------------------------------------------

def calc_strain_field(
    X: np.ndarray, Y: np.ndarray,
    U: np.ndarray, V: np.ndarray,
    step: int,
) -> Dict[str, np.ndarray]:
    """中心差分でひずみテンソル（微小ひずみ）を計算。"""
    dx = float(np.nanmedian(np.diff(X[0, :]))) if X.shape[1] > 1 else float(step)
    dy = float(np.nanmedian(np.diff(Y[:, 0]))) if Y.shape[0] > 1 else float(step)
    ny, nx = U.shape
    exx = np.full_like(U, np.nan)
    eyy = np.full_like(U, np.nan)
    exy = np.full_like(U, np.nan)
    if nx >= 3 and ny >= 3:
        exx[1:-1, 1:-1] = (U[1:-1, 2:] - U[1:-1, :-2]) / (2 * dx)
        eyy[1:-1, 1:-1] = (V[2:, 1:-1] - V[:-2, 1:-1]) / (2 * dy)
        exy[1:-1, 1:-1] = 0.5 * (
            (U[2:, 1:-1] - U[:-2, 1:-1]) / (2 * dy) +
            (V[1:-1, 2:] - V[1:-1, :-2]) / (2 * dx)
        )
    return {"exx": exx, "eyy": eyy, "exy": exy}


# ---------------------------------------------------------------------------
# 2点間ひずみ
# ---------------------------------------------------------------------------

def two_point_strain(
    cx1: int, cy1: int, u1: float, v1: float,
    cx2: int, cy2: int, u2: float, v2: float,
) -> Dict[str, float]:
    """
    2点の変位から軸ひずみ・横ひずみ・せん断ひずみを計算。
    戻り値: {'exx','eyy','exy','e_axial','e_lateral'}
      e_axial   : 2点を結ぶ方向のひずみ
      e_lateral : それに垂直方向のひずみ
    """
    dx0 = cx2 - cx1
    dy0 = cy2 - cy1
    L0 = np.hypot(dx0, dy0)
    if L0 < 1e-6:
        return {"exx": np.nan, "eyy": np.nan, "exy": np.nan,
                "e_axial": np.nan, "e_lateral": np.nan}

    du = u2 - u1
    dv = v2 - v1
    nx_, ny_ = dx0 / L0, dy0 / L0  # 軸方向単位ベクトル
    tx_, ty_ = -ny_, nx_             # 横方向単位ベクトル

    # 変位勾配テンソル成分（1D近似）
    exx = du / L0 * (nx_ ** 2) if abs(dx0) > 1e-6 else np.nan
    eyy = dv / L0 * (ny_ ** 2) if abs(dy0) > 1e-6 else np.nan
    exy = 0.5 * (du * ny_ + dv * nx_) / L0

    e_axial = (du * nx_ + dv * ny_) / L0
    e_lateral = (du * tx_ + dv * ty_) / L0

    return {"exx": exx, "eyy": eyy, "exy": exy,
            "e_axial": e_axial, "e_lateral": e_lateral}


# ---------------------------------------------------------------------------
# スペックル品質評価
# ---------------------------------------------------------------------------

def _to_float32(img: np.ndarray) -> np.ndarray:
    """uint8/uint16/float を [0,1] の float32 に正規化。"""
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    if img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0
    return img.astype(np.float32)


def calc_mig(img: np.ndarray) -> float:
    """MIG（平均輝度勾配）を計算。値が大きいほど DIC に適した画像。"""
    f = _to_float32(img)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.mean(np.sqrt(gx ** 2 + gy ** 2)))


def calc_sssig_map(
    img: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    subset_size: int,
) -> np.ndarray:
    """
    グリッド点ごとの SSSIG（サブセット内輝度勾配二乗和）を計算。
    積分画像を使って完全ベクトル化。戻り値: X/Y と同形の float32 配列。
    """
    f = _to_float32(img)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    g2 = (gx ** 2 + gy ** 2).astype(np.float64)
    g2_int = cv2.integral(g2)   # shape (H+1, W+1)

    half = subset_size // 2
    ys_flat = Y.flatten().astype(int)
    xs_flat = X.flatten().astype(int)
    y1s = ys_flat - half;  y2s = ys_flat + half + 1
    x1s = xs_flat - half;  x2s = xs_flat + half + 1

    valid = ((y1s >= 0) & (x1s >= 0) &
             (y2s <= g2.shape[0]) & (x2s <= g2.shape[1]))
    sssig_flat = np.zeros(len(ys_flat), dtype=np.float32)
    if valid.any():
        y1v, y2v = y1s[valid], y2s[valid]
        x1v, x2v = x1s[valid], x2s[valid]
        sssig_flat[valid] = (g2_int[y2v, x2v] - g2_int[y1v, x2v]
                             - g2_int[y2v, x1v] + g2_int[y1v, x1v])
    return sssig_flat.reshape(Y.shape).astype(np.float32)


def _speckle_size_from_acf(region: np.ndarray) -> float:
    """FFT 自己相関の中心ピーク半値幅からスペックルサイズ [px] を推定。"""
    f = _to_float32(region)
    f -= f.mean()
    F = np.fft.fft2(f)
    acf = np.fft.fftshift(np.fft.ifft2(F * np.conj(F)).real)
    peak = float(acf.max())
    if peak < 1e-10:
        return 0.0
    acf /= peak
    cy, cx = acf.shape[0] // 2, acf.shape[1] // 2

    def _half_width(profile: np.ndarray, c: int) -> float:
        for k in range(c + 1, len(profile)):
            if profile[k] <= 0.5:
                prev, curr = float(profile[k - 1]), float(profile[k])
                t = (prev - 0.5) / (prev - curr) if prev > curr else 0.0
                return float(k - 1 - c) + t
        return float(len(profile) // 4)

    return (_half_width(acf[cy, :], cx) + _half_width(acf[:, cx], cy)) / 2


def calc_speckle_sizes_grid(
    img: np.ndarray,
    roi_xyxy: Optional[Tuple[int, int, int, int]] = None,
    n_div: int = 3,
) -> np.ndarray:
    """
    画像（または ROI 内）を n_div×n_div に分割し、
    各領域のスペックルサイズ [px] を返す。戻り値: shape (n_div, n_div)。
    """
    h, w = img.shape[:2]
    x1, y1, x2, y2 = (0, 0, w, h) if roi_xyxy is None else [int(v) for v in roi_xyxy]
    rw = max(1, (x2 - x1) // n_div)
    rh = max(1, (y2 - y1) // n_div)
    sizes = np.zeros((n_div, n_div), dtype=np.float32)
    for row in range(n_div):
        for col in range(n_div):
            rx1 = x1 + col * rw;  rx2 = min(x2, rx1 + rw)
            ry1 = y1 + row * rh;  ry2 = min(y2, ry1 + rh)
            region = img[ry1:ry2, rx1:rx2]
            if region.size > 0:
                sizes[row, col] = _speckle_size_from_acf(region)
    return sizes
