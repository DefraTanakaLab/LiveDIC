"""
check_gpu.py  —  リアルタイムマップの GPU 経路（CuPy）の動作確認

GPU 搭載機で最初に実行する。合成変形データで
  1. CuPy / CUDA が使えるか（使えない場合は理由）
  2. GPU 経路と CPU 経路の結果が一致するか（変位・ひずみの最大差）
  3. 処理時間（整数探索のみ / IC-GN 込み）
を表示する。GPU 経路は開発時に GPU 無しの PC で numpy 互換性しか確認できていないため、
ここで数値一致を確認してから実機に使うこと。

    python check_gpu.py            # 1080p・約1,500点
    python check_gpu.py --4k       # 4K・約6,000点
"""
import sys
import time
import numpy as np

import dic_rt
import rt_source


def main():
    big = "--4k" in sys.argv
    print(f"CuPy/GPU: {'利用可 → ' + dic_rt.backend_name(True) if dic_rt.gpu_available() else '利用不可'}")
    if not dic_rt.gpu_available():
        print(f"  理由: {dic_rt.gpu_error() or 'cupy が import できない'}")
        print("  pip install cupy-cuda12x を実行し、nvidia-smi で CUDA 12.x 以降のドライバを確認してください。")
        print("  RTX 50 系 (Blackwell, sm_120) は CUDA 12.8 以降に対応した CuPy が必要です。")
        return 1

    W, H = (3840, 2160) if big else (1920, 1080)
    roi = (int(W * 0.03), int(H * 0.35), int(W * 0.97), int(H * 0.65))
    src = rt_source.SyntheticSource(W, H, 30.0, strain_rate=0.02, band_gain=3.0, noise=2.0, seed=1)
    ref = src.render(0.0)
    frames = [(i / 10.0, src.render(i / 10.0)) for i in range(1, 11)]

    results = {}
    for icgn in (False, True):
        for gpu in (False, True):
            tr = dic_rt.RTTracker(ref, 41, 20, roi, zncc_threshold=0.5, use_gpu=gpu, icgn=icgn)
            times = []
            for t, img in frames:
                t0 = time.perf_counter()
                res = tr.process(img, t=t)
                times.append(time.perf_counter() - t0)
            results[(icgn, gpu)] = res
            st = res["stats"]
            label = f"{'IC-GN ' if icgn else '整数のみ'} {'GPU' if gpu else 'CPU'}"
            print(f"{label:14s}: 点数 {tr.N:5d}  中央値 {np.median(times[1:]) * 1000:6.1f} ms/frame "
                  f"({1 / np.median(times[1:]):5.1f} fps)  ZNCC {st['median_zncc']:.3f}  不良 {st['frac_bad'] * 100:.1f}%"
                  + (f"  IC-GN {st.get('icgn_frac', 0) * 100:.0f}% {st.get('icgn_time', 0) * 1000:.0f}ms" if icgn else ""))

    print()
    ok = True
    for icgn in (False, True):
        a, b = results[(icgn, False)], results[(icgn, True)]
        for k in ("U", "V", "exx"):
            m = np.isfinite(a[k]) & np.isfinite(b[k])
            d = float(np.max(np.abs(a[k][m] - b[k][m]))) if m.any() else float("nan")
            same_nan = np.array_equal(np.isfinite(a[k]), np.isfinite(b[k]))
            tol = 1e-3 if k != "exx" else 1e-4
            flag = "OK" if (d < tol and same_nan) else "NG"
            ok &= flag == "OK"
            print(f"{'IC-GN' if icgn else '整数 '} {k}: CPU/GPU 最大差 {d:.2e}  NaN一致={same_nan}  {flag}")
    print("\n結果:", "GPU 経路は CPU と一致しています" if ok else "不一致があります。dic_rt.py / dic_icgn.py の GPU 経路を確認してください")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
