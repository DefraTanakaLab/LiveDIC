"""
make_demo_movie.py  —  リアルタイムマップモード用のデモ動画を生成する

合成スペックル試験片（横長・左端固定・右へ引張）に、一様ひずみ＋局所ひずみ帯（くびれ）の
成長を与えた動画を書き出す。「リアルタイムマップ」タブのソースで「動画ファイル…」から選ぶ。

使い方:
    python make_demo_movie.py                       # Demo_realtime/demo_tensile.avi (40秒, 1080p30)
    python make_demo_movie.py -o out.avi -d 60 --rate 0.003 --band-gain 3
"""
from __future__ import annotations
import argparse
import pathlib
import sys
import time
import numpy as np
import cv2

import rt_source


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="Demo_realtime/demo_tensile.avi")
    ap.add_argument("-d", "--duration", type=float, default=40.0, help="秒")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--rate", type=float, default=0.002, help="一様ひずみ速度 [1/s]（0.002 → 40秒で 8%）")
    ap.add_argument("--band-gain", type=float, default=3.0, help="くびれ帯の局所ひずみ倍率")
    ap.add_argument("--noise", type=float, default=2.0, help="画像ノイズ σ [階調]")
    ap.add_argument("--hold", type=float, default=3.0, help="冒頭の無変形保持時間 [s]（参照取得の余裕）")
    ap.add_argument("--codec", default="MJPG", help="MJPG(avi, 無劣化に近い) / mp4v(mp4)")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    src = rt_source.SyntheticSource(args.width, args.height, args.fps,
                                    strain_rate=args.rate, band_gain=args.band_gain,
                                    noise=args.noise)
    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    vw = cv2.VideoWriter(str(out), fourcc, args.fps, (args.width, args.height), isColor=False)
    if not vw.isOpened():
        # 日本語パスで開けない場合は一時名で書いてからリネーム
        tmp = pathlib.Path.home() / "_livedic_demo_tmp.avi"
        vw = cv2.VideoWriter(str(tmp), fourcc, args.fps, (args.width, args.height), isColor=False)
        if not vw.isOpened():
            print("VideoWriter を開けません", file=sys.stderr)
            sys.exit(1)
        out_tmp = tmp
    else:
        out_tmp = None

    n = int(round(args.duration * args.fps))
    t0 = time.perf_counter()
    for i in range(n):
        t = max(0.0, i / args.fps - args.hold)
        frame = src.render(t)
        vw.write(frame)
        if i % int(args.fps * 5) == 0:
            eps = src.eps_at(t)
            print(f"  {i / args.fps:5.1f}s  一様ひずみ {eps * 100:5.2f}%  帯中心 {eps * (1 + args.band_gain) * 100:5.2f}%"
                  f"   ({time.perf_counter() - t0:.0f}s 経過)")
    vw.release()
    if out_tmp is not None:
        out_tmp.replace(out)
    print(f"書き出し完了: {out}  ({n} フレーム, {args.duration:.0f} 秒, {args.width}×{args.height} @ {args.fps:.0f} fps)")
    print(f"  {args.hold:.0f} 秒間は無変形 → 以降 {args.rate * 100:.2f} %/s で引張、"
          f"最終: 一様 {args.rate * (args.duration - args.hold) * 100:.1f} %、"
          f"帯中心 {args.rate * (args.duration - args.hold) * (1 + args.band_gain) * 100:.1f} %")


if __name__ == "__main__":
    main()
