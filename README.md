# LiveDIC

インターバル撮影した画像のリアルタイムDIC（デジタル画像相関法）解析ツール。

フォルダを監視し、新しい画像が来るたびに自動でDIC計算・表示を更新します。

---

## 主な機能

- **2点間モード**: 登録したゲージ点間のひずみをリアルタイムグラフ表示
- **マップモード**: フルフィールドひずみマップをバックグラウンド計算・スライダーで閲覧
- **リアルタイムマップモード**: USBカメラ（1080p/30fps）から直接取得し、引張試験中のひずみマップをライブ表示
  - 前フレーム変位を予測値にした狭探索＋必要時のみ自動拡張、相関低下時の自動再参照
  - カラースケールは試験中固定（0–2 / 5 / 10 / 20 % プリセット）でひずみ集中の成長を可視化
  - CUDA（CuPy）があれば GPU、なければ CPU に自動フォールバック。CPU でも 1080p・約1,800点で 20 fps 以上
  - 一定間隔のスナップショット保存と停止後の見返し
  - タブを開くと自動でカメラを検索し、外付けカメラ（内蔵 Web カメラ以外）を優先して映像が即座に表示される
  - カメラが開けなくなったとき（open に 20 秒以上かかる）は USB を抜き差ししてから「カメラ検索」
- ZNCC（cv2.matchTemplate使用・高速化済み）
- サブピクセル補間（parabolic / gaussian / spline）
- **IC-GN サブピクセル精密化**（1次形状関数＋B-spline補間。ピクセルロッキングを 0.03px → 0.0005px 級に低減。マップ・リアルタイムモード）
- 固定参照 / 更新参照モード
- 累積ひずみ / 差分ひずみ 切り替え表示
- 手動アライメント（半透明ドラッグ）
- スペックル品質評価（MIG・SSSIGマップ・スペックルサイズ3×3）
- フレームテーブル（荷重・応力手入力、MIG列）
- CSV出力

---

## ダウンロード・使い方

👉 **[Releases](https://github.com/DefraTanakaLab/LiveDIC/releases)** から最新の `LiveDIC_win64.zip` をダウンロードしてください。

1. zipを解凍する
2. `LiveDIC.exe` をダブルクリックして起動
3. インストール不要・Python不要

> **動作環境**: Windows 10/11 64bit

---

## ソースから実行する場合

```bash
pip install -r requirements.txt
python main.py
```

リアルタイムマップを GPU で動かす場合（NVIDIA GPU、ドライバは CUDA 12 以降対応のもの）:

```bash
pip install "cupy-cuda12x[ctk]"
python check_gpu.py
```

`[ctk]` で CUDA ランタイム・ヘッダも pip から入ります（CUDA Toolkit の別途インストールは不要）。
`check_gpu.py` が「GPU 経路は CPU と一致しています」と出れば OK。
起動時に自動判定され、タブ内の「バックエンド」に GPU 名が表示されます。

## exeをビルドする場合

```bash
python make_icon.py   # アイコン更新時のみ
build.bat
```

---

## ファイル構成

| ファイル | 役割 |
|---|---|
| `main.py` | PyQt6 GUI メイン |
| `dic_core.py` | DIC計算コアモジュール |
| `dic_icgn.py` | IC-GN サブピクセル精密化（CuPy/NumPy 共通） |
| `dic_rt.py` | リアルタイムマップ用トラッカー（CuPy/NumPy 共通） |
| `rt_source.py` | USBカメラ取得（テスト用の合成ソースも含む） |
| `rt_panel.py` | リアルタイムマップタブ UI |
| `check_gpu.py` | GPU 環境で CuPy 経路の動作・数値一致・速度を確認 |
| `build.bat` | PyInstaller exeビルドスクリプト |
| `make_icon.py` | アイコン生成スクリプト |
| `requirements.txt` | 依存パッケージ |
| `Demo_2points_mode/` | 2点間モード用デモ画像 |
| `Demo_map_mode/` | マップモード用デモ画像 |

---

## License

MIT
