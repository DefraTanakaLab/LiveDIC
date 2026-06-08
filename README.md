# LiveDIC

インターバル撮影した画像のリアルタイムDIC（デジタル画像相関法）解析ツール。

フォルダを監視し、新しい画像が来るたびに自動でDIC計算・表示を更新します。

---

## 主な機能

- **2点間モード**: 登録したゲージ点間のひずみをリアルタイムグラフ表示
- **マップモード**: フルフィールドひずみマップをバックグラウンド計算・スライダーで閲覧
- ZNCC（cv2.matchTemplate使用・高速化済み）
- サブピクセル補間（parabolic / gaussian / spline）
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
| `build.bat` | PyInstaller exeビルドスクリプト |
| `make_icon.py` | アイコン生成スクリプト |
| `requirements.txt` | 依存パッケージ |
| `Demo_2points_mode/` | 2点間モード用デモ画像 |
| `Demo_map_mode/` | マップモード用デモ画像 |

---

## License

MIT
