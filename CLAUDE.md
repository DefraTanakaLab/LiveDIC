# CLAUDE.md — LiveDIC プロジェクト

## プロジェクト概要

インターバル撮影SEM画像のリアルタイムDIC解析ツール。
フォルダを監視し、新しい画像が来るたびに自動でDIC計算・表示を更新する。

### ファイル構成
| ファイル | 役割 |
|---|---|
| `main.py` | PyQt6 GUI メイン |
| `dic_core.py` | DIC計算コアモジュール |
| `build.bat` | PyInstaller exe ビルドスクリプト |
| `make_icon.py` | cow.png → LiveDIC.ico 変換スクリプト |
| `requirements.txt` | 依存パッケージ |

### モード
- **2点間モード**: 登録ゲージ点のひずみをリアルタイムグラフ表示
- **マップモード**: フルフィールドひずみマップをバックグラウンド計算・スライダーで閲覧

---

## 実装済み機能

- ZNCC（cv2.matchTemplate使用・高速化済み）
- サブピクセル補間（parabolic / gaussian / spline）
- 固定参照 / 更新参照モード
- 累積ひずみ / 差分ひずみ 切り替え表示
- 手動アライメント（半透明ドラッグ）
- 最新フレーム再解析（アライメント含む）
- スペックル品質評価（MIG・SSSIGマップ・スペックルサイズ3×3）
- フレームテーブル（荷重・応力手入力、MIG列）
- CSV出力

---

## exe ビルド

```
python make_icon.py   # アイコン更新時のみ
build.bat
```

- PyInstaller `--onedir --windowed`
- `--exclude-module PyQt5`（PyQt5/6共存環境対策）
- アイコン：`C:/Users/yamas/AppData/Local/Temp/LiveDIC.ico`（日本語パス回避）

### exe での制限事項
- **joblib並列処理は無効**（`_FROZEN` フラグで制御）。cv2.matchTemplateで十分高速。
- GUIの「コア数」設定は exe では非表示になる。

---

## 注意事項

- `dic_core.py` はビルド時に `--add-data=dic_core.py:.` で同梱される
- 画像・科学データの表示で補間・美化は禁止（`FastTransformation` のみ使用）
- PyInstaller用にすべてのimportはファイル先頭に記述すること（関数内importはNG）
- `_FROZEN = getattr(sys, 'frozen', False)` で exe/スクリプト実行を判別できる
