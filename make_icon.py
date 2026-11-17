"""cow.png → LiveDIC.ico（PyInstaller の --icon 用）。

出力先は一時フォルダ。PyInstaller の --icon は日本語やスペースを含むパスで失敗することが
あるので、リポジトリの置き場所に依存しない %TEMP% に置いて build.bat から参照する。
"""
import os
import pathlib
from PIL import Image

here = pathlib.Path(__file__).parent
out = pathlib.Path(os.environ.get("TEMP", here)) / "LiveDIC.ico"
img  = Image.open(here / "cow.png").convert("RGBA")

w, h = img.size
size = max(w, h)
square = Image.new("RGBA", (size, size), (0, 0, 0, 0))
square.paste(img, ((size - w) // 2, (size - h) // 2))

square.save(out, format="ICO",
            sizes=[(256,256),(128,128),(64,64),(48,48),(32,32),(24,24),(16,16)])
print(f"LiveDIC.ico 作成完了: {out}")
