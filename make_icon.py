from PIL import Image
import pathlib

here = pathlib.Path(__file__).parent
img  = Image.open(here / "cow.png").convert("RGBA")

w, h = img.size
size = max(w, h)
square = Image.new("RGBA", (size, size), (0, 0, 0, 0))
square.paste(img, ((size - w) // 2, (size - h) // 2))

square.save(here / "LiveDIC.ico", format="ICO",
            sizes=[(256,256),(128,128),(64,64),(48,48),(32,32),(24,24),(16,16)])
print("LiveDIC.ico 作成完了")
