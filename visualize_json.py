import json
import base64
from PIL import Image
from io import BytesIO

# JSON dosyanı oku
with open(r"C:\Users\AysimaAkkurt\Downloads\response_1777881158165.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# warped_image alanını al
img_base64 = data["warped_image"]

# Eğer başında data:image/png;base64, varsa temizle
if "," in img_base64:
    img_base64 = img_base64.split(",", 1)[1]

# Base64 -> image
img_bytes = base64.b64decode(img_base64)
img = Image.open(BytesIO(img_bytes))

# Göster
img.show()

# Kaydet
img.save("warped_image.png")
print("Saved as warped_image.png")