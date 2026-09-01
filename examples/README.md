Minimal usage without datasets or GPUs:

```bash
python embed.py codec encode cover.png stego.png --text "hello" --key 1234
python embed.py codec decode stego.png --key 1234
```

Create a dummy cover first:

```python
from PIL import Image
Image.new("RGB", (128, 128), (32, 64, 96)).save("cover.png")
```
