"""Draw the Batscribe quill icon: white feather quill on the blue rounded tile.

Writes batscribe.png and batscribe.ico next to this script. Needs Pillow
(not in requirements.txt; any Python with Pillow will do): python make_icon.py
"""
import math
from pathlib import Path

from PIL import Image, ImageDraw

S = 1024  # supersampled canvas
BLUE = (37, 99, 195, 255)
WHITE = (255, 255, 255, 255)
out = Path(__file__).parent

img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
d = ImageDraw.Draw(img)
m = S * 10 / 256
d.rounded_rectangle([m, m, S - m, S - m], radius=S * 30 / 256, fill=BLUE)

# Feather in local coords: u along shaft (0 = nib tip, 1 = top), v across.
ANG = math.radians(-45)  # shaft runs bottom-left -> top-right
cx0, cy0 = S * 0.21, S * 0.80  # nib tip
L = S * 0.86


def P(u, v):
    x, y = u * L, v * L
    return (cx0 + x * math.cos(ANG) - y * math.sin(ANG),
            cy0 + x * math.sin(ANG) + y * math.cos(ANG))


def edge(u, side):
    # vane starts at u=0.30, widest ~0.62, closes at the top (u=1)
    t = (u - 0.30) / 0.70
    w = math.sin(math.pi * t ** 0.8) * (0.15 if side < 0 else 0.11)
    return side * w + 0.03 * t * t  # slight curve to one side


N = 60
us = [0.30 + 0.70 * i / N for i in range(N + 1)]
vane = [P(u, edge(u, -1)) for u in us] + [P(u, edge(u, 1)) for u in reversed(us)]
d.polygon(vane, fill=WHITE)

# barb notches cut into each edge
for side, notches in ((-1, (0.50, 0.68)), (1, (0.58,))):
    for u in notches:
        a = P(u, edge(u, side) * 1.05)
        b = P(u - 0.10, edge(u - 0.10, side) * 1.2)
        c = P(u - 0.02, edge(u, side) * 0.45)
        d.polygon([a, b, c], fill=BLUE)

# shaft cut through the vane, then the quill/nib below it
d.line([P(0.34, 0.0), P(0.92, 0.03 * ((0.92 - 0.30) / 0.70) ** 2)],
       fill=BLUE, width=int(S * 0.018))
d.polygon([P(0.0, 0.0), P(0.12, -0.022), P(0.32, -0.02),
           P(0.32, 0.02), P(0.12, 0.022)], fill=WHITE)

# small ink stroke under the nib
d.rounded_rectangle([S * 0.19, S * 0.79, S * 0.62, S * 0.825],
                    radius=S * 0.02, fill=WHITE)

big = img.resize((256, 256), Image.LANCZOS)
big.save(out / "batscribe.png")
big.save(out / "batscribe.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (256, 256)])
