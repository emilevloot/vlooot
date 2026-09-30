"""
Draw the app icons (icons/*.png) for the installable web app: a round
Viking shield with a bearded axe behind it, on the dark fjord colour of the
game. Shields and axes are what the game is about: the shields are your
special moves, axes win the trophies.

    python tools/make_icons.py
"""

import math
import os

from PIL import Image, ImageDraw, ImageFilter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "icons")

NIGHT = (20, 29, 37)          # --night
SLATE = (35, 56, 74)          # the glow behind the board
INK = (13, 17, 21)
RED = (210, 74, 58)           # the red player
CREAM = (239, 230, 207)
RIM = (92, 103, 112)
STEEL = (201, 209, 214)
STEEL_HI = (238, 242, 244)
WOOD = (122, 84, 51)
WOOD_DARK = (58, 38, 22)


def rot(points, angle, cx, cy):
    """Turn points (x, y) around (cx, cy) by angle degrees (clockwise on screen)."""
    a = math.radians(angle)
    ca, sa = math.cos(a), math.sin(a)
    return [(cx + (x - cx) * ca - (y - cy) * sa, cy + (x - cx) * sa + (y - cy) * ca) for x, y in points]


def arc(cx, cy, r, a0, a1, n=24):
    return [(cx + r * math.cos(math.radians(a0 + (a1 - a0) * i / n)),
             cy + r * math.sin(math.radians(a0 + (a1 - a0) * i / n))) for i in range(n + 1)]


def axe(d, cx, cy, L, angle, w):
    """A bearded axe: the handle through (cx, cy), length L, turned by angle;
    the blade at the top, its edge facing right, the beard hanging down."""
    hw = L * 0.034                                   # half the handle's width
    top, bottom = cy - L * 0.52, cy + L * 0.48
    handle = [(cx - hw, bottom), (cx - hw, top + hw), (cx, top), (cx + hw, top + hw), (cx + hw, bottom)]
    d.polygon(rot(handle, angle, cx, cy), fill=WOOD, outline=WOOD_DARK, width=w)
    # the blade (in handle coordinates: x to the right of the handle, y down)
    bx, by = cx + hw, top + L * 0.06
    s = L
    blade = ([(bx, by - s * 0.02), (bx + s * 0.10, by - s * 0.05), (bx + s * 0.22, by - s * 0.09)] +
             arc(bx + s * 0.03, by + s * 0.10, s * 0.265, -44, 58, 20) +        # the cutting edge
             [(bx + s * 0.13, by + s * 0.28), (bx + s * 0.07, by + s * 0.16), (bx, by + s * 0.12)])
    d.polygon(rot(blade, angle, cx, cy), fill=STEEL, outline=INK, width=w)
    edge = arc(bx + s * 0.03, by + s * 0.10, s * 0.235, -38, 52, 16)
    d.line(rot(edge, angle, cx, cy), fill=STEEL_HI, width=max(2, w))
    # the handle's end below the shield
    kx, ky = rot([(cx, bottom)], angle, cx, cy)[0]
    d.ellipse([kx - hw * 1.3, ky - hw * 1.3, kx + hw * 1.3, ky + hw * 1.3], fill=WOOD_DARK)


def shield(d, cx, cy, r, w):
    """A round Viking shield: 4 painted planks, an iron rim with rivets and a boss."""
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=RIM, outline=INK, width=w)
    f = r * 0.86
    box = [cx - f, cy - f, cx + f, cy + f]
    for k in range(4):
        d.pieslice(box, 45 + 90 * k, 135 + 90 * k, fill=RED if k % 2 == 0 else CREAM)
    for k in range(4):                                # the planks' seams
        a = math.radians(45 + 90 * k)
        d.line([(cx, cy), (cx + f * math.cos(a), cy + f * math.sin(a))], fill=INK, width=max(1, w // 2))
    d.ellipse(box, outline=INK, width=max(1, w // 2))
    for k in range(12):                               # rivets on the rim
        a = math.radians(15 + 30 * k)
        x, y, q = cx + r * 0.93 * math.cos(a), cy + r * 0.93 * math.sin(a), r * 0.035
        d.ellipse([x - q, y - q, x + q, y + q], fill=STEEL)
    b = r * 0.24                                      # the boss in the middle
    d.ellipse([cx - b, cy - b, cx + b, cy + b], fill=STEEL, outline=INK, width=w)
    d.arc([cx - b * 0.7, cy - b * 0.7, cx + b * 0.7, cy + b * 0.7], 200, 290, fill=STEEL_HI, width=max(2, w))


def art(size, full_bleed):
    """The icon at 4x, then made smaller (smooth edges)."""
    S = size * 4
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    bg = Image.new("RGBA", (S, S), NIGHT + (255,))
    glow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    ImageDraw.Draw(glow).ellipse([S * 0.05, -S * 0.25, S * 1.15, S * 0.75], fill=SLATE + (255,))
    bg = Image.alpha_composite(bg, glow.filter(ImageFilter.GaussianBlur(S * 0.12)))
    mask = Image.new("L", (S, S), 0)
    if full_bleed:
        mask.paste(255, (0, 0, S, S))
    else:
        ImageDraw.Draw(mask).rounded_rectangle([0, 0, S - 1, S - 1], radius=int(S * 0.22), fill=255)
    img.paste(bg, (0, 0), mask)
    d = ImageDraw.Draw(img)
    # a maskable icon keeps its drawing inside the middle 80% (launchers cut the rest)
    k = 0.74 if full_bleed else 0.94
    w = max(2, int(S * 0.009))
    cx, cy = S * 0.5, S * 0.5
    axe(d, cx - S * 0.01 * k, cy - S * 0.03 * k, S * 0.74 * k, 35, w)
    shield(d, cx - S * 0.07 * k, cy + S * 0.06 * k, S * 0.31 * k, w)
    return img.resize((size, size), Image.LANCZOS)


def main():
    os.makedirs(OUT, exist_ok=True)
    for size in (192, 512):
        art(size, False).save(os.path.join(OUT, "icon-%d.png" % size))
        art(size, True).save(os.path.join(OUT, "maskable-%d.png" % size))
    art(180, True).convert("RGB").save(os.path.join(OUT, "apple-touch-icon.png"))   # iPhone: no see-through
    art(64, False).save(os.path.join(OUT, "favicon-64.png"))
    art(32, False).save(os.path.join(OUT, "favicon-32.png"))
    print("icons written to", OUT)


if __name__ == "__main__":
    main()
