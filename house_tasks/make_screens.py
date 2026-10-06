"""House tasks on the Nextion screen (NX4832K035, 480 x 320).

Draws the screen's pictures from tasks.json and writes nextion_code.txt:
which parts go where in Nextion Editor, with the code for each.
    python house_tasks/make_screens.py
Needs Pillow (pip install pillow). See README.txt.
"""
import json
import math
import os
import sys
from functools import lru_cache

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
SCREENS = os.path.join(HERE, "screens")

W, H = 480, 320       # the screen, lying down
S = 3                 # shapes are drawn 3x bigger, then shrunk: smooth edges

# Main page: a header with the points, then 3 x 4 task tiles
HEADER = 48
MARGIN, GAP = 8, 8
COLS, ROWS = 3, 4
TILE_W = (W - 2 * MARGIN - (COLS - 1) * GAP) // COLS             # 149
TILE_H = (H - HEADER - 2 * MARGIN - (ROWS - 1) * GAP) // ROWS    # 58
MAX_TASKS = COLS * ROWS
POINTS_BOX = (326, 6, 96, 36)     # the number (n0): x, y, w, h
MENU_BOX = (428, 0, 52, 48)       # the menu button (b0)
POINTS_FONT = 32                  # height of the font made in Nextion Editor

# Menu page and the "points back to 0?" page
MENU_BUTTONS = [(16, 66, 448, 72), (16, 148, 448, 72), (16, 230, 448, 72)]
CONFIRM_NO = (16, 214, 216, 72)
CONFIRM_YES = (248, 214, 216, 72)

# The screen's own memory (EEPROM, 1 KB, kept without power).
# Every number takes 4 bytes.
ADDR_POINTS = 0
ADDR_TICK0 = 4        # the tick of task i is at 4 + 4*i
ADDR_MARK = 100       # holds MARK once the memory is set up
MARK = 4271

BG = (243, 239, 232)
HEADER_BG = (36, 50, 74)
HEADER_PRESSED = (62, 82, 116)
WHITE = (255, 255, 255)
CARD_EDGE = (222, 214, 201)
CARD_PRESSED = (228, 221, 209)
INK = (36, 40, 48)
MUTED = (112, 107, 99)
STAR = (240, 168, 36)
GREEN = (30, 132, 73)
RING = (200, 192, 179)
RED = (200, 58, 54)
RED_PRESSED = (160, 40, 37)

PICTURES = [
    ("0_main.png", "main page, every task open"),
    ("1_main_done.png", "main page, every task done + menu pressed"),
    ("2_menu.png", "menu page"),
    ("3_menu_pressed.png", "menu page, buttons pressed"),
    ("4_confirm.png", "points back to 0?"),
    ("5_confirm_pressed.png", "points back to 0?, buttons pressed"),
]

# Segoe UI on Windows, Inter or DejaVu elsewhere
FONTS = {
    "bold": ["segoeuib.ttf", "Inter-Bold.otf", "DejaVuSans-Bold.ttf"],
    "semi": ["seguisb.ttf", "Inter-SemiBold.otf", "DejaVuSans-Bold.ttf"],
    "regular": ["segoeui.ttf", "Inter-Regular.otf", "DejaVuSans.ttf"],
}


@lru_cache(maxsize=None)
def font(kind, size):
    for name in FONTS[kind]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    sys.exit("No font found: install Segoe UI, Inter or DejaVu Sans")


class Canvas:
    """One screen picture. Shapes are drawn S times bigger and shrunk
    (smooth edges); text is drawn after that, at its real size (sharp)."""

    def __init__(self):
        self.big = Image.new("RGB", (W * S, H * S), BG)
        self.draw = ImageDraw.Draw(self.big)
        self.texts = []

    def rect(self, x, y, w, h, fill, radius=0, outline=None):
        self.draw.rounded_rectangle(
            [x * S, y * S, (x + w) * S - 1, (y + h) * S - 1], radius * S,
            fill=fill, outline=outline, width=S if outline else 0)

    def circle(self, cx, cy, r, fill=None, outline=None, width=0):
        self.draw.ellipse([(cx - r) * S, (cy - r) * S, (cx + r) * S, (cy + r) * S],
                          fill=fill, outline=outline, width=round(width * S))

    def poly(self, points, fill):
        self.draw.polygon([(px * S, py * S) for px, py in points], fill=fill)

    def line(self, points, fill, width):
        w = round(width * S)
        self.draw.line([(px * S, py * S) for px, py in points], fill=fill,
                       width=w, joint="curve")
        for px, py in (points[0], points[-1]):           # round ends
            self.draw.ellipse([px * S - w / 2, py * S - w / 2,
                               px * S + w / 2, py * S + w / 2], fill=fill)

    def text(self, xy, s, f, fill, anchor="la"):
        self.texts.append((xy, s, f, fill, anchor))

    def image(self):
        img = self.big.reduce(S)
        d = ImageDraw.Draw(img)
        for xy, s, f, fill, anchor in self.texts:
            d.text(xy, s, font=f, fill=fill, anchor=anchor)
        return img


def star(c, cx, cy, r, fill):
    points = []
    for k in range(10):
        a = math.pi / 5 * k - math.pi / 2
        rr = r if k % 2 == 0 else r * 0.48
        points.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    c.poly(points, fill)


def tick(c, cx, cy, r, color):
    c.line([(cx - .45 * r, cy + .02 * r), (cx - .12 * r, cy + .34 * r),
            (cx + .46 * r, cy - .32 * r)], color, .24 * r)


def tile_box(i):
    col, row = i % COLS, i // COLS
    return (MARGIN + col * (TILE_W + GAP), HEADER + MARGIN + row * (TILE_H + GAP),
            TILE_W, TILE_H)


def name_font(tasks):
    """The biggest size (17 down to 14) at which every task name fits its tile."""
    room = TILE_W - 20
    for size in range(17, 13, -1):
        f = font("bold", size)
        if all(f.getlength(t["name"]) <= room for t in tasks):
            return f
    f = font("bold", 14)
    long = [t["name"] for t in tasks if f.getlength(t["name"]) > room]
    sys.exit("Too long for a tile, make them shorter: " + ", ".join(long))


def draw_tile(c, i, task, done, nfont):
    x, y, w, h = tile_box(i)
    if done:
        c.rect(x, y, w, h, GREEN, radius=10)
    else:
        c.rect(x, y, w, h, WHITE, radius=10, outline=CARD_EDGE)
    ink = WHITE if done else INK
    c.text((x + 10, y + 9), task["name"], nfont, ink)
    star(c, x + 16, y + h - 15, 6.5, WHITE if done else STAR)
    c.text((x + 26, y + h - 15), str(task["points"]), font("semi", 14), ink, "lm")
    cx, cy, r = x + w - 18, y + h - 17, 10
    if done:
        c.circle(cx, cy, r, fill=WHITE)
        tick(c, cx, cy, r, GREEN)
    else:
        c.circle(cx, cy, r, outline=RING, width=2)


def draw_header(c, title):
    c.rect(0, 0, W, HEADER, HEADER_BG)
    c.text((14, HEADER // 2), title, font("bold", 22), WHITE, "lm")


def draw_main(title, tasks, done=(), menu_pressed=False, points=None):
    """done: the numbers of the tasks drawn as done. points: only for the
    preview (on the screen the number is n0)."""
    c = Canvas()
    draw_header(c, title)
    star(c, POINTS_BOX[0] - 15, HEADER // 2, 11, STAR)
    if points is not None:
        c.text((POINTS_BOX[0], HEADER // 2), str(points), font("bold", 26), WHITE, "lm")
    x, y, w, h = MENU_BOX
    if menu_pressed:
        c.rect(x + 4, y + 4, w - 8, h - 8, HEADER_PRESSED, radius=8)
    cx, cy = x + w / 2, y + h / 2
    for dy in (-7, 0, 7):
        c.rect(cx - 11, cy + dy - 1.5, 22, 3, WHITE, radius=1.5)
    nfont = name_font(tasks)
    for i, task in enumerate(tasks):
        draw_tile(c, i, task, i in done, nfont)
    return c.image()


def card_button(c, box, title, sub, pressed):
    x, y, w, h = box
    c.rect(x, y, w, h, CARD_PRESSED if pressed else WHITE, radius=12, outline=CARD_EDGE)
    c.text((x + 18, y + 26), title, font("bold", 19), INK, "lm")
    c.text((x + 18, y + 51), sub, font("regular", 14), MUTED, "lm")


def solid_button(c, box, label, fill, text_color, outline=None):
    x, y, w, h = box
    c.rect(x, y, w, h, fill, radius=12, outline=outline)
    c.text((x + w / 2, y + h / 2), label, font("bold", 18), text_color, "mm")


def draw_menu(pressed=False):
    c = Canvas()
    draw_header(c, "Menu")
    card_button(c, MENU_BUTTONS[0], "New day",
                "Clear all ticks. The points stay.", pressed)
    card_button(c, MENU_BUTTONS[1], "Points back to 0",
                "After the points are spent on a reward.", pressed)
    solid_button(c, MENU_BUTTONS[2], "Back to the tasks",
                 HEADER_PRESSED if pressed else HEADER_BG, WHITE)
    return c.image()


def draw_confirm(pressed=False):
    c = Canvas()
    draw_header(c, "Points back to 0")
    c.text((W / 2, 122), "Set the points back to 0?", font("bold", 24), INK, "mm")
    c.text((W / 2, 158), "This can't be undone.", font("regular", 16), MUTED, "mm")
    solid_button(c, CONFIRM_NO, "No, keep them",
                 CARD_PRESSED if pressed else WHITE, INK, CARD_EDGE)
    solid_button(c, CONFIRM_YES, "Yes, back to 0",
                 RED_PRESSED if pressed else RED, WHITE)
    return c.image()


def rgb565(color):
    r, g, b = color
    return (r >> 3) << 11 | (g >> 2) << 5 | b >> 3


# ---------------------------------------------------------------- the code

def block(lines, indent="      "):
    return [indent + line for line in lines]


def if_else(cond, yes, no=None):
    out = [f"if({cond})", "{"] + ["  " + line for line in yes]
    if no:
        out += ["}else", "{"] + ["  " + line for line in no]
    return out + ["}"]


def tick_addr(i):
    return ADDR_TICK0 + 4 * i


def clear_ticks():
    """Every tick slot to 0, also the ones without a task (yet)."""
    return [f"wepo sys0,{tick_addr(i)}" for i in range(MAX_TASKS)]


def component(name, kind, box, attrs):
    x, y, w, h = box
    return [f"  {name:<5}{kind}", f"       x={x}  y={y}  w={w}  h={h}",
            f"       {attrs}"]


def nextion_code(title, tasks):
    out = [
        f"{title.upper()} - WHAT GOES WHERE IN NEXTION EDITOR",
        "Made by make_screens.py from tasks.json: edit those, not this file.",
        "Type everything exactly as here: no spaces around = + - == !=.",
        "",
        "PICTURES (add them in this order, so the numbers match)",
    ]
    out += [f"  {i}  screens\\{f:<22} {what}" for i, (f, what) in enumerate(PICTURES)]
    out += [
        "",
        "FONT",
        f"  0  for the points: height {POINTS_FONT}, bold (README.txt, step 2)",
        "",
        "Every Button and Dual-state button: delete the text in txt (\"newtxt\").",
        "",
        "=" * 72,
        "PAGE main   (page 0)",
        "=" * 72,
        "  The page itself: sta = image, pic = 0",
        "",
        "  Preinitialize Event (click an empty spot of the page):",
    ]
    first_time = ["// the first time: everything at 0", "sys0=0",
                  f"wepo sys0,{ADDR_POINTS}"] + clear_ticks() + [
                  f"sys0={MARK}", f"wepo sys0,{ADDR_MARK}"]
    pre = ["// read the points and ticks from the screen's memory",
           f"repo sys0,{ADDR_MARK}"] + if_else(f"sys0!={MARK}", first_time) + [
           f"repo n0.val,{ADDR_POINTS}"]
    pre += [f"repo bt{i}.val,{tick_addr(i)}" for i in range(len(tasks))]
    out += block(pre)
    out += [""]
    out += component("n0", "Number (the points)", POINTS_BOX,
                     f"sta=crop image  picc=0  font=0  pco={rgb565(WHITE)}  "
                     "xcen=Left  ycen=Center")
    out += [""]
    out += component("b0", "Button (menu)", MENU_BOX,
                     "sta=crop image  picc=0  picc2=1  txt=(empty)")
    out += ["       Touch Release Event:"] + block(["page menu"], "         ")
    out += ["", "  The tasks, one Dual-state button each (make bt0, then copy and",
            "  paste it: change x, y and the numbers in its code):"]
    for i, task in enumerate(tasks):
        p = task["points"]
        bt = f"bt{i}"
        code = if_else(f"{bt}.val==1", [f"n0.val=n0.val+{p}"],
                       [f"n0.val=n0.val-{p}"] + if_else("n0.val<0", ["n0.val=0"]))
        code += [f"wepo {bt}.val,{tick_addr(i)}", f"wepo n0.val,{ADDR_POINTS}"]
        out += [""]
        out += component(bt, f"Dual-state button: {task['name']} ({p} points)",
                         tile_box(i), "sta=crop image  picc0=0  picc1=1  txt=(empty)")
        out += ["       Touch Release Event:"] + block(code, "         ")

    out += ["", "=" * 72, "PAGE menu   (page 1)", "=" * 72,
            "  The page itself: sta = image, pic = 2", ""]
    menu = [("b0", "New day", ["sys0=0"] + clear_ticks() + ["page main"]),
            ("b1", "Points back to 0", ["page confirm"]),
            ("b2", "Back to the tasks", ["page main"])]
    for (name, label, code), box in zip(menu, MENU_BUTTONS):
        out += component(name, f"Button: {label}", box,
                         "sta=crop image  picc=2  picc2=3  txt=(empty)")
        out += ["       Touch Release Event:"] + block(code, "         ") + [""]

    out += ["=" * 72, "PAGE confirm   (page 2)", "=" * 72,
            "  The page itself: sta = image, pic = 4", ""]
    confirm = [("b0", "No, keep them", CONFIRM_NO, ["page menu"]),
               ("b1", "Yes, back to 0", CONFIRM_YES,
                ["sys0=0", f"wepo sys0,{ADDR_POINTS}", "page main"])]
    for name, label, box, code in confirm:
        out += component(name, f"Button: {label}", box,
                         "sta=crop image  picc=4  picc2=5  txt=(empty)")
        out += ["       Touch Release Event:"] + block(code, "         ") + [""]
    return "\n".join(out)


def load_tasks():
    with open(os.path.join(HERE, "tasks.json"), encoding="utf-8") as f:
        data = json.load(f)
    tasks = data["tasks"]
    if not 1 <= len(tasks) <= MAX_TASKS:
        sys.exit(f"tasks.json: 1 to {MAX_TASKS} tasks, not {len(tasks)}")
    for t in tasks:
        if not (isinstance(t.get("points"), int) and 1 <= t["points"] <= 999):
            sys.exit(f"tasks.json: points must be a whole number 1-999 ({t.get('name')})")
    return data.get("title", "House tasks"), tasks


def main():
    title, tasks = load_tasks()
    os.makedirs(SCREENS, exist_ok=True)
    every = range(len(tasks))
    pictures = [draw_main(title, tasks),
                draw_main(title, tasks, done=every, menu_pressed=True),
                draw_menu(), draw_menu(pressed=True),
                draw_confirm(), draw_confirm(pressed=True)]
    for (name, _), img in zip(PICTURES, pictures):
        img.save(os.path.join(SCREENS, name))

    # How it looks with a few tasks done (only to look at)
    done = {i for i in (0, 1, 5) if i < len(tasks)}
    shots = [draw_main(title, tasks, done=done,
                       points=sum(tasks[i]["points"] for i in done)),
             draw_menu(), draw_confirm()]
    preview = Image.new("RGB", (3 * W + 4 * 16, H + 32), (60, 60, 60))
    for k, img in enumerate(shots):
        preview.paste(img, (16 + k * (W + 16), 16))
    preview.save(os.path.join(HERE, "preview.png"))

    with open(os.path.join(HERE, "nextion_code.txt"), "w", encoding="utf-8") as f:
        f.write(nextion_code(title, tasks) + "\n")
    print(f"{len(tasks)} tasks: screens\\ (6 pictures), nextion_code.txt, preview.png")


if __name__ == "__main__":
    main()
