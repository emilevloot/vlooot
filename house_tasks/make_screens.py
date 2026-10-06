"""Onze huistaken on the Nextion screen (NX4832K035, 480 x 320).

The house tasks app on the screen by itself: two people with a score per
week and month, "Samen gedaan", tasks that come back after their number of
days (with bonus points when late), and a savings goal. Like the app, but
the screen keeps its own list.

Makes Huistaken.HMI: the finished Nextion Editor project, with the
pictures and all the code. Open it in Nextion Editor and choose File > TFT
file output (README.txt). nextion_code.txt holds the same code to read.
    python house_tasks/make_screens.py
Needs Pillow (pip install pillow).

How the screen works: one page and a timer. Every 50 ms the timer draws
what changed (the task list, "wie", "terug", the menu or the clock page)
from pictures: whole screens with pic and picq, and every number and word
from strips of letters with xpic, so the project needs no fonts. A tap only
changes numbers; the next timer tick draws it.
"""
import json
import math
import os
import sys
from functools import lru_cache

from PIL import Image, ImageDraw, ImageFont

import nextion_hmi

HERE = os.path.dirname(os.path.abspath(__file__))
SCREENS = os.path.join(HERE, "screens")
PROJECT = os.path.join(HERE, "Huistaken.HMI")

W, H = 480, 320       # the screen, lying down
S = 3                 # shapes are drawn 3x bigger, then shrunk: smooth edges

# ---------------------------------------------------------------- layout
CARDS = [(8, 6, 196, 50), (210, 6, 196, 50)]     # the two people: x, y, w, h
MENU_BOX = (412, 6, 60, 50)
COLS, ROWS = 3, 4
TILE_W, TILE_H, GAP = 150, 58, 6
GRID_X = (W - COLS * TILE_W - (COLS - 1) * GAP) // 2    # 9
GRID_Y = 62
MAX_TASKS = COLS * ROWS
TILE_SHOW = (165, 50)                            # where a tile is shown again
INFO_Y = 116                                     # the line under that tile
WHO_BUTTONS = [(16, 144, 144, 84), (168, 144, 144, 84), (320, 144, 144, 84)]
WHO_CANCEL = (16, 240, 448, 64)
BACK_BUTTONS = [(16, 142, 448, 50), (16, 198, 448, 50), (16, 254, 448, 50)]
GOAL_CARD = (16, 50, 448, 100)
GOAL_BAR = (32, 106, 416, 14)
MENU_BUTTONS = [(16, 160, 448, 44), (16, 212, 448, 44), (16, 264, 448, 44)]
CLOCK_ROWS = [50, 92, 134, 176, 218]
CLOCK_ROW_H, CLOCK_PITCH = 38, 42
CLOCK_MINUS, CLOCK_VALUE, CLOCK_PLUS = (200, 60), (266, 120), (392, 60)  # x, w
CLOCK_BUTTONS = [(16, 266, 216, 46), (248, 266, 216, 46)]   # Annuleren, Opslaan
CLOCK_FIELDS = [("Dag", "gkd"), ("Maand", "gkm"), ("Jaar", "gkj"), ("Uur", "gku"),
                ("Minuut", "gkmi")]

# What the screen shows (gscherm)
TAKEN, WIE, TERUG, MENU, KLOK, TAPPED = 0, 1, 2, 3, 4, 6

# The pictures: whole screens, then the strips of letters
SCREEN_PICTURES = ["taken", "taken_gedaan", "taken_hoog", "taken_urgent",
                   "wie", "terug", "menu", "klok"]
P_OPEN, P_DONE, P_HIGH, P_URGENT, P_WIE, P_TERUG, P_MENU, P_KLOK = range(8)

# ---------------------------------------------------------------- memory
# The screen's own memory (EEPROM, 1 KB, kept without power), 4 bytes each.
A_MARK = 0            # holds the marker once the memory is set up
A_WEEK = (4, 8)       # points this week: person 1, person 2
A_MONTH = (12, 16)    # points this month
A_WEEK_KEY = 20       # the day number of this week's Monday
A_MONTH_KEY = 24      # the day number of the 1st of this month
A_GOAL = 28           # points saved for the goal
A_GOAL_NR = 32        # the goal's number: +1 for every new goal
A_TASKS = 64          # the day task i was last done: 64 + 4*i (0 = never)

# Every task has invisible Variables on the page, reached by number with
# b[...]: the day it was done, who, the points, the day before (for undo),
# the goal's number, and its two settings: every how many days, points.
TASK_VARS = ["vl", "vw", "vp", "vv", "vd", "ve", "vq"]

# The app's colours
BG = (242, 244, 239)
SURFACE = (255, 255, 255)
SUNK = (232, 236, 229)
INK = (28, 38, 32)
MUTED = (94, 107, 99)
LINE = (221, 227, 219)
ACCENT = (47, 107, 79)
PERSON = [(62, 99, 201), (194, 77, 122)]
LATE = (184, 67, 47)
SOON = (138, 106, 18)
LATE_TINT = (247, 237, 234)     # LATE at 7% on white


def mix(c1, c2, f):
    return tuple(round(a + (b - a) * f) for a, b in zip(c1, c2))


def rgb565(c):
    """The colour the screen really shows (16 bits)."""
    r, g, b = c
    return (r >> 3 << 3 | r >> 5, g >> 2 << 2 | g >> 6, b >> 3 << 3 | b >> 5)


FONT_FILES = {
    "display": "BricolageGrotesque-ExtraBold.ttf",
    "bold": "AtkinsonHyperlegible-Bold.ttf",
    "regular": "AtkinsonHyperlegible-Regular.ttf",
}


@lru_cache(maxsize=None)
def font(kind, size):
    return ImageFont.truetype(os.path.join(HERE, "fonts", FONT_FILES[kind]), size)


# ---------------------------------------------------------------- drawing
class Canvas:
    """One screen picture. Shapes are drawn S times bigger and shrunk
    (smooth edges); text is drawn after that, at its real size (sharp)."""

    def __init__(self):
        self.big = Image.new("RGB", (W * S, H * S), BG)
        self.draw = ImageDraw.Draw(self.big)
        self.texts = []

    def rect(self, x, y, w, h, fill, radius=0, outline=None, width=1):
        self.draw.rounded_rectangle(
            [x * S, y * S, (x + w) * S - 1, (y + h) * S - 1], radius * S,
            fill=fill, outline=outline, width=width * S if outline else 0)

    def circle(self, cx, cy, r, fill=None, outline=None, width=0):
        self.draw.ellipse([(cx - r) * S, (cy - r) * S, (cx + r) * S, (cy + r) * S],
                          fill=fill, outline=outline, width=round(width * S))

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
        return to565(img)


def to565(img):
    """Exactly the colours the screen can show (16 bits), so the editor and
    the screen change nothing."""
    r, g, b = img.convert("RGB").split()
    five = lambda v: (v >> 3 << 3) | (v >> 5)
    six = lambda v: (v >> 2 << 2) | (v >> 6)
    return Image.merge("RGB", (r.point(five), g.point(six), b.point(five)))


def tick(c, cx, cy, r, color):
    c.line([(cx - .45 * r, cy + .02 * r), (cx - .12 * r, cy + .34 * r),
            (cx + .46 * r, cy - .32 * r)], color, .24 * r)


def tile_box(i):
    col, row = i % COLS, i // COLS
    return (GRID_X + col * (TILE_W + GAP), GRID_Y + row * (TILE_H + GAP), TILE_W, TILE_H)


def bonus_points(task):
    """Points when a bit late (x1.5) and very late (x2), as in the app."""
    p = task["points"]
    return p + math.ceil(p / 2), 2 * p


def name_font(tasks):
    """The biggest size (15 down to 13) at which every task name fits."""
    room = TILE_W - 40
    for size in (15, 14, 13):
        f = font("bold", size)
        if all(f.getlength(t["name"]) <= room for t in tasks):
            return f
    f = font("bold", 13)
    long = [t["name"] for t in tasks if f.getlength(t["name"]) > room]
    sys.exit("Too long for a tile, make them shorter: " + ", ".join(long))


def draw_tile(c, i, task, state, nfont):
    x, y, w, h = tile_box(i)
    pts = {P_OPEN: task["points"], P_DONE: task["points"]}.get(state)
    if pts is None:
        pts = bonus_points(task)[state == P_URGENT]
    if state == P_DONE:
        c.rect(x, y, w, h, SUNK, radius=12)
    elif state == P_URGENT:
        c.rect(x, y, w, h, LATE_TINT, radius=12, outline=mix(LATE_TINT, LATE, .35))
    else:
        c.rect(x, y, w, h, SURFACE, radius=12, outline=LINE)
    if state in (P_HIGH, P_URGENT):                       # the app's side bar
        c.rect(x, y + 8, 4, h - 16, SOON if state == P_HIGH else LATE, radius=2)
    c.text((x + 11, y + 9), task["name"], nfont, MUTED if state == P_DONE else INK)
    cx, cy, r = x + w - 17, y + 17, 9                     # the tick circle
    if state == P_DONE:
        c.circle(cx, cy, r, fill=ACCENT)
        tick(c, cx, cy, r, SURFACE)
    else:
        ring = {P_HIGH: SOON, P_URGENT: LATE}.get(state, MUTED)
        c.circle(cx, cy, r, outline=ring, width=1.6)
    color = {P_HIGH: SOON, P_URGENT: LATE, P_DONE: MUTED}.get(state, INK)
    fill = {P_OPEN: SUNK, P_DONE: mix(SUNK, MUTED, .12)}.get(state, mix(SURFACE, color, .14))
    nf, uf = font("display", 14), font("regular", 11)     # the points, like the app's pill
    pw = nf.getlength(str(pts)) + uf.getlength("pt") + 14
    c.rect(x + 9, y + 34, pw, 18, fill, radius=7)
    c.text((x + 15, y + 43), str(pts), nf, color, "lm")
    c.text((x + 16 + nf.getlength(str(pts)), y + 44), "pt", uf, MUTED, "lm")


def draw_taken(cfg, state):
    c = Canvas()
    for k, (x, y, w, h) in enumerate(CARDS):
        c.rect(x, y, w, h, SURFACE, radius=14, outline=LINE)
        c.circle(x + 15, y + 15, 5, fill=PERSON[k])
        c.text((x + 25, y + 15), cfg["people"][k], font("bold", 15), INK, "lm")
        c.text((x + w - 10, y + 10), "DEZE WEEK", font("bold", 10), MUTED, "rm")
        st = STYLES["month"]
        c.text((x + 10, y + MONTH_DY + baseline(st)), "maand", style_font(st), st[2], "ls")
    x, y, w, h = MENU_BOX
    c.rect(x, y, w, h, SURFACE, radius=14, outline=LINE)
    for dy in (-7, 0, 7):
        c.rect(x + w / 2 - 11, y + h / 2 + dy - 1, 22, 2.4, MUTED, radius=1.2)
    nfont = name_font(cfg["tasks"])
    for i, task in enumerate(cfg["tasks"]):
        draw_tile(c, i, task, state, nfont)
    return c.image()


def title(c, text):
    c.text((24, 28), text, font("display", 22), INK, "lm")


def button(c, box, label, style="card", sub=None, dot=None):
    """style: card (white), primary (green), danger (red), off (greyed)."""
    x, y, w, h = box
    fill, ink, edge = {
        "card": (SURFACE, INK, LINE),
        "primary": (ACCENT, SURFACE, None),
        "danger": (LATE, SURFACE, None),
        "off": (SUNK, mix(MUTED, SUNK, .45), None),
    }[style]
    c.rect(x, y, w, h, fill, radius=14, outline=edge)
    f = font("bold", 17)
    cy = y + h / 2 - (10 if sub else 0)
    tx = x + w / 2
    if dot:
        tw = f.getlength(label) + 16
        c.circle(tx - tw / 2 + 5, cy, 5, fill=dot)
        tx += 8
    c.text((tx, cy), label, f, ink, "mm")
    if sub:
        c.text((x + w / 2, cy + 22), sub, font("regular", 13), MUTED, "mm")


def draw_wie(cfg):
    c = Canvas()
    title(c, "Wie heeft dit gedaan?")
    for k, box in enumerate(WHO_BUTTONS[:2]):
        button(c, box, cfg["people"][k], sub="krijgt de punten", dot=PERSON[k])
    button(c, WHO_BUTTONS[2], "Samen", sub="allebei de punten")
    button(c, WHO_CANCEL, "Annuleren")
    return c.image()


def draw_terug(cfg, undo_off=False):
    c = Canvas()
    title(c, "Al gedaan")
    labels = ["Afvinken ongedaan maken", "Nog een keer gedaan", "Terug naar de taken"]
    for k, (box, label) in enumerate(zip(BACK_BUTTONS, labels)):
        button(c, box, label, "off" if k == 0 and undo_off else "primary" if k == 2 else "card")
    return c.image()


def draw_menu(cfg, sure=False, full=False):
    c = Canvas()
    title(c, "Menu")
    x, y, w, h = GOAL_CARD
    c.rect(x, y, w, h, SURFACE, radius=16, outline=LINE)
    c.text((x + 16, y + 16), "SPAARDOEL", font("bold", 11), MUTED, "lm")
    c.text((x + 16, y + 38), cfg["goal"]["name"], font("display", 20), INK, "lm")
    st = STYLES["goal"]
    c.text((GOAL_RIGHT, GOAL_NUM_Y + baseline(st)), goal_suffix(cfg), style_font(st), st[2], "rs")
    bx, by, bw, bh = GOAL_BAR
    c.rect(bx, by, bw, bh, ACCENT if full else SUNK, radius=bh / 2)
    if sure:
        button(c, MENU_BUTTONS[0], "Zeker? Tik nog een keer", "danger")
    else:
        button(c, MENU_BUTTONS[0], "Nieuw spaardoel beginnen")
    button(c, MENU_BUTTONS[1], "Klok gelijkzetten")
    button(c, MENU_BUTTONS[2], "Terug naar de taken", "primary")
    return c.image()


def clock_box(row, part):
    x, w = {"minus": CLOCK_MINUS, "value": CLOCK_VALUE, "plus": CLOCK_PLUS}[part]
    return (x, CLOCK_ROWS[row], w, CLOCK_ROW_H)


def draw_klok(cfg):
    c = Canvas()
    title(c, "Klok gelijkzetten")
    for row, (label, _) in enumerate(CLOCK_FIELDS):
        y = CLOCK_ROWS[row]
        c.text((32, y + CLOCK_ROW_H / 2), label, font("bold", 17), INK, "lm")
        for part, sign in (("minus", "-"), ("plus", "+")):
            x, y0, w, h = clock_box(row, part)
            c.rect(x, y0, w, h, SURFACE, radius=12, outline=LINE)
            cx, cy = x + w / 2, y0 + h / 2
            c.rect(cx - 8, cy - 1.25, 16, 2.5, INK, radius=1.25)
            if sign == "+":
                c.rect(cx - 1.25, cy - 8, 2.5, 16, INK, radius=1.25)
        x, y0, w, h = clock_box(row, "value")
        c.rect(x, y0, w, h, SUNK, radius=12)
    button(c, CLOCK_BUTTONS[0], "Annuleren")
    button(c, CLOCK_BUTTONS[1], "Opslaan", "primary")
    return c.image()


# ---------------------------------------------------------------- letters
# Every changing number and word comes from a strip of letters in its own
# colours (style): font, size, colour, background, height.
STYLES = {
    "soon": ("regular", 13, SOON, SURFACE, 16),     # nooit gedaan, vandaag
    "done": ("regular", 13, MUTED, SUNK, 16),       # morgen, over 4 d
    "late": ("regular", 13, LATE, SURFACE, 16),     # 2 d te laat
    "late3": ("regular", 13, LATE, LATE_TINT, 16),  # on a very late tile
    "week0": ("display", 26, PERSON[0], SURFACE, 30),
    "week1": ("display", 26, PERSON[1], SURFACE, 30),
    "month": ("regular", 12, MUTED, SURFACE, 16),
    "info": ("regular", 15, INK, BG, 20),           # +6 punten, Gedaan door ...
    "infom": ("regular", 15, MUTED, BG, 20),
    "date": ("regular", 13, MUTED, BG, 16),
    "goal": ("bold", 15, INK, SURFACE, 20),
    "foot": ("regular", 13, MUTED, SURFACE, 16),
    "footok": ("regular", 13, ACCENT, SURFACE, 16),
    "klok": ("display", 26, INK, SUNK, 30),
}
MONTH_DY = 30          # the month line in a person's card (top, from the card's top)
WEEK_DY = 17           # the week number (top)
GOAL_RIGHT, GOAL_NUM_Y, GOAL_FOOT_Y = 448, 78, 128
DATE_RIGHT, DATE_Y = 464, 20
STATUS_RIGHT, STATUS_DY = 142, 35     # in a tile


def style_font(st):
    return font(st[0], st[1])


def baseline(st):
    asc, desc = style_font(st).getmetrics()
    return round((st[4] - asc - desc) / 2 + asc)


def text_width(style, text):
    return math.ceil(style_font(STYLES[style]).getlength(text))


@lru_cache(maxsize=None)
def digit_font(style):
    """The style's font with its equal-width digits (OpenType tnum, as the
    app shows them); without libraqm the plain digits, centred in a cell."""
    st = STYLES[style]
    try:
        f = ImageFont.truetype(os.path.join(HERE, "fonts", FONT_FILES[st[0]]), st[1],
                               layout_engine=ImageFont.Layout.RAQM)
        f.getlength("0", features=["tnum"])
        return f, ["tnum"]
    except (OSError, KeyError, ValueError):
        return style_font(st), None


def digit_width(style):
    f, features = digit_font(style)
    return math.ceil(max(f.getlength(d, features=features) for d in "0123456789") - 0.25)


def goal_suffix(cfg):
    return f" / {int(cfg['goal']['points'])} punten"


def words(cfg):
    a, b = cfg["people"]
    return [("soon", "nooit gedaan"), ("soon", "vandaag"), ("done", "morgen"),
            ("done", "over "), ("done", " d"), ("late", " d te laat"), ("late3", " d te laat"),
            ("info", "+"), ("info", " punten"), ("info", ", met "), ("info", " bonus"),
            ("info", f"Gedaan door {a}, +"), ("info", f"Gedaan door {b}, +"),
            ("info", "Samen gedaan, allebei +"),
            ("infom", "Nog een keer gedaan? Dat mag altijd."),
            ("date", ":"), ("date", "-"), ("foot", "Nog "), ("foot", " punten te gaan"),
            ("footok", "Gehaald! Tijd voor de beloning.")]


DIGIT_STYLES = ["done", "late", "late3", "week0", "week1", "month", "info", "date", "goal",
                "foot", "klok"]


def render_text(style, text, width=None):
    st = STYLES[style]
    w = width or text_width(style, text)
    img = Image.new("RGB", (max(w, 1), st[4]), st[3])
    ImageDraw.Draw(img).text((0, baseline(st)), text, font=style_font(st), fill=st[2], anchor="ls")
    return img


def render_digits(style):
    st, dw = STYLES[style], digit_width(style)
    f, features = digit_font(style)
    img = Image.new("RGB", (10 * dw, st[4]), st[3])
    d = ImageDraw.Draw(img)
    for k in range(10):
        d.text((k * dw + dw / 2, baseline(st)), str(k), font=f, fill=st[2], anchor="ms",
               features=features)
    return img


class Sprite:
    def __init__(self, pic, x, y, w, h):
        self.pic, self.x, self.y, self.w, self.h = pic, x, y, w, h

    def xpic(self, x, y, w=None):
        return f"xpic {x},{y},{w or self.w},{self.h},{self.x},{self.y},{self.pic}"


def make_sprites(cfg):
    """Packs every strip into as few 480 x 320 pictures as needed (shelf by
    shelf). Returns the pictures and {key: Sprite}; digits are keyed
    ("digits", style)."""
    items = [((style, text), render_text(style, text)) for style, text in words(cfg)]
    items += [(("digits", s), render_digits(s)) for s in DIGIT_STYLES]
    items += [("zeker", draw_menu(cfg, sure=True).crop(box_xy(MENU_BUTTONS[0]))),
              ("undo_off", draw_terug(cfg, undo_off=True).crop(box_xy(BACK_BUTTONS[0]))),
              ("bar", draw_menu(cfg, full=True).crop(box_xy(GOAL_BAR)))]
    items.sort(key=lambda kv: (-kv[1].size[1], -kv[1].size[0]))
    sheets, sprites = [], {}
    x = y = shelf = 0
    for key, img in items:
        w, h = img.size
        if x + w > W:
            x, y, shelf = 0, y + shelf, 0
        if not sheets or y + h > H:
            sheets.append(Image.new("RGB", (W, H), BG))
            x = y = shelf = 0
        sheets[-1].paste(img, (x, y))
        sprites[key] = Sprite(len(SCREEN_PICTURES) + len(sheets) - 1, x, y, w, h)
        x, shelf = x + w, max(shelf, h)
    return [to565(sheet) for sheet in sheets], sprites


def box_xy(box):
    x, y, w, h = box
    return (x, y, x + w, y + h)


# ---------------------------------------------------------------- the code
# Nextion's rules: no spaces in a line (except after the command), no
# brackets in sums, and a sum is worked out from left to right.

def indent(lines):
    return ["  " + line for line in lines]


def IF(*branches, otherwise=None):
    """IF((cond, lines), (cond, lines), ..., otherwise=lines)"""
    out = []
    for k, (cond, lines) in enumerate(branches):
        out += [("if(" if k == 0 else "}else if(") + cond + ")", "{"] + indent(lines)
    if otherwise is not None:
        out += ["}else", "{"] + indent(otherwise)
    return out + ["}"]


def WHILE(cond, lines):
    return [f"while({cond})", "{"] + indent(lines) + ["}"]


class Code:
    """Writes the screen's code for one tasks.json."""

    def __init__(self, cfg, sprites):
        self.cfg, self.sp = cfg, sprites
        self.n = len(cfg["tasks"])

    # ---- the parts on the page
    def var_id(self, block, i=0):
        return 2 + TASK_VARS.index(block) * self.n + i

    def components(self):
        """(kind, name, attributes, events), in id order: the page, tm0, the variables."""
        parts = [("timer", "tm0", {"tim": 50, "en": 1}, {"timer": self.timer()})]
        for block in TASK_VARS:
            for i, task in enumerate(self.cfg["tasks"]):
                val = {"ve": task["every"], "vq": task["points"]}.get(block, 0)
                parts.append(("variable", f"{block}{i}", {"sta": 0, "val": val, "txt": "",
                                                           "txt_maxl": 1, "vscope": 0}, {}))
        return parts

    def mark(self):
        return 4270 + int(self.cfg.get("fresh_start", 1))

    # ---- drawing numbers and words, right to left from gx
    def word(self, key, y="gy2"):
        s = self.sp[key]
        return [f"gx=gx-{s.w}", s.xpic("gx", y)]

    def style(self, style):
        """Which strip of digits the next number comes from."""
        s = self.sp[("digits", style)]
        return [f"gdw={s.w // 10}", f"gdh={s.h}", f"gsx0={s.x}", f"gsy={s.y}", f"gsp={s.pic}"]

    # Draws gnum (at least gmin digits) right-aligned at gx in row gy2;
    # afterwards gx is its left edge.
    DIGITS = ["gc=0", "gmore=1"] + WHILE("gmore==1", [
        "gtmp=gnum/10", "gtmp=gtmp*10", "gdg=gnum-gtmp", "gnum=gnum/10",
        "gx=gx-gdw", "gsx=gdg*gdw", "gsx=gsx+gsx0", "xpic gx,gy2,gdw,gdh,gsx,gsy,gsp",
        "gc=gc+1", "gmore=0"] + IF(("gnum>0", ["gmore=1"])) + IF(("gc<gmin", ["gmore=1"])))

    def digits(self, style, y="gy2", load=(), x=None, minimum=1):
        out = list(load)
        if x is not None:
            out.append(f"gx={x}")
        if y != "gy2":
            out.append(f"gy2={y}")
        out += self.style(style)
        if minimum is not None:
            out.append(f"gmin={minimum}")
        return out + self.DIGITS

    @staticmethod
    def count(var):
        """gc: how many digits var has."""
        return ["gc=1", f"gtmp={var}/10"] + WHILE("gtmp>0", ["gc=gc+1", "gtmp=gtmp/10"])

    def centered(self, center, var_parts):
        """gx so that a line of the given widths (numbers: their var) is centred."""
        out = ["gw=0"]
        for part in var_parts:
            if isinstance(part, int):
                out.append(f"gw=gw+{part}")
            else:
                var, style = part
                out += self.count(var) + [f"gtmp=gc*{digit_width(style)}", "gw=gw+gtmp"]
        return out + ["gx=gw/2", f"gx=gx+{center}"]

    # ---- the clock
    @staticmethod
    def day_numbers():
        """Today as a day number (gdag), this week's Monday (gweek) and the
        1st of this month (gmaand)."""
        return ["// today as a day number, this week's Monday, the 1st of the month",
                "gy=rtc0-2000", "gm=rtc1"] + IF(("gm<3", ["gy=gy-1", "gm=gm+12"])) + [
            "gdag=gy*365", "gtmp=gy/4", "gdag=gdag+gtmp", "gtmp=gm*153", "gtmp=gtmp-457",
            "gtmp=gtmp/5", "gdag=gdag+gtmp", "gdag=gdag+rtc2", "gmaand=gdag-rtc2",
            "gmaand=gmaand+1"] + IF(("rtc6==0", ["gweek=gdag-6"]),
                                    otherwise=["gweek=gdag-rtc6", "gweek=gweek+1"])

    @staticmethod
    def clock_defaults():
        return ["gkj=2026", "gkm=1", "gkd=1", "gku=12", "gkmi=0"]

    @staticmethod
    def month_days():
        """gmax: the days in month gkm of year gkj; the day stays within it."""
        return ["gmax=31"] + IF(
            ("gkm==4||gkm==6||gkm==9||gkm==11", ["gmax=30"]),
            ("gkm==2", ["gmax=28", "gtmp=gkj/4", "gtmp=gtmp*4"] + IF(("gtmp==gkj", ["gmax=29"])))
        ) + IF(("gkd>gmax", ["gkd=gmax"]))

    # ---- Program.s and the page's start
    def program_s(self):
        return [
            "// Onze huistaken: the numbers the code uses (made by make_screens.py)",
            "int sys0=0,sys1=0,sys2=0",
            "int gscherm=0,grender=1,gk=0,gtik=19,gtoon=0,gnul=0",
            "int gdag=0,gweek=0,gmaand=0,gy=0,gm=0,gtmp=0,gid=0",
            "int gi=0,gn=0,gtaak=0,gb=0,gbon=0,gl=0,ge=0,gp=0,gs=0,gq=0,gd=0,gst=0",
            "int gwie=0,gopts=0,gdoel=0,gprev=0,gpp=0,gsave=0",
            "int gx=0,gy2=0,gtx=0,gty=0,gnum=0,gdg=0,gsx=0,gc=0,gmin=1,gw=0",
            "int gdw=0,gdh=0,gsx0=0,gsy=0,gsp=0,gmore=0",
            "int gzeker=0,gkd=1,gkm=1,gkj=2026,gku=12,gkmi=0,gmax=31,gf=0,gdelta=0",
            "page 0",
        ]

    def postinit(self):
        clear = [f"wepo gnul,{a}" for a in list(A_WEEK) + list(A_MONTH)
                 + [A_WEEK_KEY, A_MONTH_KEY, A_GOAL, A_GOAL_NR]]
        clear += [f"wepo gnul,{A_TASKS + 4 * i}" for i in range(MAX_TASKS)]
        load = [f"repo vl{i}.val,{A_TASKS + 4 * i}" for i in range(self.n)]
        return (["// the screen's memory: set up the first time (or after a fresh start)",
                 f"repo gtmp,{A_MARK}"]
                + IF((f"gtmp!={self.mark()}", clear + [f"gtmp={self.mark()}",
                                                       f"wepo gtmp,{A_MARK}"]))
                + ["// the day every task was last done"] + load
                + ["// the timer looks at the clock and draws, right away", "gtik=19",
                   "grender=1"])

    # ---- the timer: the clock once a second, and drawing what changed
    def timer(self):
        new_day = ["gtoon=gdag", "// a new week or month starts at 0",
                   f"repo gtmp,{A_WEEK_KEY}"] + IF(
            ("gtmp!=gweek", [f"wepo gnul,{A_WEEK[0]}", f"wepo gnul,{A_WEEK[1]}",
                             f"wepo gweek,{A_WEEK_KEY}"])) + [f"repo gtmp,{A_MONTH_KEY}"] + IF(
            ("gtmp!=gmaand", [f"wepo gnul,{A_MONTH[0]}", f"wepo gnul,{A_MONTH[1]}",
                              f"wepo gmaand,{A_MONTH_KEY}"])) + IF(
            (f"gscherm=={TAKEN}", ["grender=1"]))
        clock = IF(("rtc0<2025", IF((f"gscherm!={KLOK}",
                                     ["// the clock isn't set: set it first"]
                                     + self.clock_defaults() + [f"gscherm={KLOK}",
                                                                "grender=1"]))),
                   otherwise=self.day_numbers() + IF(("gdag!=gtoon", new_day)))
        return (["gtik=gtik+1"] + IF(("gtik>=20", ["gtik=0"] + clock))
                + IF(("grender>0", ["gk=grender", "grender=0"] + self.draw())))

    def draw(self):
        return (self.draw_tasks() + IF(
            (f"gscherm=={TAKEN}", self.draw_scores()),
            (f"gscherm=={WIE}", self.draw_wie()),
            (f"gscherm=={TERUG}", self.draw_terug()),
            (f"gscherm=={MENU}", self.draw_menu()),
            (f"gscherm=={KLOK}", self.draw_klok())))

    def task_state(self):
        """For task gi: gs = its picture (0 open, 1 done, 2 a bit late, 3 very
        late), gq = the points it gives now, gst = what it says, gd = days."""
        late = ["gd=0-gd", "gst=4", "// very late: a whole round, at least 3 days",
                "gtmp=ge"] + IF(("gtmp<3", ["gtmp=3"])) + IF(
            ("gd>=gtmp", ["gs=3", "gq=gp*2"]),
            otherwise=["// a bit late: half a round, at least 2 days", "gtmp=ge+1",
                       "gtmp=gtmp/2"] + IF(("gtmp<2", ["gtmp=2"])) + IF(
                ("gd>=gtmp", ["gs=2", "gtmp=gp+1", "gtmp=gtmp/2", "gq=gp+gtmp"])))
        return [f"gid=gi+{self.var_id('vl')}", "gl=b[gid].val",
                f"gid=gi+{self.var_id('ve')}", "ge=b[gid].val",
                f"gid=gi+{self.var_id('vq')}", "gp=b[gid].val",
                "gq=gp", "gs=0", "gst=0", "gd=0"] + IF(
            ("gl>0", ["gd=gl+ge", "gd=gd-gdag", "// days until it comes back"] + IF(
                ("gd>0", ["gs=1", "gst=3"] + IF(("gd==1", ["gst=2"]))),
                ("gd==0", ["gst=1"]),
                otherwise=late)))

    def draw_tasks(self):
        n = self.n
        status = IF(
            ("gst==0", self.word(("soon", "nooit gedaan"))),
            ("gst==1", self.word(("soon", "vandaag"))),
            ("gst==2", self.word(("done", "morgen"))),
            otherwise=IF(("gd>999", ["gd=999"])) + IF(
                ("gst==3", self.word(("done", " d")) + self.style("done")),
                ("gs==3", self.word(("late3", " d te laat")) + self.style("late3")),
                otherwise=self.word(("late", " d te laat")) + self.style("late"))
            + ["gnum=gd", "gmin=1"] + self.DIGITS + IF(("gst==3", self.word(("done", "over ")))))
        tile = IF(("gs>0", [f"picq gtx,gty,{TILE_W},{TILE_H},gs"])) + [
            f"gx=gtx+{STATUS_RIGHT}", f"gy2=gty+{STATUS_DY}"] + status
        where = ["// where its tile is", "gtmp=gi/3", f"gty=gtmp*{TILE_H + GAP}",
                 f"gty=gty+{GRID_Y}", "gtmp=gtmp*3", "gtx=gi-gtmp", f"gtx=gtx*{TILE_W + GAP}",
                 f"gtx=gtx+{GRID_X}"]
        tapped = IF(("gs==1", [f"gscherm={TERUG}"]), otherwise=[f"gscherm={WIE}"])
        return ["// the tasks: all of them (the list) or the one that was tapped"] + IF(
            (f"gscherm<={TERUG}||gscherm=={TAPPED}",
             IF((f"gscherm=={TAKEN}", [f"pic 0,0,{P_OPEN}", "gi=0", f"gn={n}"]),
                otherwise=["gi=gtaak", "gn=gtaak+1"])
             + WHILE("gi<gn", self.task_state() + where
                     + IF((f"gscherm=={TAKEN}", tile)) + ["gi=gi+1"])
             + IF((f"gscherm=={TAPPED}", ["// a done task: undo or again"] + tapped))))

    def draw_scores(self):
        (xa, y, w, _), (xb, *_) = CARDS
        week = IF(("gpp==0", [f"repo gnum,{A_WEEK[0]}", f"gx={xa + w - 10}"] + self.style("week0")),
                  otherwise=[f"repo gnum,{A_WEEK[1]}", f"gx={xb + w - 10}"] + self.style("week1"))
        month = IF(("gpp==0", [f"repo gnum,{A_MONTH[0]}", f"gx={xa + 10 + text_width('month', 'maand ')}"]),
                   otherwise=[f"repo gnum,{A_MONTH[1]}", f"gx={xb + 10 + text_width('month', 'maand ')}"])
        return (["// the points of both people: this week (big), this month", "gpp=0"]
                + WHILE("gpp<2", week + [f"gy2={y + WEEK_DY}", "gmin=1"] + self.DIGITS
                        + ["// this month, after the word maand"] + month + self.count("gnum")
                        + [f"gtmp=gc*{digit_width('month')}", "gx=gx+gtmp", f"gy2={y + MONTH_DY}"]
                        + self.style("month") + self.DIGITS + ["gpp=gpp+1"]))

    def draw_wie(self):
        x, y = TILE_SHOW
        info = lambda t: text_width("info", t)
        line = self.centered(W // 2, [info("+") + info(" punten"), ("gq", "info")])
        bonus = [f"gw=gw+{info(', met ') + info(' bonus')}"] + self.count("gbon") + [
            f"gtmp=gc*{digit_width('info')}", "gw=gw+gtmp", "gx=gw/2", f"gx=gx+{W // 2}"]
        return ([f"pic 0,0,{P_WIE}", f"xpic {x},{y},{TILE_W},{TILE_H},gtx,gty,gs",
                 "// +6 punten (, met 3 bonus): in the middle", "gbon=gq-gp"]
                + line + IF(("gbon>0", bonus))
                + IF(("gbon>0", self.word(("info", " bonus"), INFO_Y)
                      + self.digits("info", INFO_Y, ["gnum=gbon"])
                      + self.word(("info", ", met "), INFO_Y)))
                + self.word(("info", " punten"), INFO_Y)
                + self.digits("info", INFO_Y, ["gnum=gq"]) + self.word(("info", "+"), INFO_Y))

    def draw_terug(self):
        x, y = TILE_SHOW
        a, b = self.cfg["people"]
        prefixes = [("info", f"Gedaan door {a}, +"), ("info", f"Gedaan door {b}, +"),
                    ("info", "Samen gedaan, allebei +")]
        again = self.sp[("infom", "Nog een keer gedaan? Dat mag altijd.")]
        off = self.sp["undo_off"]
        widths = IF(("gwie==1", [f"gw={text_width(*prefixes[0])}"]),
                    ("gwie==2", [f"gw={text_width(*prefixes[1])}"]),
                    otherwise=[f"gw={text_width(*prefixes[2])}"])
        said = (widths + [f"gw=gw+{text_width('info', ' punten')}"] + self.count("gopts")
                + [f"gtmp=gc*{digit_width('info')}", "gw=gw+gtmp", "gx=gw/2", f"gx=gx+{W // 2}"]
                + self.word(("info", " punten"), INFO_Y)
                + self.digits("info", INFO_Y, ["gnum=gopts"])
                + IF(("gwie==1", self.word(prefixes[0], INFO_Y)),
                     ("gwie==2", self.word(prefixes[1], INFO_Y)),
                     otherwise=self.word(prefixes[2], INFO_Y)))
        return ([f"pic 0,0,{P_TERUG}", f"xpic {x},{y},{TILE_W},{TILE_H},gtx,gty,{P_DONE}",
                 "// who did it last (gone after a power cut: then no undo)",
                 f"gid=gtaak+{self.var_id('vw')}", "gwie=b[gid].val",
                 f"gid=gtaak+{self.var_id('vp')}", "gopts=b[gid].val"]
                + IF(("gwie==0", [off.xpic(BACK_BUTTONS[0][0], BACK_BUTTONS[0][1]),
                                  again.xpic((W - again.w) // 2, INFO_Y)]),
                     otherwise=said))

    def draw_menu(self):
        target = int(self.cfg["goal"]["points"])
        bx, by, bw, bh = GOAL_BAR
        bar = self.sp["bar"]
        date = ["// the date and time: 6-10-2026  9:05", f"gx={DATE_RIGHT}", "gf=0"] + WHILE(
            "gf<5", ["gmin=1"] + IF(("gf==0", ["gnum=rtc4", "gmin=2"]), ("gf==1", ["gnum=rtc3"]),
                                    ("gf==2", ["gnum=rtc0"]), ("gf==3", ["gnum=rtc1"]),
                                    otherwise=["gnum=rtc2"])
            + self.digits("date", DATE_Y, minimum=None)
            + IF(("gf==0", self.word(("date", ":"), DATE_Y)), ("gf==1", ["gx=gx-12"]),
                 ("gf<4", self.word(("date", "-"), DATE_Y)))
            + ["gf=gf+1"])
        suffix = text_width("goal", goal_suffix(self.cfg))
        rest = text_width("foot", "Nog ") + text_width("foot", " punten te gaan")
        done = self.sp[("footok", "Gehaald! Tijd voor de beloning.")]
        goal = (["// the goal: the points so far, the bar, how many to go", f"repo gd,{A_GOAL}"]
                + self.digits("goal", GOAL_NUM_Y, ["gnum=gd"], GOAL_RIGHT - suffix)
                + IF(("gd>0", [f"gtmp=gd*{bw}", f"gtmp=gtmp/{target}"]
                      + IF((f"gtmp>{bw}", [f"gtmp={bw}"]))
                      + IF(("gtmp>0", [bar.xpic(bx, by, "gtmp")]))))
                + IF((f"gd>={target}", [done.xpic(bx, GOAL_FOOT_Y)]),
                     otherwise=[f"gnum={target}-gd", "gtmp=gnum"] + self.count("gtmp")
                     + [f"gtmp=gc*{digit_width('foot')}", f"gx=gtmp+{bx + rest}"]
                     + self.word(("foot", " punten te gaan"), GOAL_FOOT_Y)
                     + self.digits("foot", GOAL_FOOT_Y)
                     + self.word(("foot", "Nog "), GOAL_FOOT_Y)))
        sure = self.sp["zeker"]
        return (IF(("gk==1", [f"pic 0,0,{P_MENU}"] + date + goal))
                + IF(("gzeker==1", [sure.xpic(MENU_BUTTONS[0][0], MENU_BUTTONS[0][1])])))

    def draw_klok(self):
        x, w = CLOCK_VALUE
        dh = STYLES["klok"][4]
        values = IF(*[(f"gf=={k}", [f"gnum={var}"]) for k, (_, var) in enumerate(CLOCK_FIELDS[:4])],
                    otherwise=["gnum=gkmi", "gmin=2"])
        return IF(("gk==1", [f"pic 0,0,{P_KLOK}"])) + ["// the five values", "gf=0"] + WHILE(
            "gf<5", ["gmin=1"] + values + [
                f"gy2=gf*{CLOCK_PITCH}", f"gy2=gy2+{CLOCK_ROWS[0]}",
                f"picq {x},gy2,{w},{CLOCK_ROW_H},{P_KLOK}",
                f"gy2=gy2+{(CLOCK_ROW_H - dh) // 2}"] + self.count("gnum") + IF(
                ("gc<gmin", ["gc=gmin"])) + [
                f"gtmp=gc*{digit_width('klok')}", "gtmp=gtmp/2", f"gx=gtmp+{x + w // 2}"]
            + self.digits("klok", minimum=None) + ["gf=gf+1"])

    # ---- a tap
    def press(self):
        n = self.n
        col, row = TILE_W + GAP, TILE_H + GAP
        taken = IF((f"tch1<{GRID_Y - 4}", IF((f"tch0>={MENU_BOX[0]}",
                                               [f"gscherm={MENU}", "gzeker=0", "grender=1"]))),
                   otherwise=["// which tile", f"gtmp=tch0-{GRID_X}"]
                   + IF(("gtmp<0", ["gtmp=0"])) + [f"gc=gtmp/{col}"]
                   + IF(("gc>2", ["gc=2"])) + [f"gtmp=tch1-{GRID_Y}"]
                   + IF(("gtmp<0", ["gtmp=0"])) + [f"gtmp=gtmp/{row}"]
                   + IF(("gtmp>3", ["gtmp=3"])) + ["gtmp=gtmp*3", "gtmp=gtmp+gc"]
                   + IF((f"gtmp<{n}", ["gtaak=gtmp", f"gscherm={TAPPED}", "grender=1"])))
        wy0, wy1 = WHO_BUTTONS[0][1], WHO_BUTTONS[0][1] + WHO_BUTTONS[0][3]
        wie = ["gb=0"] + IF(
            (f"tch1>={wy0}&&tch1<{wy1}", ["gb=3"] + IF(
                (f"tch0<{WHO_BUTTONS[1][0] - 4}", ["gb=1"]),
                (f"tch0<{WHO_BUTTONS[2][0] - 4}", ["gb=2"]))),
            (f"tch1>={WHO_CANCEL[1]}", [f"gscherm={TAKEN}", "grender=1"])) + IF(
            ("gb>0", self.save_tick() + [f"gscherm={TAKEN}", "grender=1"]))
        b0, b1, b2 = BACK_BUTTONS
        terug = IF(
            (f"tch1>={b0[1]}&&tch1<{b0[1] + b0[3]}",
             IF(("gwie>0", self.undo() + [f"gscherm={TAKEN}", "grender=1"]))),
            (f"tch1>={b1[1]}&&tch1<{b1[1] + b1[3]}", [f"gscherm={WIE}", "grender=1"]),
            (f"tch1>={b2[1]}", [f"gscherm={TAKEN}", "grender=1"]))
        m0, m1, m2 = MENU_BUTTONS
        menu = IF(
            (f"tch1>={m0[1]}&&tch1<{m0[1] + m0[3]}", IF(
                ("gzeker==0", ["gzeker=1", "grender=2"]),
                otherwise=["// a new goal: count from 0, with the next number",
                           f"wepo gnul,{A_GOAL}", f"repo gtmp,{A_GOAL_NR}", "gtmp=gtmp+1",
                           f"wepo gtmp,{A_GOAL_NR}", "gzeker=0", "grender=1"])),
            (f"tch1>={m1[1]}&&tch1<{m1[1] + m1[3]}", IF(
                ("rtc0<2025", self.clock_defaults()),
                otherwise=["gkj=rtc0", "gkm=rtc1", "gkd=rtc2", "gku=rtc3", "gkmi=rtc4"])
             + [f"gscherm={KLOK}", "grender=1"]),
            (f"tch1>={m2[1]}", [f"gscherm={TAKEN}", "grender=1"]))
        return IF((f"gscherm=={TAKEN}", taken), (f"gscherm=={WIE}", wie),
                  (f"gscherm=={TERUG}", terug), (f"gscherm=={MENU}", menu),
                  (f"gscherm=={KLOK}", self.klok_press())) + IF(
            ("gsave==1", ["gsave=0"] + self.remember_day()))

    def remember_day(self):
        """The day of task gtaak into the memory."""
        return ["// the day it was done, also in the memory"] + IF(
            *[(f"gtaak=={i}", [f"wepo vl{i}.val,{A_TASKS + 4 * i}"]) for i in range(self.n)])

    def save_tick(self):
        add = []
        for k, skip in ((0, 2), (1, 1)):
            lines = []
            for a in (A_WEEK[k], A_MONTH[k]):
                lines += [f"repo gtmp,{a}", "gtmp=gtmp+gq", f"wepo gtmp,{a}"]
            add += IF((f"gb!={skip}", lines))
        return ([f"gid=gtaak+{self.var_id('vl')}", "gprev=b[gid].val", "b[gid].val=gdag",
                 f"gid=gtaak+{self.var_id('vw')}", "b[gid].val=gb",
                 f"gid=gtaak+{self.var_id('vp')}", "b[gid].val=gq",
                 f"gid=gtaak+{self.var_id('vv')}", "b[gid].val=gprev",
                 f"gid=gtaak+{self.var_id('vd')}", f"repo gtmp,{A_GOAL_NR}", "b[gid].val=gtmp"]
                + ["gsave=1", "// the points (together: both, as in the app)"]
                + add + [f"repo gtmp,{A_GOAL}", "gtmp=gtmp+gq"]
                + IF(("gb==3", ["gtmp=gtmp+gq"])) + [f"wepo gtmp,{A_GOAL}"])

    @staticmethod
    def take_off(a, double=False):
        out = [f"repo gtmp,{a}", "gtmp=gtmp-gopts"]
        if double:
            out += IF(("gwie==3", ["gtmp=gtmp-gopts"]))
        return out + IF(("gtmp<0", ["gtmp=0"])) + [f"wepo gtmp,{a}"]

    def undo(self):
        """The day before comes back; the points come off where they still count."""
        off = []
        for keys, var in ((A_WEEK, "gweek"), (A_MONTH, "gmaand")):
            off += IF((f"gl>={var}", IF(("gwie!=2", self.take_off(keys[0])))
                       + IF(("gwie!=1", self.take_off(keys[1])))))
        return ([f"gid=gtaak+{self.var_id('vv')}", "gprev=b[gid].val",
                 f"gid=gtaak+{self.var_id('vl')}", "gl=b[gid].val", "b[gid].val=gprev",
                 f"gid=gtaak+{self.var_id('vw')}", "gwie=b[gid].val", "b[gid].val=0",
                 f"gid=gtaak+{self.var_id('vp')}", "gopts=b[gid].val",
                 f"gid=gtaak+{self.var_id('vd')}", "gdoel=b[gid].val"]
                + ["gsave=1"] + off + [f"repo gtmp,{A_GOAL_NR}"]
                + IF(("gdoel==gtmp", self.take_off(A_GOAL, double=True))))

    def klok_press(self):
        cancel, save = CLOCK_BUTTONS
        day = IF(("gf==0", ["gkd=gkd+gdelta"] + IF(("gkd<1", ["gkd=gmax"]))
                  + IF(("gkd>gmax", ["gkd=1"]))))
        change = IF(
            ("gf==1", ["gkm=gkm+gdelta"] + IF(("gkm<1", ["gkm=12"])) + IF(("gkm>12", ["gkm=1"]))),
            ("gf==2", ["gkj=gkj+gdelta"] + IF(("gkj<2025", ["gkj=2025"]))
             + IF(("gkj>2099", ["gkj=2099"]))),
            ("gf==3", ["gku=gku+gdelta"] + IF(("gku<0", ["gku=23"])) + IF(("gku>23", ["gku=0"]))),
            ("gf==4", ["gkmi=gkmi+gdelta"] + IF(("gkmi<0", ["gkmi=59"]))
             + IF(("gkmi>59", ["gkmi=0"]))))
        mx, mw = CLOCK_MINUS
        px, pw = CLOCK_PLUS
        last = CLOCK_ROWS[-1] + CLOCK_PITCH
        return IF(
            (f"tch1>={save[1]}", IF(
                (f"tch0>={save[0]}", ["// save: the day goes to 1 first, so every date in between exists",
                                      "rtc2=1", "rtc0=gkj", "rtc1=gkm", "rtc2=gkd", "rtc3=gku",
                                      "rtc4=gkmi", "rtc5=0", "gtoon=0", "gtik=19",
                                      f"gscherm={TAKEN}", "grender=1"]),
                (f"tch0<{cancel[0] + cancel[2]}", [f"gscherm={TAKEN}", "grender=1"]))),
            (f"tch1>={CLOCK_ROWS[0]}&&tch1<{last}", ["gdelta=0"] + IF(
                (f"tch0>={mx}&&tch0<{mx + mw}", ["gdelta=-1"]),
                (f"tch0>={px}&&tch0<{px + pw}", ["gdelta=1"])) + IF(
                ("gdelta!=0", [f"gf=tch1-{CLOCK_ROWS[0]}", f"gf=gf/{CLOCK_PITCH}"]
                 + change + self.month_days() + day + ["grender=2"]))))


# ---------------------------------------------------------------- the project
def load_config():
    with open(os.path.join(HERE, "tasks.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    tasks = cfg["tasks"]
    if not 1 <= len(tasks) <= MAX_TASKS:
        sys.exit(f"tasks.json: 1 to {MAX_TASKS} tasks, not {len(tasks)}")
    for t in tasks:
        for key, hi in (("points", 999), ("every", 365)):
            if not (isinstance(t.get(key), int) and 1 <= t[key] <= hi):
                sys.exit(f"tasks.json: {key} must be a whole number 1-{hi} ({t.get('name')})")
    if len(cfg.get("people", [])) != 2:
        sys.exit("tasks.json: people has two names")
    return cfg


def build(cfg):
    """Everything the project holds: pictures, Program.s, the page and its parts."""
    screens = [draw_taken(cfg, P_OPEN), draw_taken(cfg, P_DONE), draw_taken(cfg, P_HIGH),
               draw_taken(cfg, P_URGENT), draw_wie(cfg), draw_terug(cfg), draw_menu(cfg),
               draw_klok(cfg)]
    sheets, sprites = make_sprites(cfg)
    code = Code(cfg, sprites)
    page = {"load": [], "loadend": code.postinit(), "down": code.press()}
    return {"pictures": screens + sheets, "program": code.program_s(), "page": page,
            "parts": code.components(), "code": code, "sprites": sprites}


def hmi_bytes(project):
    objects = [nextion_hmi.component("page", 0, "taken", project["page"], x=0, y=0, w=W, h=H,
                                     sta=2, bco=0xFFFF, pic=P_OPEN)]
    for cid, (kind, name, attrs, events) in enumerate(project["parts"], start=1):
        objects.append(nextion_hmi.component(kind, cid, name, events, **attrs))
    return nextion_hmi.build(project["program"], "taken", objects, project["pictures"])


def code_text(project):
    bar = "=" * 72
    parts = project["parts"]
    out = ["ONZE HUISTAKEN - ALL THE CODE IN Huistaken.HMI",
           "Made by make_screens.py from tasks.json, to read. Huistaken.HMI already holds",
           "all of it: you don't need to type or paste anything.", "",
           bar, "PROGRAM.S", bar] + project["program"] + ["", bar,
           "PAGE taken (the only page): sta = image, pic = 0", bar,
           f"  tm0  Timer, tim = 50 ms, en = 1",
           f"  {len(parts) - 1} Variables (numbers), per task: " + ", ".join(
               f"{b}0-{b}{len(project['code'].cfg['tasks']) - 1}" for b in TASK_VARS), "",
           "---- Postinitialize Event (the page) ----"] + project["page"]["loadend"] + [
           "", "---- Touch Press Event (the page) ----"] + project["page"]["down"] + [
           "", "---- Timer Event (tm0) ----"] + parts[0][3]["timer"]
    return "\n".join(out) + "\n"


def main():
    cfg = load_config()
    project = build(cfg)
    os.makedirs(SCREENS, exist_ok=True)
    names = [f"{k:02d}_{n}.png" for k, n in enumerate(SCREEN_PICTURES)]
    names += [f"{len(names) + k:02d}_letters.png" for k in range(len(project["pictures"]) - 8)]
    for old in os.listdir(SCREENS):
        if old.endswith(".png") and old not in names:
            os.remove(os.path.join(SCREENS, old))
    for name, img in zip(names, project["pictures"]):
        img.save(os.path.join(SCREENS, name))
    with open(PROJECT, "wb") as f:
        f.write(hmi_bytes(project))
    with open(os.path.join(HERE, "nextion_code.txt"), "w", encoding="utf-8") as f:
        f.write(code_text(project))
    lines = sum(1 for k in ("loadend", "down") for l in project["page"][k] if not l.strip().startswith("//"))
    lines += sum(1 for l in project["parts"][0][3]["timer"] if not l.strip().startswith("//"))
    print(f"{len(cfg['tasks'])} tasks: Huistaken.HMI ({len(project['pictures'])} pictures, "
          f"{lines} lines of code), screens\\, nextion_code.txt")
    print("Check it and make preview.png: python house_tasks/check.py")


if __name__ == "__main__":
    main()
