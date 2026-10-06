"""Onze huistaken on the Nextion screen (NX4832K035, 480 x 320).

The house tasks app on the screen by itself: two people with a score per
week and month, "Samen gedaan", tasks that come back after their number of
days (with bonus points when late), and a savings goal. Like the app, but
the screen keeps its own list.

Draws the pictures from tasks.json and writes nextion_code.txt: the code
for Program.s and every page, ready to paste into Nextion Editor.
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

# ---------------------------------------------------------------- layout
# Main page: the two people at the top, then 3 x 4 task tiles
CARDS = [(8, 6, 196, 50), (210, 6, 196, 50)]     # x, y, w, h
MENU_BOX = (412, 6, 60, 50)
COLS, ROWS = 3, 4
TILE_W, TILE_H, GAP = 150, 58, 6
GRID_X = (W - COLS * TILE_W - (COLS - 1) * GAP) // 2    # 9
GRID_Y = 62
MAX_TASKS = COLS * ROWS

# The other pages
TILE_SHOW = (165, 50)                            # where a tile is shown again
INFO_BOX = (16, 114, 448, 20)                    # the line under that tile
WHO_BUTTONS = [(16, 144, 144, 84), (168, 144, 144, 84), (320, 144, 144, 84)]
WHO_CANCEL = (16, 240, 448, 64)
BACK_BUTTONS = [(16, 142, 448, 50), (16, 198, 448, 50), (16, 254, 448, 50)]
DATE_BOX = (180, 18, 284, 20)
GOAL_CARD = (16, 50, 448, 100)
GOAL_NUMS = (232, 78, 216, 20)
GOAL_BAR = (32, 106, 416, 14)
GOAL_FOOT = (32, 126, 416, 18)
MENU_BUTTONS = [(16, 160, 448, 44), (16, 212, 448, 44), (16, 264, 448, 44)]
CLOCK_ROWS = [50, 92, 134, 176, 218]
CLOCK_ROW_H = 38
CLOCK_MINUS, CLOCK_VALUE, CLOCK_PLUS = (200, 60), (266, 120), (392, 60)  # x, w
CLOCK_BUTTONS = [(16, 266, 216, 46), (248, 266, 216, 46)]   # Annuleren, Opslaan

# The fonts made in Nextion Editor
SMALL, BIG = 0, 1
SMALL_H, BIG_H = 16, 32

# ---------------------------------------------------------------- memory
# The screen's own memory (EEPROM, 1 KB, kept without power).
# Every number takes 4 bytes.
A_MARK = 0            # holds the marker once the memory is set up
A_WEEK = (4, 8)       # points this week: person 1, person 2
A_MONTH = (12, 16)    # points this month
A_WEEK_KEY = 20       # the day number of this week's Monday
A_MONTH_KEY = 24      # the day number of the 1st of this month
A_GOAL = 28           # points saved for the goal
A_GOAL_NR = 32        # the goal's number: +1 for every new goal
A_TASKS = 64          # task i: 64 + 20*i


def a_last(i):        # the day it was last done (0 = never)
    return A_TASKS + 20 * i


def a_who(i):         # who did it last: 1, 2, 3 = together (0 = can't undo)
    return A_TASKS + 20 * i + 4


def a_pts(i):         # the points it gave
    return A_TASKS + 20 * i + 8


def a_prev(i):        # the day before that (for undo)
    return A_TASKS + 20 * i + 12


def a_goal_nr(i):     # the goal it counted for (for undo)
    return A_TASKS + 20 * i + 16


# ---------------------------------------------------------------- pictures
PICTURES = [
    ("00_taken.png", "taken: every task open"),
    ("01_taken_gedaan.png", "taken: every task done"),
    ("02_taken_hoog.png", "taken: every task a bit late (x1.5)"),
    ("03_taken_urgent.png", "taken: every task very late (x2)"),
    ("04_wie.png", "wie"),
    ("05_wie_in.png", "wie, buttons pressed"),
    ("06_terug.png", "terug"),
    ("07_terug_in.png", "terug, buttons pressed"),
    ("08_terug_uit.png", "terug, undo greyed out"),
    ("09_menu.png", "menu"),
    ("10_menu_in.png", "menu, buttons pressed"),
    ("11_menu_zeker.png", "menu, 'Zeker?' button"),
    ("12_menu_vol.png", "menu, full goal bar"),
    ("13_klok.png", "klok"),
    ("14_klok_in.png", "klok, buttons pressed"),
]
(P_OPEN, P_DONE, P_HIGH, P_URGENT, P_WHO, P_WHO_IN, P_BACK, P_BACK_IN,
 P_BACK_OFF, P_MENU, P_MENU_IN, P_MENU_SURE, P_MENU_FULL, P_CLOCK,
 P_CLOCK_IN) = range(len(PICTURES))

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
PRESSED = (226, 231, 223)
LATE_TINT = (247, 237, 234)     # LATE at 7% on white


def mix(c1, c2, f):
    return tuple(round(a + (b - a) * f) for a, b in zip(c1, c2))


FONT_FILES = {
    "display": "BricolageGrotesque-ExtraBold.ttf",
    "bold": "AtkinsonHyperlegible-Bold.ttf",
    "regular": "AtkinsonHyperlegible-Regular.ttf",
}


@lru_cache(maxsize=None)
def font(kind, size):
    return ImageFont.truetype(os.path.join(HERE, "fonts", FONT_FILES[kind]), size)


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
        return img


def tick(c, cx, cy, r, color):
    c.line([(cx - .45 * r, cy + .02 * r), (cx - .12 * r, cy + .34 * r),
            (cx + .46 * r, cy - .32 * r)], color, .24 * r)


def tile_box(i):
    col, row = i % COLS, i // COLS
    return (GRID_X + col * (TILE_W + GAP), GRID_Y + row * (TILE_H + GAP), TILE_W, TILE_H)


def status_box(i):
    """Where the screen writes "vandaag", "over 3 d", "2 d te laat"."""
    x, y, w, h = tile_box(i)
    return (x + 60, y + 34, w - 68, 18)


def bonus_points(task):
    """Points when a bit late (x1.5) and very late (x2), as in the app."""
    p = task["points"]
    return p + math.ceil(p / 2), 2 * p


def late_days(task):
    """Days late from which the bonus starts: a bit late, very late
    (the app: half the repeat and 2 days, the whole repeat and 3 days)."""
    every = task["every"]
    return max(math.ceil(every / 2), 2), max(every, 3)


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
    # The tick circle, top right
    cx, cy, r = x + w - 17, y + 17, 9
    if state == P_DONE:
        c.circle(cx, cy, r, fill=ACCENT)
        tick(c, cx, cy, r, SURFACE)
    else:
        ring = {P_HIGH: SOON, P_URGENT: LATE}.get(state, MUTED)
        c.circle(cx, cy, r, outline=ring, width=1.6)
    # The points, bottom left, like the app's pill
    color = {P_HIGH: SOON, P_URGENT: LATE, P_DONE: MUTED}.get(state, INK)
    fill = {P_OPEN: SUNK, P_DONE: mix(SUNK, MUTED, .12)}.get(state, mix(SURFACE, color, .14))
    nf, uf = font("display", 14), font("regular", 11)
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


def button(c, box, label, pressed, style="card", sub=None, dot=None):
    """style: card (white), primary (green), danger (red), off (greyed)."""
    x, y, w, h = box
    fill, ink, edge = {
        "card": (PRESSED if pressed else SURFACE, INK, LINE),
        "primary": (mix(ACCENT, INK, .35) if pressed else ACCENT, SURFACE, None),
        "danger": (mix(LATE, INK, .3) if pressed else LATE, SURFACE, None),
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


def draw_wie(cfg, pressed=False):
    c = Canvas()
    title(c, "Wie heeft dit gedaan?")
    for k, box in enumerate(WHO_BUTTONS[:2]):
        button(c, box, cfg["people"][k], pressed, sub="krijgt de punten", dot=PERSON[k])
    button(c, WHO_BUTTONS[2], "Samen", pressed, sub="allebei de punten")
    button(c, WHO_CANCEL, "Annuleren", pressed)
    return c.image()


def draw_terug(cfg, pressed=False, undo_off=False):
    c = Canvas()
    title(c, "Al gedaan")
    labels = ["Afvinken ongedaan maken", "Nog een keer gedaan", "Terug naar de taken"]
    for k, (box, label) in enumerate(zip(BACK_BUTTONS, labels)):
        style = "off" if k == 0 and undo_off else "primary" if k == 2 else "card"
        button(c, box, label, pressed, style)
    return c.image()


def draw_menu(cfg, pressed=False, sure=False, full=False):
    c = Canvas()
    title(c, "Menu")
    x, y, w, h = GOAL_CARD
    c.rect(x, y, w, h, SURFACE, radius=16, outline=LINE)
    c.text((x + 16, y + 16), "SPAARDOEL", font("bold", 11), MUTED, "lm")
    c.text((x + 16, y + 38), cfg["goal"]["name"], font("display", 20), INK, "lm")
    bx, by, bw, bh = GOAL_BAR
    c.rect(bx, by, bw, bh, SUNK, radius=bh / 2)
    if full:
        c.rect(bx, by, bw, bh, ACCENT, radius=bh / 2)
    if sure:
        button(c, MENU_BUTTONS[0], "Zeker? Tik nog een keer", False, "danger")
    else:
        button(c, MENU_BUTTONS[0], "Nieuw spaardoel beginnen", pressed)
    button(c, MENU_BUTTONS[1], "Klok gelijkzetten", pressed)
    button(c, MENU_BUTTONS[2], "Terug naar de taken", pressed, "primary")
    return c.image()


CLOCK_FIELDS = [("Dag", "gkd"), ("Maand", "gkm"), ("Jaar", "gkj"), ("Uur", "gku"),
                ("Minuut", "gkmi")]


def clock_box(row, part):
    x, w = {"minus": CLOCK_MINUS, "value": CLOCK_VALUE, "plus": CLOCK_PLUS}[part]
    return (x, CLOCK_ROWS[row], w, CLOCK_ROW_H)


def draw_klok(cfg, pressed=False):
    c = Canvas()
    title(c, "Klok gelijkzetten")
    for row, (label, _) in enumerate(CLOCK_FIELDS):
        y = CLOCK_ROWS[row]
        c.text((32, y + CLOCK_ROW_H / 2), label, font("bold", 17), INK, "lm")
        for part, sign in (("minus", "-"), ("plus", "+")):
            x, y0, w, h = clock_box(row, part)
            c.rect(x, y0, w, h, PRESSED if pressed else SURFACE, radius=12, outline=LINE)
            cx, cy = x + w / 2, y0 + h / 2
            c.rect(cx - 8, cy - 1.25, 16, 2.5, INK, radius=1.25)
            if sign == "+":
                c.rect(cx - 1.25, cy - 8, 2.5, 16, INK, radius=1.25)
        x, y0, w, h = clock_box(row, "value")
        c.rect(x, y0, w, h, SUNK, radius=12)
    button(c, CLOCK_BUTTONS[0], "Annuleren", pressed)
    button(c, CLOCK_BUTTONS[1], "Opslaan", pressed, "primary")
    return c.image()


def draw_pictures(cfg):
    return [
        draw_taken(cfg, P_OPEN), draw_taken(cfg, P_DONE),
        draw_taken(cfg, P_HIGH), draw_taken(cfg, P_URGENT),
        draw_wie(cfg), draw_wie(cfg, pressed=True),
        draw_terug(cfg), draw_terug(cfg, pressed=True), draw_terug(cfg, undo_off=True),
        draw_menu(cfg), draw_menu(cfg, pressed=True), draw_menu(cfg, sure=True),
        draw_menu(cfg, full=True),
        draw_klok(cfg), draw_klok(cfg, pressed=True),
    ]


# ---------------------------------------------------------------- the code
# Nextion's rules: no spaces in a line (except after the command), no
# brackets in sums, and a sum is worked out from left to right.

def rgb565(color):
    r, g, b = color
    return (r >> 3) << 11 | (g >> 2) << 5 | b >> 3


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


def inside(box, var=("tch0", "tch1"), grow=0):
    x, y, w, h = box
    return (f"{var[0]}>={x - grow}&&{var[0]}<{x + w + grow}&&"
            f"{var[1]}>={y - grow}&&{var[1]}<{y + h + grow}")


def xstr(box, fnt, color, pic, text, xcen=0):
    x, y, w, h = box
    return f"xstr {x},{y},{w},{h},{fnt},{rgb565(color)},{pic},{xcen},1,0,{text}"


def picq(box, pic):
    x, y, w, h = box
    return f"picq {x},{y},{w},{h},{pic}"


def day_numbers():
    """Today as a day number (gdag), with this week's Monday (gweek) and the
    1st of this month (gmaand)."""
    return [
        "// today as a day number: gdag; this week's Monday: gweek; the 1st: gmaand",
        "gy=rtc0-2000",
        "gm=rtc1",
    ] + IF(("gm<3", ["gy=gy-1", "gm=gm+12"])) + [
        "gdag=gy*365",
        "gtmp=gy/4",
        "gdag=gdag+gtmp",
        "gtmp=gm*153",
        "gtmp=gtmp-457",
        "gtmp=gtmp/5",
        "gdag=gdag+gtmp",
        "gdag=gdag+rtc2",
        "gmaand=gdag-rtc2",
        "gmaand=gmaand+1",
    ] + IF(("rtc6==0", ["gweek=gdag-6"]), otherwise=["gweek=gdag-rtc6", "gweek=gweek+1"])


def show_scores():
    out = []
    for k, (x, y, w, h) in enumerate(CARDS):
        out += [
            f"repo gtmp,{A_WEEK[k]}",
            "covx gtmp,va0.txt,0,0",
            xstr((x + w - 100, y + 16, 90, BIG_H), BIG, PERSON[k], P_OPEN, "va0.txt", 2),
            f"repo gtmp,{A_MONTH[k]}",
            "covx gtmp,va0.txt,0,0",
            'va1.txt="maand "+va0.txt',
            xstr((x + 10, y + 28, 90, 18), SMALL, MUTED, P_OPEN, "va1.txt"),
        ]
    return out


def show_task(i, task):
    t2, t3 = late_days(task)
    box, sbox, s = tile_box(i), status_box(i), f"gs{i}"
    late = ["gtmp=0-gtmp", "covx gtmp,va0.txt,0,0", 'va1.txt=va0.txt+" d te laat"'] + IF(
        (f"gtmp>={t3}", [f"{s}={P_URGENT}", picq(box, P_URGENT),
                         xstr(sbox, SMALL, LATE, P_URGENT, "va1.txt", 2)]),
        (f"gtmp>={t2}", [f"{s}={P_HIGH}", picq(box, P_HIGH),
                         xstr(sbox, SMALL, LATE, P_HIGH, "va1.txt", 2)]),
        otherwise=[f"{s}={P_OPEN}", xstr(sbox, SMALL, LATE, P_OPEN, "va1.txt", 2)])
    done = [f"{s}={P_DONE}", picq(box, P_DONE)] + IF(
        ("gtmp==1", [xstr(sbox, SMALL, MUTED, P_DONE, '"morgen"', 2)]),
        otherwise=["covx gtmp,va0.txt,0,0", 'va1.txt="over "+va0.txt+" d"',
                   xstr(sbox, SMALL, MUTED, P_DONE, "va1.txt", 2)])
    return [f"// {i + 1}. {task['name']}: every {task['every']} days, {task['points']} points",
            f"repo glast,{a_last(i)}"] + IF(
        ("glast<=0", [f"{s}={P_OPEN}", xstr(sbox, SMALL, SOON, P_OPEN, '"nooit gedaan"', 2)]),
        otherwise=[f"gtmp=glast+{task['every']}", "gtmp=gtmp-gdag"] + IF(
            ("gtmp>0", done),
            ("gtmp==0", [f"{s}={P_OPEN}", xstr(sbox, SMALL, SOON, P_OPEN, '"vandaag"', 2)]),
            otherwise=late))


def first_start(cfg):
    """Everything at 0, also for task places without a task (yet)."""
    out = ["// the first start (or a fresh start): everything at 0",
           "gtmp=0"]
    for a in list(A_WEEK) + list(A_MONTH) + [A_WEEK_KEY, A_MONTH_KEY, A_GOAL, A_GOAL_NR]:
        out.append(f"wepo gtmp,{a}")
    for i in range(MAX_TASKS):
        out += [f"wepo gtmp,{a}" for a in (a_last(i), a_who(i), a_pts(i), a_prev(i),
                                            a_goal_nr(i))]
    return out + [f"gtmp={mark(cfg)}", f"wepo gtmp,{A_MARK}"]


def mark(cfg):
    return 4270 + int(cfg.get("fresh_start", 1))


def taken_code(cfg):
    tasks = cfg["tasks"]
    post = day_numbers() + ["gtoon=gdag", f"repo gtmp,{A_MARK}"]
    post += IF((f"gtmp!={mark(cfg)}", first_start(cfg)))
    post += ["// a new week or month starts at 0", f"repo gtmp,{A_WEEK_KEY}"]
    post += IF(("gtmp!=gweek", ["gtmp=0", f"wepo gtmp,{A_WEEK[0]}", f"wepo gtmp,{A_WEEK[1]}",
                                f"wepo gweek,{A_WEEK_KEY}"]))
    post += [f"repo gtmp,{A_MONTH_KEY}"]
    post += IF(("gtmp!=gmaand", ["gtmp=0", f"wepo gtmp,{A_MONTH[0]}",
                                 f"wepo gtmp,{A_MONTH[1]}", f"wepo gmaand,{A_MONTH_KEY}"]))
    post += ["// the points"] + show_scores() + ["// the tasks"]
    for i, task in enumerate(tasks):
        post += show_task(i, task)
    post = ["// the clock isn't set yet: set it first"] + IF(
        ("rtc0<2025", ["page klok"]), otherwise=post)

    press = ["gknop=0"] + IF(
        (inside(MENU_BOX), ["gknop=99"]),
        *[(inside(tile_box(i), grow=GAP // 2), [f"gknop={i + 1}"]) for i in range(len(tasks))])

    branches = [("gknop==99", ["page menu"])]
    for i, task in enumerate(tasks):
        x, y, _, _ = tile_box(i)
        p2, p3 = bonus_points(task)
        p = task["points"]
        branches.append((f"gknop=={i + 1}", [
            f"gtaak={i}", f"gtx={x}", f"gty={y}", f"gpts={p}", "gbonus=0"] + IF(
            (f"gs{i}=={P_DONE}", [f"gstaat={P_DONE}", "page terug"]),
            otherwise=[f"gstaat=gs{i}"] + IF(
                (f"gs{i}=={P_URGENT}", [f"gpts={p3}", f"gbonus={p3 - p}"]),
                (f"gs{i}=={P_HIGH}", [f"gpts={p2}", f"gbonus={p2 - p}"])) + ["page wie"])))
    release = IF(*branches)

    timer = ["// once a minute: a new day draws the page again"] + day_numbers() + IF(
        ("gdag!=gtoon", ["page taken"]))
    return {"Preinitialize Event": ["gknop=0"], "Postinitialize Event": post,
            "Touch Press Event": press, "Touch Release Event": release,
            "tm0 Timer Event": timer}


def presses(buttons, pressed_pic, skip=None):
    """Touch Press: which button (gknop), drawn pressed."""
    branches = []
    for code, box in buttons:
        draw = [picq(box, pressed_pic)]
        if skip and code in skip:
            draw = IF((skip[code], draw))
        branches.append((inside(box), [f"gknop={code}"] + draw))
    return ["gknop=0"] + IF(*branches)


def per_task(cfg, make):
    return IF(*[(f"gtaak=={i}", make(i)) for i in range(len(cfg["tasks"]))])


def add_points():
    out = []
    for k, skip in ((0, 2), (1, 1)):
        lines = []
        for a in (A_WEEK[k], A_MONTH[k]):
            lines += [f"repo gtmp,{a}", "gtmp=gtmp+gpts", f"wepo gtmp,{a}"]
        out += IF((f"gwie!={skip}", lines))
    out += ["// the goal: together counts for both, as in the app",
            f"repo gtmp,{A_GOAL}", "gtmp=gtmp+gpts"]
    out += IF(("gwie==3", ["gtmp=gtmp+gpts"])) + [f"wepo gtmp,{A_GOAL}"]
    return out


def take_off(a, double=False):
    out = [f"repo gtmp,{a}", "gtmp=gtmp-gopts"]
    if double:
        out += IF(("gwie==3", ["gtmp=gtmp-gopts"]))
    return out + IF(("gtmp<0", ["gtmp=0"])) + [f"wepo gtmp,{a}"]


def remove_points():
    """Undo: the points come off, if they still count in this week, this
    month and for this goal."""
    out = []
    for keys, var in ((A_WEEK, "gweek"), (A_MONTH, "gmaand")):
        out += IF((f"glast>={var}",
                   IF(("gwie!=2", take_off(keys[0]))) + IF(("gwie!=1", take_off(keys[1])))))
    return out + [f"repo gtmp,{A_GOAL_NR}"] + IF(
        ("gdoel==gtmp", take_off(A_GOAL, double=True)))


def wie_code(cfg):
    x, y = TILE_SHOW
    post = [f"xpic {x},{y},{TILE_W},{TILE_H},gtx,gty,gstaat",
            "covx gpts,taken.va0.txt,0,0"] + IF(
        ("gbonus>0", ["covx gbonus,taken.va1.txt,0,0",
                      'taken.va2.txt="+"+taken.va0.txt+" punten, met "+taken.va1.txt+" bonus"']),
        otherwise=['taken.va2.txt="+"+taken.va0.txt+" punten"']) + [
        xstr(INFO_BOX, SMALL, INK, P_WHO, "taken.va2.txt", 1)]
    buttons = [(1, WHO_BUTTONS[0]), (2, WHO_BUTTONS[1]), (3, WHO_BUTTONS[2]), (9, WHO_CANCEL)]
    save = [f"repo gdoel,{A_GOAL_NR}"] + per_task(
        cfg, lambda i: [f"repo gprev,{a_last(i)}", f"wepo gprev,{a_prev(i)}",
                        f"wepo gdag,{a_last(i)}", f"wepo gwie,{a_who(i)}",
                        f"wepo gpts,{a_pts(i)}", f"wepo gdoel,{a_goal_nr(i)}"])
    release = IF(("gknop==9", ["page taken"]),
                 ("gknop>=1&&gknop<=3",
                  ["gwie=gknop", "// remember it with the task"] + save + add_points()
                  + ["page taken"]))
    return {"Preinitialize Event": ["gknop=0"], "Postinitialize Event": post,
            "Touch Press Event": presses(buttons, P_WHO_IN), "Touch Release Event": release}


def terug_code(cfg):
    a, b = cfg["people"]
    pre = ["gknop=0", "// who did it last, with how many points"] + per_task(
        cfg, lambda i: [f"repo gwie,{a_who(i)}", f"repo gopts,{a_pts(i)}",
                        f"repo glast,{a_last(i)}", f"repo gdoel,{a_goal_nr(i)}"])
    x, y = TILE_SHOW
    post = [f"xpic {x},{y},{TILE_W},{TILE_H},gtx,gty,{P_DONE}"] + IF(
        ("gwie==0", [picq(BACK_BUTTONS[0], P_BACK_OFF),
                     xstr(INFO_BOX, SMALL, MUTED, P_BACK, '"Nog een keer gedaan? Dat mag altijd."', 1)]),
        otherwise=["covx gopts,taken.va0.txt,0,0"] + IF(
            ("gwie==1", [f'taken.va2.txt="Gedaan door {a}, +"+taken.va0.txt+" punten"']),
            ("gwie==2", [f'taken.va2.txt="Gedaan door {b}, +"+taken.va0.txt+" punten"']),
            otherwise=['taken.va2.txt="Samen gedaan, allebei +"+taken.va0.txt+" punten"'])
        + [xstr(INFO_BOX, SMALL, INK, P_BACK, "taken.va2.txt", 1)])
    buttons = [(1, BACK_BUTTONS[0]), (2, BACK_BUTTONS[1]), (9, BACK_BUTTONS[2])]
    press = ["gknop=0"] + IF(
        (inside(BACK_BUTTONS[0]), IF(("gwie>0", ["gknop=1", picq(BACK_BUTTONS[0], P_BACK_IN)]))),
        *[(inside(box), [f"gknop={code}", picq(box, P_BACK_IN)]) for code, box in buttons[1:]])
    undo = per_task(cfg, lambda i: [f"repo gprev,{a_prev(i)}", f"wepo gprev,{a_last(i)}",
                                    "gtmp=0", f"wepo gtmp,{a_who(i)}"])
    release = IF(("gknop==9", ["page taken"]),
                 ("gknop==2", [f"gstaat={P_DONE}", "gbonus=0", "page wie"]),
                 ("gknop==1", ["// undo: the task gets its day before back"] + undo
                  + remove_points() + ["page taken"]))
    return {"Preinitialize Event": pre, "Postinitialize Event": post,
            "Touch Press Event": press, "Touch Release Event": release}


DAYS = ["zondag", "maandag", "dinsdag", "woensdag", "donderdag", "vrijdag", "zaterdag"]
MONTHS = ["januari", "februari", "maart", "april", "mei", "juni", "juli", "augustus",
          "september", "oktober", "november", "december"]


def menu_code(cfg):
    target = int(cfg["goal"]["points"])
    v0, v1, v2 = "taken.va0.txt", "taken.va1.txt", "taken.va2.txt"
    date = ["// the date and time"] + IF(
        *[(f"rtc6=={k}", [f'{v0}="{d}"']) for k, d in enumerate(DAYS)]) + [
        f"covx rtc2,{v1},0,0", f'{v2}={v0}+" "+{v1}'] + IF(
        *[(f"rtc1=={k + 1}", [f'{v0}={v2}+" {m}"']) for k, m in enumerate(MONTHS)]) + [
        f"covx rtc3,{v1},0,0", f'{v2}={v0}+", "+{v1}'] + IF(
        ("rtc4<10", [f'{v0}={v2}+":0"']), otherwise=[f'{v0}={v2}+":"']) + [
        f"covx rtc4,{v1},0,0", f"{v2}={v0}+{v1}",
        xstr(DATE_BOX, SMALL, MUTED, P_MENU, v2, 2)]
    bx, by, bw, bh = GOAL_BAR
    goal = ["// the goal", f"repo gtmp,{A_GOAL}"] + IF(
        ("gtmp>0", [f"gprev=gtmp*{bw}", f"gprev=gprev/{target}"]
         + IF((f"gprev>{bw}", [f"gprev={bw}"]))
         + IF(("gprev>0", [f"picq {bx},{by},gprev,{bh},{P_MENU_FULL}"])))) + [
        f"covx gtmp,{v0},0,0", f'{v2}={v0}+" / {target} punten"',
        xstr(GOAL_NUMS, SMALL, INK, P_MENU, v2, 2)] + IF(
        (f"gtmp>={target}", [xstr(GOAL_FOOT, SMALL, ACCENT, P_MENU,
                                  '"Gehaald! Tijd voor de beloning."')]),
        otherwise=[f"gprev={target}-gtmp", f"covx gprev,{v0},0,0",
                   f'{v2}="Nog "+{v0}+" punten te gaan"',
                   xstr(GOAL_FOOT, SMALL, MUTED, P_MENU, v2)])
    buttons = [(1, MENU_BUTTONS[0]), (2, MENU_BUTTONS[1]), (9, MENU_BUTTONS[2])]
    release = IF(("gknop==9", ["page taken"]),
                 ("gknop==2", ["page klok"]),
                 ("gknop==1", IF(("gzeker==0", ["gzeker=1", picq(MENU_BUTTONS[0], P_MENU_SURE)]),
                                 otherwise=["// a new goal: count from 0, with the next number",
                                            "gtmp=0", f"wepo gtmp,{A_GOAL}",
                                            f"repo gtmp,{A_GOAL_NR}", "gtmp=gtmp+1",
                                            f"wepo gtmp,{A_GOAL_NR}", "page menu"])))
    return {"Preinitialize Event": ["gknop=0", "gzeker=0"],
            "Postinitialize Event": date + goal,
            "Touch Press Event": presses(buttons, P_MENU_IN, skip={1: "gzeker==0"}),
            "Touch Release Event": release}


def clock_value(row):
    var = CLOCK_FIELDS[row][1]
    box = clock_box(row, "value")
    out = [f"covx {var},taken.va0.txt,0,0"]
    if var == "gkmi":
        out += IF(("gkmi<10", ['taken.va1.txt="0"+taken.va0.txt',
                               xstr(box, BIG, INK, P_CLOCK, "taken.va1.txt", 1)]),
                  otherwise=[xstr(box, BIG, INK, P_CLOCK, "taken.va0.txt", 1)])
        return out
    return out + [xstr(box, BIG, INK, P_CLOCK, "taken.va0.txt", 1)]


def month_days():
    """gmax: the days in month gkm of year gkj; the day stays within it."""
    return ["gmax=31"] + IF(
        ("gkm==4||gkm==6||gkm==9||gkm==11", ["gmax=30"]),
        ("gkm==2", ["gmax=28", "gtmp=gkj/4", "gtmp=gtmp*4"] + IF(("gtmp==gkj", ["gmax=29"])))
    ) + IF(("gkd>gmax", ["gkd=gmax"]))


def klok_code(cfg):
    pre = ["gknop=0"] + IF(("gkladen==0", ["gkladen=1"] + IF(
        ("rtc0<2025", ["gkj=2026", "gkm=1", "gkd=1", "gku=12", "gkmi=0"]),
        otherwise=["gkj=rtc0", "gkm=rtc1", "gkd=rtc2", "gku=rtc3", "gkmi=rtc4"])))
    post = []
    for row in range(len(CLOCK_FIELDS)):
        post += clock_value(row)
    buttons = [(9, CLOCK_BUTTONS[0]), (1, CLOCK_BUTTONS[1])]
    for row in range(len(CLOCK_FIELDS)):
        buttons += [(11 + row, clock_box(row, "minus")), (21 + row, clock_box(row, "plus"))]
    # What -/+ does per field: (minus, plus)
    change = {
        "gkd": (month_days() + ["gkd=gkd-1"] + IF(("gkd<1", ["gkd=gmax"])),
                month_days() + ["gkd=gkd+1"] + IF(("gkd>gmax", ["gkd=1"]))),
        "gkm": (["gkm=gkm-1"] + IF(("gkm<1", ["gkm=12"])) + month_days(),
                ["gkm=gkm+1"] + IF(("gkm>12", ["gkm=1"])) + month_days()),
        "gkj": (["gkj=gkj-1"] + IF(("gkj<2025", ["gkj=2025"])) + month_days(),
                ["gkj=gkj+1"] + IF(("gkj>2099", ["gkj=2099"])) + month_days()),
        "gku": (["gku=gku-1"] + IF(("gku<0", ["gku=23"])),
                ["gku=gku+1"] + IF(("gku>23", ["gku=0"]))),
        "gkmi": (["gkmi=gkmi-1"] + IF(("gkmi<0", ["gkmi=59"])),
                 ["gkmi=gkmi+1"] + IF(("gkmi>59", ["gkmi=0"]))),
    }
    branches = [
        ("gknop==1", ["// the day goes to 1 first, so every in-between date exists",
                      "rtc2=1", "rtc0=gkj", "rtc1=gkm", "rtc2=gkd", "rtc3=gku", "rtc4=gkmi",
                      "rtc5=0", "gkladen=0", "page taken"]),
        ("gknop==9", ["gkladen=0", "page taken"])]
    for row, (_, var) in enumerate(CLOCK_FIELDS):
        redraw = clock_value(row) + (clock_value(0) if var in ("gkm", "gkj") else [])
        for k, sign in enumerate(("minus", "plus")):
            code = (11, 21)[k] + row
            branches.append((f"gknop=={code}", [picq(clock_box(row, sign), P_CLOCK)]
                             + change[var][k] + redraw))
    return {"Preinitialize Event": pre, "Postinitialize Event": post,
            "Touch Press Event": presses(buttons, P_CLOCK_IN),
            "Touch Release Event": IF(*branches)}


def program_s(cfg):
    states = ",".join(f"gs{i}=0" for i in range(len(cfg["tasks"])))
    return [
        "// Onze huistaken: the numbers every page uses",
        "int sys0=0,sys1=0,sys2=0",
        "int gdag=0,gweek=0,gmaand=0,gtoon=0,gy=0,gm=0,gtmp=0,gprev=0",
        "int gtaak=0,gpts=0,gbonus=0,gwie=0,gopts=0,glast=0,gstaat=0,gdoel=0",
        "int gtx=0,gty=0,gknop=0,gzeker=0",
        "// the state of every task: 0 open, 1 done, 2 a bit late, 3 very late",
        f"int {states}",
        "// the clock page",
        "int gkladen=0,gkd=1,gkm=1,gkj=2026,gku=12,gkmi=0,gmax=31",
        "dim=100",
        "recmod=0",
        "page 0",
    ]


def pages(cfg):
    """The pages in order (page 0 first): name, picture, parts, events."""
    var = "Variable: sta=String, txt_maxl=60, vscope=global"
    return [
        ("taken", P_OPEN, [("va0", var), ("va1", var), ("va2", var),
                           ("tm0", "Timer: tim=60000, en=1")], taken_code(cfg)),
        ("wie", P_WHO, [], wie_code(cfg)),
        ("terug", P_BACK, [], terug_code(cfg)),
        ("menu", P_MENU, [], menu_code(cfg)),
        ("klok", P_CLOCK, [], klok_code(cfg)),
    ]


def nextion_code(cfg):
    bar = "=" * 72
    out = [
        "ONZE HUISTAKEN - THE CODE FOR NEXTION EDITOR",
        "Made by make_screens.py from tasks.json: edit those, not this file.",
        "Copy every block between its ---- lines exactly as it is.",
        "",
        "PICTURES (add them all at once, in this order, so the numbers match)",
    ]
    out += [f"  {i:>2}  screens\\{f:<22} {what}" for i, (f, what) in enumerate(PICTURES)]
    out += ["", "FONTS (README.txt, step 3)",
            f"   0  small: Atkinson Hyperlegible, height {SMALL_H}",
            f"   1  big:   Bricolage Grotesque ExtraBold, height {BIG_H}", "",
            bar, "PROGRAM.S  (the tab next to the pages: replace everything in it)", bar,
            "----"] + program_s(cfg) + ["----", ""]
    for k, (name, pic, parts, events) in enumerate(pages(cfg)):
        out += [bar, f"PAGE {name}   (page {k})", bar,
                f"  The page itself: sta = image, pic = {pic}"]
        if parts:
            out += ["  Add to this page (Toolbox, left):"]
            out += [f"    {pname:<4} {what}" for pname, what in parts]
        out += [""]
        for event, code in events.items():
            where = ("the timer tm0" if event.startswith("tm0") else "the page")
            out += [f"---- {event.replace('tm0 ', '')} ({where}) " + "-" * 20] + code + ["----", ""]
    return "\n".join(out)


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
        if '"' in t["name"]:
            sys.exit(f'tasks.json: no " in a name ({t["name"]})')
    if len(cfg.get("people", [])) != 2:
        sys.exit("tasks.json: people has two names")
    return cfg


def main():
    cfg = load_config()
    os.makedirs(SCREENS, exist_ok=True)
    for old in os.listdir(SCREENS):
        if old.endswith(".png") and old not in dict(PICTURES):
            os.remove(os.path.join(SCREENS, old))
    for (name, _), img in zip(PICTURES, draw_pictures(cfg)):
        img.save(os.path.join(SCREENS, name))
    with open(os.path.join(HERE, "nextion_code.txt"), "w", encoding="utf-8") as f:
        f.write(nextion_code(cfg) + "\n")
    print(f"{len(cfg['tasks'])} tasks: screens\\ ({len(PICTURES)} pictures), nextion_code.txt")
    print("Check it and make preview.png: python house_tasks/check.py")


if __name__ == "__main__":
    main()
