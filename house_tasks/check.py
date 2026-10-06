"""Tests the screen code in the small Nextion (nextion_sim.py) and makes
preview.png. Run it after make_screens.py:
    python house_tasks/check.py
It plays a year of random ticks, undos and power cuts and compares the
screen with the app's rules after every step.
"""
import math
import os
import random
import sys
from datetime import date, datetime, timedelta

from PIL import Image, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_screens as ms                                  # noqa: E402
from nextion_sim import Nextion, NextionError, parse      # noqa: E402

HERE = ms.HERE


def nextion_font(kind, height):
    """The font at the size the Font Generator makes for this height."""
    path = os.path.join(HERE, "fonts", ms.FONT_FILES[kind])
    size = height
    while sum(ImageFont.truetype(path, size).getmetrics()) > height:
        size -= 1
    return ImageFont.truetype(path, size)


def new_screen(cfg, pictures, clock=None):
    fonts = {ms.SMALL: nextion_font("regular", ms.SMALL_H),
             ms.BIG: nextion_font("display", ms.BIG_H)}
    nx = Nextion(ms.program_s(cfg), ms.pages(cfg), pictures, fonts, clock)
    nx.boot()
    return nx


def tile_center(i):
    x, y, w, h = ms.tile_box(i)
    return x + w // 2, y + h // 2


class Model:
    """The app's rules (index.html of Onze huistaken), for comparison."""

    def __init__(self, cfg):
        self.tasks = cfg["tasks"]
        self.last = [None] * len(self.tasks)      # date of the last tick
        self.undo = [None] * len(self.tasks)      # (prev date, who, points) or None
        self.ticks = []                           # [day, who, points, goal nr, alive]
        self.goal_nr = 0

    def state(self, i, today):
        t = self.tasks[i]
        if self.last[i] is None:
            return ms.P_OPEN, t["points"]
        days = (self.last[i] + timedelta(days=t["every"]) - today).days
        if days > 0:
            return ms.P_DONE, t["points"]
        late = -days
        if late / t["every"] >= 1 and late >= 3:
            return ms.P_URGENT, 2 * t["points"]
        if late / t["every"] >= .5 and late >= 2:
            return ms.P_HIGH, t["points"] + math.ceil(t["points"] / 2)
        return ms.P_OPEN, t["points"]

    def tick(self, i, who, today):
        pts = self.state(i, today)[1]
        self.undo[i] = (self.last[i], who, pts, len(self.ticks))
        self.ticks.append([today, who, pts, self.goal_nr, True])
        self.last[i] = today
        return pts

    def untick(self, i):
        prev, _, _, k = self.undo[i]
        self.ticks[k][4] = False
        self.last[i] = prev
        self.undo[i] = None

    def scores(self, today):
        monday = today - timedelta(days=today.weekday())
        first = today.replace(day=1)
        week, month, goal = [0, 0], [0, 0], 0
        for day, who, pts, goal_nr, alive in self.ticks:
            if not alive:
                continue
            for k in (0, 1):
                if who in (k + 1, 3):
                    week[k] += pts if day >= monday else 0
                    month[k] += pts if day >= first else 0
            if goal_nr == self.goal_nr:
                goal += pts * (2 if who == 3 else 1)
        return week, month, goal


def check_screen(nx, model, cfg):
    today = nx.clock.date()
    assert nx.page == "taken", nx.page
    week, month, goal = model.scores(today)
    got = ([nx.peek(a) for a in ms.A_WEEK], [nx.peek(a) for a in ms.A_MONTH], nx.peek(ms.A_GOAL))
    assert got == (week, month, goal), f"{today}: scores {got} != {(week, month, goal)}"
    for i in range(len(cfg["tasks"])):
        want = model.state(i, today)[0]
        assert nx.vars[f"gs{i}"] == want, f"{today} task {i}: state {nx.vars[f'gs{i}']} != {want}"


def test_day_numbers(cfg, pictures):
    """gdag counts days, gweek is the Monday, gmaand the 1st (2025-2099)."""
    nx = new_screen(cfg, pictures, datetime(2026, 1, 1, 12))
    code = parse(ms.day_numbers(), "day numbers")
    d, base = date(2025, 1, 1), None
    while d < date(2100, 1, 1):
        nx.clock = datetime(d.year, d.month, d.day, 12)
        nx.run_nodes(code)
        off = nx.vars["gdag"] - d.toordinal()
        base = off if base is None else base
        assert off == base, f"day number of {d}"
        assert nx.vars["gweek"] - base == (d - timedelta(days=d.weekday())).toordinal(), d
        assert nx.vars["gmaand"] - base == d.replace(day=1).toordinal(), d
        d += timedelta(days=1)
    print("day numbers: right for every day 2025-2099")


def set_clock(nx, when):
    """Set the clock with the - and + buttons, as a person would."""
    assert nx.page == "klok"
    targets = [when.day, when.month, when.year, when.hour, when.minute]
    for _ in range(3):                    # the day can depend on month and year
        for row in (1, 2, 0, 3, 4):
            var = ms.CLOCK_FIELDS[row][1]
            while nx.vars[var] != targets[row]:
                part = "plus" if nx.vars[var] < targets[row] else "minus"
                nx.tap_box(ms.clock_box(row, part))
    nx.tap_box(ms.CLOCK_BUTTONS[1])


def test_clock(cfg, pictures):
    nx = new_screen(cfg, pictures)                       # never set: the clock page
    assert nx.page == "klok"
    for when in (datetime(2027, 2, 28, 7, 5), datetime(2028, 2, 29, 23, 59),
                 datetime(2026, 12, 31, 0, 0)):
        set_clock(nx, when)
        assert nx.clock == when, (nx.clock, when)
        nx.tap_box(ms.MENU_BOX)
        nx.tap_box(ms.MENU_BUTTONS[1])
    # 31 January -> February: the day goes to 28 (or 29)
    nx.tap_box(ms.clock_box(1, "plus"))                  # December -> January
    while nx.vars["gkd"] != 31:
        nx.tap_box(ms.clock_box(0, "plus"))
    nx.tap_box(ms.clock_box(1, "plus"))
    assert (nx.vars["gkm"], nx.vars["gkd"]) == (2, 28), (nx.vars["gkm"], nx.vars["gkd"])
    nx.tap_box(ms.clock_box(2, "plus"))                  # 2027 -> 2028 keeps 28
    nx.tap_box(ms.CLOCK_BUTTONS[0])                      # Annuleren: clock unchanged
    assert nx.clock == datetime(2026, 12, 31, 0, 0) and nx.page == "taken"
    print("clock page: sets the clock, keeps the day within the month")


def test_year(cfg, pictures, seed=1, days=400):
    """Random ticks, undos and power cuts; compared with the app's rules."""
    rnd = random.Random(seed)
    nx = new_screen(cfg, pictures)
    set_clock(nx, datetime(2026, 10, 6, 7, 30))
    model = Model(cfg)
    n = len(cfg["tasks"])
    counts = {"tick": 0, "again": 0, "undo": 0, "cancel": 0, "goal": 0, "power": 0}
    check_screen(nx, model, cfg)
    for _ in range(days):
        for _ in range(rnd.randint(0, 5)):
            today = nx.clock.date()
            i = rnd.randrange(n)
            nx.tap(*tile_center(i))
            state, pts = model.state(i, today)
            if state == ms.P_DONE:
                assert nx.page == "terug"
                undo = model.undo[i]
                assert (nx.vars["gwie"] > 0) == (undo is not None)
                choice = rnd.choice(["undo", "again", "back"])
                if choice == "undo" and undo:
                    nx.tap_box(ms.BACK_BUTTONS[0])
                    model.untick(i)
                    counts["undo"] += 1
                elif choice == "again":
                    nx.tap_box(ms.BACK_BUTTONS[1])
                    assert nx.page == "wie" and nx.vars["gpts"] == pts
                    who = rnd.randint(1, 3)
                    nx.tap_box(ms.WHO_BUTTONS[who - 1])
                    model.tick(i, who, today)
                    counts["again"] += 1
                else:
                    nx.tap_box(ms.BACK_BUTTONS[2])
            else:
                assert nx.page == "wie", nx.page
                assert nx.vars["gpts"] == pts, (today, i, nx.vars["gpts"], pts)
                who = rnd.randint(0, 3)
                if who == 0:
                    nx.tap_box(ms.WHO_CANCEL)
                    counts["cancel"] += 1
                else:
                    nx.tap_box(ms.WHO_BUTTONS[who - 1])
                    model.tick(i, who, today)
                    counts["tick"] += 1
            check_screen(nx, model, cfg)
        if rnd.random() < .03:                           # a new goal (two taps)
            nx.tap_box(ms.MENU_BOX)
            nx.tap_box(ms.MENU_BUTTONS[0])
            assert nx.vars["gzeker"] == 1
            nx.tap_box(ms.MENU_BUTTONS[0])
            nx.tap_box(ms.MENU_BUTTONS[2])
            model.goal_nr += 1
            counts["goal"] += 1
        if rnd.random() < .05:                           # the power goes off and on
            nx.power_cycle()
            counts["power"] += 1
        nx.later(hours=rnd.choice([24, 24, 24, 48]))
        nx.timer()                                       # the minute timer sees a new day
        check_screen(nx, model, cfg)
    print(f"{days} days: {counts}; scores and task states match the app's rules")
    return nx, model


def screenshots(cfg, pictures):
    """A few days of use, for preview.png."""
    nx = new_screen(cfg, pictures)
    set_clock(nx, datetime(2026, 9, 26, 9, 0))
    klok_shot = None
    plan = {0: [(6, 3), (7, 1), (9, 2), (10, 1), (11, 2)],          # (task, who)
            4: [(3, 2), (4, 1), (5, 3)],
            8: [(0, 1), (1, 2), (3, 1)],
            10: [(0, 2), (2, 1), (4, 3), (8, 2)]}
    for day in range(11):
        for i, who in plan.get(day, []):
            nx.tap(*tile_center(i))
            if nx.page == "terug":
                nx.tap_box(ms.BACK_BUTTONS[1])
            nx.tap_box(ms.WHO_BUTTONS[who - 1])
        nx.later(days=1)
        nx.timer()
    shots = [nx.screen.copy()]
    late = max(range(len(cfg["tasks"])), key=lambda i: nx.vars[f"gs{i}"] % 3 * 10 - i)
    nx.tap(*tile_center(late))
    shots.append(nx.screen.copy())
    nx.tap_box(ms.WHO_CANCEL)
    done = next(i for i in range(len(cfg["tasks"])) if nx.vars[f"gs{i}"] == ms.P_DONE
                and nx.peek(ms.a_who(i)) > 0)
    nx.tap(*tile_center(done))
    shots.append(nx.screen.copy())
    nx.tap_box(ms.BACK_BUTTONS[2])
    nx.tap_box(ms.MENU_BOX)
    shots.append(nx.screen.copy())
    nx.tap_box(ms.MENU_BUTTONS[1])
    nx.press(*[v + d // 2 for v, d in zip(ms.clock_box(3, "plus")[:2], ms.clock_box(3, "plus")[2:])])
    klok_shot = nx.screen.copy()
    nx.release()
    shots.append(klok_shot)
    return shots, nx


def main():
    cfg = ms.load_config()
    pictures = ms.draw_pictures(cfg)
    try:
        test_day_numbers(cfg, pictures)
        test_clock(cfg, pictures)
        nx, _ = test_year(cfg, pictures)
        shots, nx2 = screenshots(cfg, pictures)
    except (NextionError, AssertionError) as e:
        sys.exit(f"PROBLEM: {e}")
    overflow = {(p, s): (tw, w) for p, s, tw, w in nx.overflow + nx2.overflow}
    for (p, s), (tw, w) in sorted(overflow.items()):
        print(f"  too wide on {p}: '{s}' ({tw} px in {w})")
    lines = {f"{name} / {ev}": len([l for l in code if l.strip() and not l.strip().startswith('//')])
             for name, _, _, events in ms.pages(cfg) for ev, code in events.items()}
    print(f"code: {sum(lines.values())} lines; longest: "
          + ", ".join(f"{k} {v}" for k, v in sorted(lines.items(), key=lambda kv: -kv[1])[:3]))
    pad = 16
    sheet = Image.new("RGB", (3 * ms.W + 4 * pad, 2 * ms.H + 3 * pad), (60, 60, 60))
    for k, img in enumerate(shots):
        sheet.paste(img, (pad + k % 3 * (ms.W + pad), pad + k // 3 * (ms.H + pad)))
    sheet.save(os.path.join(HERE, "preview.png"))
    print("preview.png: the task list, wie, terug, menu, klok")


if __name__ == "__main__":
    main()
