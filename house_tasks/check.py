"""Tests Huistaken.HMI on the computer and makes preview.png. Run it after
make_screens.py:
    python house_tasks/check.py
It reads the project file like Nextion Editor would (every checksum), runs
its code in the small Nextion (nextion_sim.py), and plays a year of random
ticks, undos, new goals and power cuts, comparing the screen with the
app's rules after every step.
"""
import math
import os
import random
import sys
from datetime import date, datetime, timedelta

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_screens as ms                                   # noqa: E402
from nextion_sim import Nextion, NextionError, parse       # noqa: E402

HERE = ms.HERE


def tile_center(i):
    x, y, w, h = ms.tile_box(i)
    return x + w // 2, y + h // 2


class Model:
    """The app's rules (index.html of Onze huistaken), for comparison."""

    def __init__(self, cfg):
        self.tasks = cfg["tasks"]
        self.last = [None] * len(self.tasks)      # date of the last tick
        self.undo = [None] * len(self.tasks)      # (prev date, tick) while undo is possible
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
        tick = [today, who, pts, self.goal_nr, True]
        self.ticks.append(tick)
        self.undo[i] = (self.last[i], tick)
        self.last[i] = today

    def untick(self, i):
        prev, tick = self.undo[i]
        tick[4] = False
        self.last[i] = prev
        self.undo[i] = None

    def power_cut(self):
        self.undo = [None] * len(self.tasks)      # undo lives in the screen's RAM only

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


def check_screen(nx, model, cfg, state_code, pictures):
    """The scores in the memory and every task (worked out by the screen's own
    code, and as drawn) against the app's rules."""
    today = nx.clock.date()
    assert nx.vars["gscherm"] == ms.TAKEN, nx.vars["gscherm"]
    week, month, goal = model.scores(today)
    got = ([nx.peek(a) for a in ms.A_WEEK], [nx.peek(a) for a in ms.A_MONTH], nx.peek(ms.A_GOAL))
    assert got == (week, month, goal), f"{today}: scores {got} != {(week, month, goal)}"
    for i in range(len(cfg["tasks"])):
        nx.vars["gi"] = i
        nx.run(state_code)
        want = model.state(i, today)
        assert (nx.vars["gs"], nx.vars["gq"]) == want, f"{today} task {i}: {nx.vars['gs'], nx.vars['gq']} != {want}"
        x, y, w, h = ms.tile_box(i)                 # the name part of the tile, as drawn
        box = (x, y, x + w, y + 30)
        assert nx.screen.crop(box).tobytes() == pictures[want[0]].crop(box).tobytes(), \
            f"{today} task {i}: tile drawn wrong"


def new_screen(data, clock=None):
    nx = Nextion(data, clock)
    nx.ticks(1)
    return nx


def set_clock(nx, when):
    """Set the clock with the - and + buttons, as a person would."""
    assert nx.vars["gscherm"] == ms.KLOK
    targets = [when.day, when.month, when.year, when.hour, when.minute]
    for _ in range(3):                    # the day can depend on month and year
        for row in (1, 2, 0, 3, 4):
            var = ms.CLOCK_FIELDS[row][1]
            while nx.vars[var] != targets[row]:
                part = "plus" if nx.vars[var] < targets[row] else "minus"
                nx.tap_box(ms.clock_box(row, part))
    nx.tap_box(ms.CLOCK_BUTTONS[1])
    nx.ticks(1)


def test_project(data, project):
    """The file on disk is the one tasks.json makes, and it reads back whole."""
    path = ms.PROJECT
    on_disk = open(path, "rb").read() if os.path.exists(path) else b""
    assert on_disk == data, "Huistaken.HMI is not up to date: run make_screens.py"
    nx = Nextion(data)
    assert len(nx.pics) == len(project["pictures"])
    for a, b in zip(nx.pics, project["pictures"]):
        assert a.tobytes() == b.tobytes(), "a picture changed on the way"
    assert nx.program == project["program"] + [""]
    print(f"Huistaken.HMI: {len(data) // 1024} KB, every checksum right, "
          f"{len(nx.pics)} pictures, 1 page, {len(nx.parts) - 1} parts read back")


def test_day_numbers(data):
    """gdag counts days, gweek is the Monday, gmaand the 1st (2025-2099)."""
    nx = Nextion(data, datetime(2026, 1, 1, 12))
    code = parse(ms.Code.day_numbers(), "day numbers")
    d, base = date(2025, 1, 1), None
    while d < date(2100, 1, 1):
        nx.clock = datetime(d.year, d.month, d.day, 12)
        nx.run(code)
        off = nx.vars["gdag"] - d.toordinal()
        base = off if base is None else base
        assert off == base, f"day number of {d}"
        assert nx.vars["gweek"] - base == (d - timedelta(days=d.weekday())).toordinal(), d
        assert nx.vars["gmaand"] - base == d.replace(day=1).toordinal(), d
        d += timedelta(days=1)
    print("day numbers: right for every day 2025-2099")


def test_clock(data):
    nx = new_screen(data)                               # never set: the clock page
    assert nx.vars["gscherm"] == ms.KLOK
    for when in (datetime(2027, 2, 28, 7, 5), datetime(2028, 2, 29, 23, 59),
                 datetime(2026, 12, 31, 0, 0)):
        set_clock(nx, when)
        assert nx.clock == when, (nx.clock, when)
        nx.tap_box(ms.MENU_BOX)
        nx.tap_box(ms.MENU_BUTTONS[1])
    nx.tap_box(ms.clock_box(1, "plus"))                 # December -> January
    while nx.vars["gkd"] != 31:
        nx.tap_box(ms.clock_box(0, "plus"))
    nx.tap_box(ms.clock_box(1, "plus"))                 # 31 January -> February: 28
    assert (nx.vars["gkm"], nx.vars["gkd"]) == (2, 28), (nx.vars["gkm"], nx.vars["gkd"])
    nx.tap_box(ms.CLOCK_BUTTONS[0])                     # Annuleren: clock unchanged
    nx.ticks(1)
    assert nx.clock == datetime(2026, 12, 31, 0, 0) and nx.vars["gscherm"] == ms.TAKEN
    nx.power_cycle(keep_clock=False)                    # no battery: asks for the time again
    nx.ticks(20)
    assert nx.vars["gscherm"] == ms.KLOK
    print("clock page: sets the clock, keeps the day within the month, asks again without battery")


def test_year(data, project, seed=1, days=400):
    """Random ticks, undos and power cuts; compared with the app's rules."""
    cfg, pictures = project["code"].cfg, project["pictures"]
    state_code = parse(project["code"].task_state(), "task state")
    rnd = random.Random(seed)
    nx = new_screen(data)
    set_clock(nx, datetime(2026, 10, 6, 7, 30))
    model = Model(cfg)
    n = len(cfg["tasks"])
    counts = {"tick": 0, "again": 0, "undo": 0, "cancel": 0, "goal": 0, "power": 0}
    check_screen(nx, model, cfg, state_code, pictures)
    for _ in range(days):
        for _ in range(rnd.randint(0, 5)):
            today = nx.clock.date()
            i = rnd.randrange(n)
            nx.tap(*tile_center(i))
            state, pts = model.state(i, today)
            if state == ms.P_DONE:
                assert nx.vars["gscherm"] == ms.TERUG
                undo = model.undo[i]
                assert (nx.vars["gwie"] > 0) == (undo is not None)
                choice = rnd.choice(["undo", "again", "back"])
                if choice == "undo" and undo:
                    nx.tap_box(ms.BACK_BUTTONS[0])
                    model.untick(i)
                    counts["undo"] += 1
                elif choice == "again":
                    nx.tap_box(ms.BACK_BUTTONS[1])
                    assert nx.vars["gscherm"] == ms.WIE and nx.vars["gq"] == pts
                    who = rnd.randint(1, 3)
                    nx.tap_box(ms.WHO_BUTTONS[who - 1])
                    model.tick(i, who, today)
                    counts["again"] += 1
                else:
                    nx.tap_box(ms.BACK_BUTTONS[2])
            else:
                assert nx.vars["gscherm"] == ms.WIE, nx.vars["gscherm"]
                assert nx.vars["gq"] == pts, (today, i, nx.vars["gq"], pts)
                who = rnd.randint(0, 3)
                if who == 0:
                    nx.tap_box(ms.WHO_CANCEL)
                    counts["cancel"] += 1
                else:
                    nx.tap_box(ms.WHO_BUTTONS[who - 1])
                    model.tick(i, who, today)
                    counts["tick"] += 1
            check_screen(nx, model, cfg, state_code, pictures)
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
            model.power_cut()
            counts["power"] += 1
        nx.later(hours=rnd.choice([24, 24, 24, 48]))     # the timer sees the new day
        check_screen(nx, model, cfg, state_code, pictures)
    print(f"{days} days: {counts}; scores, tasks and tiles match the app's rules "
          f"({nx.lines_run // days} lines of code run per day)")


def test_fit(project):
    """The longest texts stay inside their space."""
    sp, cfg = project["sprites"], project["code"].cfg
    dw = lambda s: ms.digit_width(s)
    worst = max(sp[("soon", "nooit gedaan")].w, sp[("done", "over ")].w + 3 * dw("done")
                + sp[("done", " d")].w, 3 * dw("late") + sp[("late", " d te laat")].w)
    for i, task in enumerate(cfg["tasks"]):
        f, uf = ms.font("display", 14), ms.font("regular", 11)
        pill = 9 + max(f.getlength(str(p)) for p in (task["points"], *ms.bonus_points(task))) \
            + uf.getlength("pt") + 14
        assert ms.STATUS_RIGHT - worst > pill + 4, f"status text runs into the points of task {i}"
    print("texts: every status fits next to its points")


def screenshots(data, project):
    """A few days of use, for preview.png."""
    nx = new_screen(data)
    set_clock(nx, datetime(2026, 9, 26, 9, 0))
    plan = {0: [(6, 3), (7, 1), (9, 2), (10, 1), (11, 2)],          # (task, who)
            4: [(3, 2), (4, 1), (5, 3)],
            8: [(0, 1), (1, 2), (3, 1)],
            10: [(0, 2), (2, 1), (4, 3), (8, 2)]}
    for day in range(11):
        for i, who in plan.get(day, []):
            nx.tap(*tile_center(i))
            if nx.vars["gscherm"] == ms.TERUG:
                nx.tap_box(ms.BACK_BUTTONS[1])
            nx.tap_box(ms.WHO_BUTTONS[who - 1])
        nx.later(days=1)
    shots = [nx.screen.copy()]
    cfg, state = project["code"].cfg, parse(project["code"].task_state(), "state")
    states = []
    for i in range(len(cfg["tasks"])):
        nx.vars["gi"] = i
        nx.run(state)
        states.append(nx.vars["gs"])
    late = max(range(len(states)), key=lambda i: states[i] % 3 * 10 - i)
    nx.tap(*tile_center(late))
    shots.append(nx.screen.copy())
    nx.tap_box(ms.WHO_CANCEL)
    done = next(i for i in range(len(states)) if states[i] == ms.P_DONE)
    nx.tap(*tile_center(done))
    shots.append(nx.screen.copy())
    nx.tap_box(ms.BACK_BUTTONS[2])
    nx.tap_box(ms.MENU_BOX)
    shots.append(nx.screen.copy())
    nx.tap_box(ms.MENU_BUTTONS[1])
    nx.tap_box(ms.clock_box(3, "plus"))
    shots.append(nx.screen.copy())
    return shots


def main():
    cfg = ms.load_config()
    project = ms.build(cfg)
    data = ms.hmi_bytes(project)
    try:
        test_project(data, project)
        test_day_numbers(data)
        test_clock(data)
        test_fit(project)
        test_year(data, project)
        shots = screenshots(data, project)
    except (NextionError, AssertionError) as e:
        sys.exit(f"PROBLEM: {e}")
    pad = 16
    sheet = Image.new("RGB", (3 * ms.W + 4 * pad, 2 * ms.H + 3 * pad), (60, 60, 60))
    for k, img in enumerate(shots):
        sheet.paste(img, (pad + k % 3 * (ms.W + pad), pad + k // 3 * (ms.H + pad)))
    sheet.save(os.path.join(HERE, "preview.png"))
    print("preview.png: the task list, wie, terug, menu, klok")


if __name__ == "__main__":
    main()
