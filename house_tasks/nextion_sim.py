"""A small Nextion screen on the computer, to test Huistaken.HMI before it
goes on the real screen (check.py uses it).

It reads the project file itself (with nextion_hmi.py) and runs its code
with Nextion's rules: no spaces in a line except after the command, no
brackets in sums, sums from left to right, 4-byte numbers, a 1 KB memory,
the clock (rtc0-rtc6), touch (tch0-tch3), the parts on the page (b[...]),
the timer and the drawing commands this project uses (pic, picq, xpic).
It stops with an error on anything the real screen would refuse or do
differently, and draws what the screen would show.

Two stricter rules than Nextion's own keep the code on safe ground: a
memory address is always a plain number, and b[...] holds a plain variable.
"""
import io
import re
import struct
from datetime import datetime, timedelta

from PIL import Image

import nextion_hmi

COMMANDS = {"pic", "picq", "xpic", "wepo", "repo", "page"}
SYSTEM = {"dim", "recmod", "baud"}
INT_MIN, INT_MAX = -2 ** 31, 2 ** 31 - 1
EVENTS = {"loadend": "Postinitialize", "down": "Touch Press", "timer": "Timer"}


class NextionError(Exception):
    pass


def outside_quotes(s):
    return re.sub(r'"[^"]*"', '""', s)


def parse(lines, where):
    """Lines -> a tree: ("do", line), ("if", [(cond, body)...], else_body),
    ("while", cond, body)."""
    code = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith("//"):
            code.append(line)
    pos = 0

    def fail(msg):
        raise NextionError(f"{where}: {msg}: {code[pos] if pos < len(code) else 'end'}")

    def expect_open():
        nonlocal pos
        pos += 1
        if pos >= len(code) or code[pos] != "{":
            fail("{ must follow on its own line")
        pos += 1

    def condition(line, head):
        cond = line[len(head):-1]
        if not line.endswith(")") or "(" in cond or ")" in cond or " " in cond:
            fail("a condition is one line without spaces or inner brackets")
        return cond

    def block(inner):
        nonlocal pos
        nodes = []
        while pos < len(code):
            line = code[pos]
            if line == "}" or line.startswith("}else"):
                if not inner:
                    fail("} without if")
                return nodes
            if line.startswith("while("):
                cond = condition(line, "while(")
                expect_open()
                body = block(True)
                if pos >= len(code) or code[pos] != "}":
                    fail("missing } after while")
                pos += 1
                nodes.append(("while", cond, body))
                continue
            if line.startswith("if("):
                branches, otherwise = [], None
                cond = condition(line, "if(")
                expect_open()
                branches.append((cond, block(True)))
                while True:
                    if pos >= len(code):
                        fail("missing }")
                    line = code[pos]
                    if line == "}":
                        pos += 1
                        break
                    if line.startswith("}else if("):
                        cond = condition(line, "}else if(")
                        expect_open()
                        branches.append((cond, block(True)))
                    elif line == "}else":
                        expect_open()
                        otherwise = block(True)
                        if pos >= len(code) or code[pos] != "}":
                            fail("missing } after else")
                        pos += 1
                        break
                    else:
                        fail("expected } or }else")
                nodes.append(("if", branches, otherwise))
                continue
            if line == "{":
                fail("{ without if")
            bare = outside_quotes(line)
            name, _, rest = bare.partition(" ")
            if " " in (rest if name in COMMANDS | {"int"} else bare):
                fail("a space inside the line")
            if "(" in bare or ")" in bare:
                fail("brackets (Nextion has none in sums)")
            nodes.append(("do", line))
            pos += 1
        if inner:
            fail("missing }")
        return nodes

    return block(False)


def load_hmi(data):
    """Huistaken.HMI -> what the screen needs: Program.s, the pictures, the
    page and its parts (each with its attributes and event code)."""
    sections = nextion_hmi.read(data)
    program = sections["Program.s"].decode("latin-1").split("\r\n")
    pictures, n = [], 0
    while f"{n}.is" in sections:
        full, preview = sections[f"{n}.is"], sections[f"{n}.i"]
        if full[:4] != b"\x0a\x64\x01\x01" or full[0x18:0x1B] != b"png":
            raise NextionError(f"picture {n}: not a png section")
        img = Image.open(io.BytesIO(full[0x1B:])).convert("RGB")
        w, h = struct.unpack_from("<HH", full, 12)
        if img.size != (w, h) or struct.unpack_from("<HH", preview, 12) != (w, h):
            raise NextionError(f"picture {n}: sizes differ")
        rgb = img.tobytes()
        for k in range(0, w * h, 997):                  # the preview shows the same picture
            r, g, b = rgb[3 * k:3 * k + 3]
            if struct.unpack_from("<H", preview, 24 + 2 * k)[0] != (r >> 3 << 11 | g >> 2 << 5 | b >> 3):
                raise NextionError(f"picture {n}: preview differs")
        pictures.append(img)
        n += 1
    objects = nextion_hmi.read_page(sections["0.pa"])
    parts = []
    for attrs, events in objects:
        values = {k: (v.decode("latin-1") if k in ("objname", "txt") else
                      int.from_bytes(v, "little", signed=k == "val")) for k, v in attrs.items()}
        parts.append((values, events))
    return program, pictures, parts


class Nextion:
    def __init__(self, data, clock=None):
        """data: the .HMI file; clock: a datetime, or None when never set."""
        self.program, self.pics, parts = load_hmi(data)
        self.page, self.page_events = parts[0][0], self.compile(parts[0][1], "page")
        self.parts = [values for values, _ in parts]          # by id (0 = the page)
        self.names = {values["objname"]: values for values in self.parts}
        self.timer_part = next(v for v in self.parts if v["type"] == 51)
        self.timer_code = self.compile(parts[self.timer_part["id"]][1], "tm0")["timer"]
        self.vars = {}
        self.eeprom = bytearray(b"\xff" * 1024)
        self.clock = clock or datetime(2000, 1, 1, 0, 0)
        self.touch = [0, 0, 0, 0]
        self.screen = Image.new("RGB", (480, 320))
        self.lines_run = 0
        self.boot()

    @staticmethod
    def compile(events, where):
        return {ev: parse(lines, f"{where} / {EVENTS.get(ev, ev)}") for ev, lines in events.items()}

    # ------------------------------------------------------------ the world
    def boot(self):
        """Power on: Program.s, then the page loads (its picture, then its code)."""
        for part in self.parts:
            if part["type"] == 52:
                part["val"] = part.get("start", part["val"])
                part["start"] = part["val"]
        self.vars = {}
        for node in parse(self.program, "Program.s"):
            line = node[1]
            if line.startswith("int "):
                for decl in line[4:].split(","):
                    name, value = decl.split("=")
                    self.vars[name] = int(value)
            elif line.startswith("page "):
                if line != "page 0":
                    raise NextionError("Program.s must open page 0")
            else:
                self.run([node])
        if self.page["sta"] == 2:
            self.screen.paste(self.picture(self.page["pic"]))
        self.run(self.page_events.get("load", []))
        self.run(self.page_events.get("loadend", []))

    def tap(self, x, y):
        """A finger on the screen (the code acts on the press), then a tick to draw."""
        self.touch[0], self.touch[1] = x, y
        self.run(self.page_events.get("down", []))
        self.touch = [0, 0, x, y]
        self.ticks(1)

    def tap_box(self, box):
        x, y, w, h = box
        self.tap(x + w // 2, y + h // 2)

    def ticks(self, n):
        """n timer ticks (each tim ms of time; the clock is moved separately)."""
        for _ in range(n):
            if self.timer_part["en"]:
                self.run(self.timer_code)

    def later(self, **delta):
        """Time passes (the clock runs on its battery), then a second of ticks."""
        self.clock += timedelta(**delta)
        self.ticks(20)

    def power_cycle(self, keep_clock=True):
        """Power off and on: the memory and (with a battery) the clock stay."""
        if not keep_clock:
            self.clock = datetime(2000, 1, 1, 0, 0)
        self.boot()
        self.ticks(1)

    def peek(self, addr):
        return int.from_bytes(self.eeprom[addr:addr + 4], "little", signed=True)

    # ------------------------------------------------------------ running code
    def run(self, nodes):
        for node in nodes:
            if node[0] == "do":
                self.lines_run += 1
                self.do(node[1])
            elif node[0] == "while":
                guard = 0
                while self.test(node[1]):
                    self.run(node[2])
                    guard += 1
                    if guard > 10000:
                        raise NextionError(f"endless loop: while({node[1]})")
            else:
                for cond, body in node[1]:
                    if self.test(cond):
                        self.run(body)
                        break
                else:
                    if node[2] is not None:
                        self.run(node[2])

    def part(self, ref):
        """vl0.val or b[gid].val -> (the part's attributes, attribute name)."""
        m = re.fullmatch(r"b\[([A-Za-z_][A-Za-z0-9_]*|\d+)\]\.(\w+)", ref)
        if m:
            cid = self.get(m[1])
            if not 0 < cid < len(self.parts):
                raise NextionError(f"b[{cid}]: no such part")
            return self.parts[cid], m[2]
        name, _, attr = ref.partition(".")
        if name not in self.names:
            raise NextionError(f"no part {name}")
        return self.names[name], attr

    def get(self, name):
        if re.fullmatch(r"-?\d+", name):
            return int(name)
        if re.fullmatch(r"rtc[0-6]", name):
            c = self.clock
            return [c.year, c.month, c.day, c.hour, c.minute, c.second,
                    (c.weekday() + 1) % 7][int(name[3])]
        if re.fullmatch(r"tch[0-3]", name):
            return self.touch[int(name[3])]
        if "." in name:
            part, attr = self.part(name)
            if attr not in part:
                raise NextionError(f"{name}: no attribute {attr}")
            return part[attr]
        if name in self.vars:
            return self.vars[name]
        raise NextionError(f"unknown number {name}")

    def set(self, name, value):
        if not INT_MIN <= value <= INT_MAX:
            raise NextionError(f"{name}={value} doesn't fit in 4 bytes")
        if re.fullmatch(r"rtc[0-5]", name):
            field = ["year", "month", "day", "hour", "minute", "second"][int(name[3])]
            try:
                self.clock = self.clock.replace(**{field: value})
            except ValueError:
                raise NextionError(f"{name}={value}: not a real date/time "
                                   f"(clock {self.clock:%Y-%m-%d %H:%M})")
        elif "." in name:
            part, attr = self.part(name)
            if attr not in ("val", "en", "tim"):
                raise NextionError(f"{name}: can't be changed")
            part[attr] = value
        elif name in SYSTEM:
            pass
        elif name in self.vars:
            self.vars[name] = value
        else:
            raise NextionError(f"unknown number {name}")

    def number(self, expr):
        tokens = re.findall(r"b\[\w+\]\.\w+|[A-Za-z_][A-Za-z0-9_.]*|\d+|[-+*/]", expr)
        if "".join(tokens) != expr or not tokens:
            raise NextionError(f"can't read the sum {expr}")
        if tokens[0] == "-" and len(tokens) == 2:         # a negative number: gdelta=-1
            return -self.get(tokens[1])
        value = self.get(tokens[0])
        for op, operand in zip(tokens[1::2], tokens[2::2]):
            b = self.get(operand)
            if op == "+":
                value += b
            elif op == "-":
                value -= b
            elif op == "*":
                value *= b
            else:
                if b == 0:
                    raise NextionError(f"divide by 0 in {expr}")
                value = int(value / b)        # towards 0, as in C
            if not INT_MIN <= value <= INT_MAX:
                raise NextionError(f"{expr} doesn't fit in 4 bytes")
        return value

    def test(self, cond):
        parts = re.split(r"(&&|\|\|)", cond)
        result = self.compare(parts[0])
        for op, part in zip(parts[1::2], parts[2::2]):   # left to right
            result = (result and self.compare(part)) if op == "&&" else (result or self.compare(part))
        return result

    def compare(self, s):
        m = re.fullmatch(r"(.+?)(==|!=|>=|<=|>|<)(.+)", s)
        if not m:
            raise NextionError(f"can't read the condition {s}")
        a, op, b = self.number(m[1]), m[2], self.number(m[3])
        return {"==": a == b, "!=": a != b, ">=": a >= b, "<=": a <= b,
                ">": a > b, "<": a < b}[op]

    def do(self, line):
        name, _, rest = line.partition(" ")
        if name in COMMANDS and rest:
            return getattr(self, "cmd_" + name)(rest.split(","))
        lhs, eq, rhs = line.partition("=")
        if not eq:
            raise NextionError(f"can't read {line}")
        self.set(lhs, self.number(rhs))

    def nums(self, args, count):
        if len(args) != count:
            raise NextionError(f"expected {count} values: {args}")
        return [self.number(a) for a in args]

    def region(self, x, y, w, h):
        if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > 480 or y + h > 320:
            raise NextionError(f"off the screen: {x},{y},{w},{h}")

    def picture(self, pid):
        if not 0 <= pid < len(self.pics):
            raise NextionError(f"no picture {pid}")
        return self.pics[pid]

    def cmd_page(self, args):
        raise NextionError("this project never changes page")

    def cmd_pic(self, args):
        x, y, pid = self.nums(args, 3)
        img = self.picture(pid)
        self.region(x, y, *img.size)
        self.screen.paste(img, (x, y))

    def cmd_picq(self, args):
        x, y, w, h, pid = self.nums(args, 5)
        self.region(x, y, w, h)
        self.screen.paste(self.picture(pid).crop((x, y, x + w, y + h)), (x, y))

    def cmd_xpic(self, args):
        x, y, w, h, sx, sy, pid = self.nums(args, 7)
        self.region(x, y, w, h)
        self.region(sx, sy, w, h)
        self.screen.paste(self.picture(pid).crop((sx, sy, sx + w, sy + h)), (x, y))

    @staticmethod
    def addr(a):
        if not re.fullmatch(r"\d+", a):
            raise NextionError(f"memory address {a}: this project uses plain numbers")
        addr = int(a)
        if not 0 <= addr <= 1020:
            raise NextionError(f"memory place {addr} is outside the 1 KB")
        return addr

    def cmd_wepo(self, args):
        if len(args) != 2:
            raise NextionError(f"wepo value,address: {args}")
        value, addr = self.number(args[0]), self.addr(args[1])
        self.eeprom[addr:addr + 4] = value.to_bytes(4, "little", signed=True)

    def cmd_repo(self, args):
        if len(args) != 2:
            raise NextionError(f"repo name,address: {args}")
        self.set(args[0], self.peek(self.addr(args[1])))
