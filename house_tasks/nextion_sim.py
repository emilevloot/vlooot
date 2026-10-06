"""A small Nextion screen on the computer, to test the code before it goes
on the real screen (check.py uses it).

It runs the same code as nextion_code.txt, with Nextion's rules: no spaces
in a line except after the command, no brackets in sums, sums from left
to right, 4-byte numbers, a 1 KB memory, the clock (rtc0-rtc6), pictures
and the commands this project uses (page, picq, xpic, xstr, covx, wepo,
repo). It stops with an error on anything the real screen would refuse or
do differently, and draws what the screen would show.
"""
import re
from datetime import datetime, timedelta

from PIL import Image, ImageDraw

COMMANDS = {"page", "picq", "xpic", "xstr", "covx", "wepo", "repo"}
SYSTEM = {"dim", "recmod"}
TXT_MAX = 60
INT_MIN, INT_MAX = -2 ** 31, 2 ** 31 - 1


class NextionError(Exception):
    pass


class PageChange(Exception):
    def __init__(self, name):
        self.name = name


def split_args(s):
    """Split at commas outside quotes."""
    out, cur, quoted = [], "", False
    for ch in s:
        if ch == '"':
            quoted = not quoted
        if ch == "," and not quoted:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    return out + [cur]


def outside_quotes(s):
    return re.sub(r'"[^"]*"', '""', s)


def parse(lines, where):
    """Lines -> a tree of ("do", line) and ("if", [(cond, body), ...], else_body)."""
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

    def block(inner):
        nonlocal pos
        nodes = []
        while pos < len(code):
            line = code[pos]
            if line == "}" or line.startswith("}else"):
                if not inner:
                    fail("} without if")
                return nodes
            if line.startswith("if("):
                if not line.endswith(")"):
                    fail("if(...) must end the line")
                branches, otherwise = [], None
                cond = line[3:-1]
                expect_open()
                branches.append((cond, block(True)))
                while True:
                    if pos >= len(code):
                        fail("missing }")
                    line = code[pos]
                    if line == "}":
                        pos += 1
                        break
                    if line.startswith("}else if(") and line.endswith(")"):
                        cond = line[len("}else if("):-1]
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


class Nextion:
    def __init__(self, program, pages, pictures, fonts, clock=None):
        """pages: [(name, picture, parts, {event: lines})]; fonts: {id: PIL font};
        clock: a datetime, or None when the clock was never set."""
        self.pics = pictures
        self.fonts = fonts
        self.pages = {}
        self.page_ids = []
        self.strings = {}
        for name, pic, parts, events in pages:
            self.page_ids.append(name)
            trees = {ev: parse(code, f"{name} / {ev}") for ev, code in events.items()}
            self.pages[name] = (pic, trees)
            for pname, what in parts:
                if what.startswith("Variable"):
                    self.strings[f"{name}.{pname}"] = ""
        self.program = parse(program, "Program.s")
        self.vars = {}
        self.eeprom = bytearray(b"\xff" * 1024)
        self.clock = clock or datetime(2000, 1, 1, 0, 0)
        self.touch = [0, 0, 0, 0]
        self.page = None
        self.screen = Image.new("RGB", (480, 320))
        self.overflow = []          # text wider than its box: (page, text)
        self.lines_run = 0

    # ------------------------------------------------------------ start
    def boot(self):
        """Power on: Program.s, then page 0."""
        try:
            for node in self.program:
                if node[0] == "do" and node[1].startswith("int "):
                    for decl in node[1][4:].split(","):
                        name, value = decl.split("=")
                        self.vars[name] = int(value)
                else:
                    self.run_nodes([node])
        except PageChange as change:
            self.goto(change.name)

    def goto(self, name):
        for _ in range(10):
            if name.isdigit():
                name = self.page_ids[int(name)]
            if name not in self.pages:
                raise NextionError(f"no page {name}")
            self.page = name
            pic, trees = self.pages[name]
            self.screen.paste(self.pics[pic])
            try:
                for ev in ("Preinitialize Event", "Postinitialize Event"):
                    if ev in trees:
                        self.run_nodes(trees[ev])
                return
            except PageChange as change:
                name = change.name
        raise NextionError("pages keep changing")

    def event(self, ev):
        trees = self.pages[self.page][1]
        if ev in trees:
            try:
                self.run_nodes(trees[ev])
            except PageChange as change:
                self.goto(change.name)

    # ------------------------------------------------------------ the world
    def press(self, x, y):
        self.touch[0], self.touch[1] = x, y
        self.event("Touch Press Event")

    def release(self):
        self.touch = [0, 0, self.touch[0], self.touch[1]]
        self.event("Touch Release Event")

    def tap(self, x, y):
        self.press(x, y)
        self.release()

    def tap_box(self, box):
        x, y, w, h = box
        self.tap(x + w // 2, y + h // 2)

    def timer(self):
        self.event("tm0 Timer Event")

    def later(self, **delta):
        """Time passes (the clock runs on its battery)."""
        self.clock += timedelta(**delta)

    def power_cycle(self, keep_clock=True):
        """Power off and on: the memory and (with a battery) the clock stay."""
        self.vars = {}
        for k in self.strings:
            self.strings[k] = ""
        if not keep_clock:
            self.clock = datetime(2000, 1, 1, 0, 0)
        self.boot()

    def peek(self, addr):
        """A number from the memory."""
        return int.from_bytes(self.eeprom[addr:addr + 4], "little", signed=True)

    # ------------------------------------------------------------ running code
    def run_nodes(self, nodes):
        for node in nodes:
            if node[0] == "do":
                self.lines_run += 1
                self.do(node[1])
            else:
                _, branches, otherwise = node
                for cond, body in branches:
                    if self.test(cond):
                        self.run_nodes(body)
                        break
                else:
                    if otherwise is not None:
                        self.run_nodes(otherwise)

    def get(self, name):
        if re.fullmatch(r"-?\d+", name):
            return int(name)
        if re.fullmatch(r"rtc[0-6]", name):
            c = self.clock
            return [c.year, c.month, c.day, c.hour, c.minute, c.second,
                    (c.weekday() + 1) % 7][int(name[3])]
        if re.fullmatch(r"tch[0-3]", name):
            return self.touch[int(name[3])]
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
        elif name in SYSTEM:
            pass
        elif name in self.vars:
            self.vars[name] = value
        else:
            raise NextionError(f"unknown number {name}")

    def string_key(self, ref):
        parts = ref.split(".")
        if len(parts) == 2:
            parts = [self.page] + parts
        if len(parts) != 3 or parts[2] != "txt":
            raise NextionError(f"not a text: {ref}")
        key = f"{parts[0]}.{parts[1]}"
        if key not in self.strings:
            raise NextionError(f"unknown text {ref} on page {self.page}")
        return key

    def number(self, expr):
        tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[-+*/]", expr)
        if "".join(tokens) != expr or not tokens:
            raise NextionError(f"can't read the sum {expr}")
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

    def text(self, expr):
        out = ""
        for part in split_plus(expr):
            if part.startswith('"') and part.endswith('"'):
                out += part[1:-1]
            else:
                out += self.strings[self.string_key(part)]
        return out

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
            return getattr(self, "cmd_" + name)(split_args(rest))
        lhs, eq, rhs = line.partition("=")
        if not eq:
            raise NextionError(f"can't read {line}")
        if lhs.endswith(".txt"):
            value = self.text(rhs)
            if len(value) > TXT_MAX:
                raise NextionError(f"text longer than txt_maxl {TXT_MAX}: {value}")
            self.strings[self.string_key(lhs)] = value
        else:
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
        raise PageChange(args[0])

    def cmd_picq(self, args):
        x, y, w, h, pid = self.nums(args, 5)
        self.region(x, y, w, h)
        self.screen.paste(self.picture(pid).crop((x, y, x + w, y + h)), (x, y))

    def cmd_xpic(self, args):
        x, y, w, h, sx, sy, pid = self.nums(args, 7)
        self.region(x, y, w, h)
        self.region(sx, sy, w, h)
        self.screen.paste(self.picture(pid).crop((sx, sy, sx + w, sy + h)), (x, y))

    def cmd_xstr(self, args):
        if len(args) != 11:
            raise NextionError(f"xstr has 11 values: {args}")
        x, y, w, h, fid, pco, bco, xcen, ycen, sta = [self.number(a) for a in args[:10]]
        s = self.text(args[10])
        self.region(x, y, w, h)
        if sta != 0:
            raise NextionError("this project only uses sta 0 (crop image)")
        box = self.picture(bco).crop((x, y, x + w, y + h))
        f = self.fonts[fid]
        tw = f.getlength(s)
        if tw > w:
            self.overflow.append((self.page, s, round(tw), w))
        color = ((pco >> 11 & 31) * 255 // 31, (pco >> 5 & 63) * 255 // 63, (pco & 31) * 255 // 31)
        tx = [0, (w - tw) / 2, w - tw][xcen]
        anchor = ["lt", "lm", "lb"][ycen]
        ty = [0, h / 2, h][ycen]
        ImageDraw.Draw(box).text((tx, ty), s, font=f, fill=color, anchor=anchor)
        self.screen.paste(box, (x, y))

    def cmd_covx(self, args):
        if len(args) != 4 or args[2:] != ["0", "0"]:
            raise NextionError(f"covx: this project uses covx number,text,0,0: {args}")
        self.strings[self.string_key(args[1])] = str(self.number(args[0]))

    def addr(self, a):
        addr = self.number(a)
        if not 0 <= addr <= 1020:
            raise NextionError(f"memory place {addr} is outside the 1 KB")
        return addr

    def cmd_wepo(self, args):
        value, addr = self.number(args[0]), self.addr(args[1])
        self.eeprom[addr:addr + 4] = value.to_bytes(4, "little", signed=True)

    def cmd_repo(self, args):
        addr = self.addr(args[1])
        self.set(args[0], self.peek(addr))


def split_plus(expr):
    out, cur, quoted = [], "", False
    for ch in expr:
        if ch == '"':
            quoted = not quoted
        if ch == "+" and not quoted:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    return out + [cur]
