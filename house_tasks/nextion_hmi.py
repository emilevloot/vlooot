"""Writes a Nextion Editor project file (.HMI) without Nextion Editor, and
reads one back to check it.

The .HMI is the project Nextion Editor opens: a container of sections
(main.HMI with the screen model, Program.s, one section per page, two per
picture). Its layout and its three checksums are documented by Newmatik's
nextion-hmi-writer research (MIT licence):
    https://github.com/newmatik/nextion-hmi-writer
The fixed bytes below (the project header and page header) are those of a
real NX4832K035_011 project saved by Nextion Editor 1.65 (WPSD_Nextion,
M17 Project), and the checksum code was checked against that file and nine
other real projects.
"""
import io
import struct

POLY = 0x04C11DB7
PAYLOAD_START = 0x700000            # sections always start at 7 MiB
MIRROR = 0x80000                    # a copy of the directory
DIR_RECORD = 28


def _table_entry(value):
    for _ in range(8):
        value = ((value << 1) & 0xFFFFFFFF) ^ (POLY if value & 0x80000000 else 0)
    return value


TABLE = [_table_entry(i << 24) for i in range(256)]


def crc(data, state=0xFFFFFFFF):
    """The editor's checksum: CRC-32/MPEG-2 table, four rounds per byte."""
    for value in data:
        state ^= value
        for _ in range(4):
            state = ((state << 8) & 0xFFFFFFFF) ^ TABLE[state >> 24]
    return state


def crc_words(data, state=0xFFFFFFFF):
    for (value,) in struct.iter_unpack("<I", data):
        state ^= value
        for _ in range(4):
            state = ((state << 8) & 0xFFFFFFFF) ^ TABLE[state >> 24]
    return state


def page_checksum(page):
    state = crc(page[4:])
    state = crc(struct.pack("<I", len(page)), state)
    state = crc(page[12:16], state)
    return crc(b"\x00\x4f", state)


def main_checksum(main):
    state = crc(main[4:])
    for a, b in ((16, 20), (4, 8), (10, 11), (14, 15)):
        state = crc(main[a:b], state)
    return state


def directory_checksum(directory):
    return crc_words(directory + b"ADEC")


def model_code(model):
    """The screen model as stored in main.HMI, e.g. NX4832K035_011."""
    return crc(model.encode("ascii"))


# ---------------------------------------------------------------- components
# Attribute order and widths as Nextion Editor writes them (width None: text)
COMMON = [("type", 1), ("id", 1), ("objname", None), ("vscope", 1)]
GROUPS = [("lockobj", 1), ("groupid0", 4), ("groupid1", 4)]
PAGE = COMMON + [
    ("drag", 1), ("sendkey", 1), ("aph", 1), ("movex", 2), ("movey", 2), ("x", 2), ("y", 2),
    ("w", 2), ("h", 2), ("endx", 2), ("endy", 2), ("effect", 1), ("first", 1), ("time", 2),
] + GROUPS + [("up", 1), ("down", 1), ("left", 1), ("right", 1), ("sta", 1), ("bco", 2),
              ("pic", 2)]
TIMER = COMMON + GROUPS + [("tim", 2), ("en", 1)]
VARIABLE = COMMON + GROUPS + [("sta", 1), ("txt", None), ("txt_maxl", 2), ("val", 4)]
TYPES = {"page": (121, PAGE, ["load", "loadend", "down", "up", "unload"]),
         "timer": (51, TIMER, ["timer"]),
         "variable": (52, VARIABLE, [])}
DEFAULTS = {"aph": 127, "time": 300, "up": 255, "down": 255, "left": 255, "right": 255,
            "sta": 1, "pic": 65535, "tim": 400, "en": 1, "txt": "", "txt_maxl": 1}


def _record(raw):
    return struct.pack("<I", len(raw)) + raw


def component(kind, cid, name, events=None, **values):
    """One object: its attributes, then its event code, then 4 zero bytes."""
    type_id, attributes, event_names = TYPES[kind]
    values = {**DEFAULTS, **values, "type": type_id, "id": cid, "objname": name}
    if "w" in values:
        values["endx"] = values.get("x", 0) + values["w"] - 1
        values["endy"] = values.get("y", 0) + values["h"] - 1
    out = _record(f"att-{len(attributes)}".encode("ascii"))
    for attr, width in attributes:
        value = values.get(attr, 0)
        raw = value.encode("latin-1") if width is None else int(value).to_bytes(
            width, "little", signed=attr == "val")
        out += _record(attr.encode("ascii").ljust(16, b"\x00") + raw)
    events = events or {}
    for event in event_names:
        lines = events.get(event, [])
        out += _record(f"codes{event}-{len(lines)}".encode("ascii"))
        for line in lines:
            out += _record(line.encode("latin-1"))
    return out + b"\x00\x00\x00\x00"


def page_section(name, objects):
    """A page: 56-byte header, object table, objects; with its checksum."""
    table = b""
    start = len(objects) * 12
    for obj in objects:
        table += struct.pack("<III", start, len(obj), 0)
        start += len(obj)
    header = bytearray(56)
    struct.pack_into("<III", header, 8, 0x38, len(objects), 0)
    header[0x10:0x18] = bytes.fromhex("0000000000 4f2100".replace(" ", ""))
    header[0x18:0x18 + len(name)] = name.encode("ascii")
    header[0x28:0x2C] = bytes.fromhex("01410100")
    page = bytearray(header + table + b"".join(objects))
    struct.pack_into("<I", page, 4, len(page))
    struct.pack_into("<I", page, 0, page_checksum(page))
    return bytes(page)


# ---------------------------------------------------------------- pictures
def picture_sections(img):
    """A picture: .is holds the PNG, .i the editor's preview (RGB565)."""
    w, h = img.size
    png = io.BytesIO()
    img.convert("RGB").save(png, "PNG", optimize=True)
    png = png.getvalue()
    full = (b"\x0a\x64\x01\x01" + bytes(4) + struct.pack("<IHHI", 0x1B, w, h, len(png))
            + bytes(4) + b"png" + png)
    rgb = img.convert("RGB").tobytes()
    pixels = bytearray(w * h * 2)
    for k in range(w * h):
        r, g, b = rgb[3 * k], rgb[3 * k + 1], rgb[3 * k + 2]
        struct.pack_into("<H", pixels, 2 * k, (r >> 3) << 11 | (g >> 2) << 5 | b >> 3)
    preview = (b"\x0b\x64\x01\x00" + bytes(4) + struct.pack("<IHHI", 0x18, w, h, len(pixels))
               + bytes(4) + bytes(pixels))
    return preview, full


# ---------------------------------------------------------------- the file
MAIN_HEADER = bytes.fromhex(
    "00000000 60000000 01412101 01034f00 be24d31b 00000000 60000000 00000000"
    "00000000 01000000" + "00" * 56)        # NX4832K035_011, iso-8859-1, editor 1.65


def main_section(pictures, pages, model="NX4832K035_011"):
    resources = [("i", f"{n}.i") for n in range(pictures)]
    resources += [("pa", f"{n}.pa") for n in range(pages)]
    main = bytearray(MAIN_HEADER)
    struct.pack_into("<I", main, 0x10, model_code(model))
    struct.pack_into("<I", main, 0x1C, len(resources))
    for ext, name in resources:
        main += ext.encode("ascii").ljust(8, b"\x00") + name.encode("ascii").ljust(8, b"\x00")
    struct.pack_into("<I", main, 0, main_checksum(main))
    return bytes(main)


def build(program_s, page_name, objects, images, model="NX4832K035_011"):
    """The whole .HMI: Program.s, one page, the pictures (PIL images)."""
    pics = [picture_sections(img) for img in images]
    sections = [("Program.s", "\r\n".join(program_s).encode("latin-1") + b"\r\n")]
    sections += [(f"{n}.is", full) for n, (_, full) in enumerate(pics)]
    sections += [(f"{n}.i", preview) for n, (preview, _) in enumerate(pics)]
    sections += [("0.pa", page_section(page_name, objects)),
                 ("main.HMI", main_section(len(pics), 1, model))]
    directory = struct.pack("<I", len(sections))
    payload, start = b"", PAYLOAD_START
    for name, data in sections:
        directory += name.encode("ascii").ljust(16, b"\x00") + struct.pack("<II", start, len(data))
        directory += bytes(4)                       # live, reserved
        payload += data
        start += len(data)
    directory += struct.pack("<I", directory_checksum(directory))
    lead = bytearray(PAYLOAD_START)
    lead[0:len(directory)] = directory
    lead[MIRROR:MIRROR + len(directory)] = directory
    lead[0x380000:0x380004] = b"\xff\xff\xff\xff"
    lead[0x6FFFF8:0x700000] = b"ver21234"
    return bytes(lead) + payload


# ---------------------------------------------------------------- reading back
class HmiError(Exception):
    pass


def read(data):
    """Checks every rule above and returns {section name: bytes}."""
    count = struct.unpack_from("<I", data, 0)[0]
    end = 4 + count * DIR_RECORD
    directory = data[:end]
    if struct.unpack_from("<I", data, end)[0] != directory_checksum(directory):
        raise HmiError("directory checksum")
    if data[MIRROR:MIRROR + end + 4] != data[:end + 4]:
        raise HmiError("directory copy at 0x80000 differs")
    if data[0x6FFFF8:0x700000] != b"ver21234":
        raise HmiError("marker ver21234 missing")
    sections = {}
    for k in range(count):
        o = 4 + k * DIR_RECORD
        name = data[o:o + 16].split(b"\x00")[0].decode("ascii")
        start, size = struct.unpack_from("<II", data, o + 16)
        if start < PAYLOAD_START or start + size > len(data):
            raise HmiError(f"{name} outside the file")
        if name in sections:
            raise HmiError(f"{name} twice")
        sections[name] = data[start:start + size]
    main = sections["main.HMI"]
    if struct.unpack_from("<I", main, 0)[0] != main_checksum(main):
        raise HmiError("main.HMI checksum")
    listed = [main[o + 8:o + 16].split(b"\x00")[0].decode("ascii")
              for o in range(0x60, len(main), 16)]
    if len(listed) != struct.unpack_from("<I", main, 0x1C)[0]:
        raise HmiError("main.HMI resource count")
    for name in listed:
        if name not in sections or (name.endswith(".i") and name + "s" not in sections):
            raise HmiError(f"main.HMI lists {name}, which is missing")
    for name, sec in sections.items():
        if name.endswith(".pa"):
            if struct.unpack_from("<I", sec, 0)[0] != page_checksum(sec):
                raise HmiError(f"{name} checksum")
            if struct.unpack_from("<I", sec, 4)[0] != len(sec):
                raise HmiError(f"{name} size")
    return sections


def read_page(sec):
    """A page section -> [(attributes dict, {event: [lines]})] per object."""
    count = struct.unpack_from("<I", sec, 12)[0]
    objects, expect = [], count * 12
    for k in range(count):
        start, size, _ = struct.unpack_from("<III", sec, 0x38 + 12 * k)
        if start != expect:
            raise HmiError("object table not packed")
        expect += size
        body, p = sec[0x38 + start:0x38 + start + size], 0
        attrs, events, lines_left, event = {}, {}, 0, None
        while True:
            n = struct.unpack_from("<I", body, p)[0]
            p += 4
            if n == 0:
                break
            raw = body[p:p + n]
            p += n
            if lines_left:
                events[event].append(raw.decode("latin-1"))
                lines_left -= 1
            elif n < 16:
                text = raw.decode("ascii")
                if text.startswith("codes"):
                    event, lines = text[5:].rsplit("-", 1)
                    events[event], lines_left = [], int(lines)
            else:
                attrs[raw[:16].split(b"\x00")[0].decode("ascii")] = raw[16:]
        if p != len(body):
            raise HmiError("object has bytes after its end")
        objects.append((attrs, events))
    if 0x38 + expect != len(sec):
        raise HmiError("page size and objects differ")
    return objects
