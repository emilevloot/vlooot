"""A case for the Nextion NX4832K035 (3.5"): front, back and a stand.

Makes houder.step (all parts, to open and change in Fusion 360) and one
STL per part to print, plus preview.png.
    pip install cadquery matplotlib
    python house_tasks/case/make_case.py

The screen is held by its edges, so the case doesn't need the screen's own
mounting holes; front and back close with 4 M3 screws (16 mm) beside it.
Measure your screen and change the numbers below if they differ.
"""
import math
import os

import cadquery as cq

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- the screen (mm) -------------------------------------------------------
PCB_L, PCB_W = 100.5, 54.94   # the board (the glass is as high as the board)
MODULE_T = 5.45               # front of the glass to the back of the board
GLASS_L = 85.5                # the glass, along the length
GLASS_FROM_EDGE = 15.0        # board edge (side WITHOUT the white plug) to the glass
VIEW_L, VIEW_W = 73.44, 48.96 # the part that lights up
PARTS_H = 10.0                # room behind the board (parts, battery, plug)
PLUG_SPACE = 14.0             # room past the board at the white plug
PLUG_SLOT = (16.0, 8.0)       # opening for the cable: width, height

# ---- the case --------------------------------------------------------------
CLEAR = 0.4                   # play around the board
WALL = 2.4
FRONT_T = 1.6                 # in front of the glass
BACK_T = 2.0
RADIUS = 5.0
END_SPACE = 9.0               # room for the screws at the other end
WINDOW_MARGIN = 0.8           # window a bit bigger than the lit part
SCREW_HOLE, SCREW_PILOT, BOSS_D = 3.4, 2.5, 6.5
TILT = 15                     # the stand leans the screen back by this many degrees

INNER_L = END_SPACE + PCB_L + 2 * CLEAR + PLUG_SPACE
INNER_W = PCB_W + 2 * CLEAR
OUT_L, OUT_W = INNER_L + 2 * WALL, INNER_W + 2 * WALL
FRONT_H = FRONT_T + MODULE_T              # the front holds the screen
BACK_H = PARTS_H + BACK_T                 # the back covers its back
PCB_X0 = WALL + END_SPACE + CLEAR         # where the board starts (x)
SCREWS = [(WALL + END_SPACE / 2, y) for y in (WALL + 5, OUT_W - WALL - 5)] + \
         [(OUT_L - WALL - PLUG_SPACE / 2, y) for y in (WALL + 5, OUT_W - WALL - 5)]


def box(l, w, h, r=RADIUS):
    return cq.Workplane("XY").box(l, w, h, centered=False).edges("|Z").fillet(r)


def front():
    """Lies face down when printed: z = 0 is the front of the case."""
    body = box(OUT_L, OUT_W, FRONT_H)
    body = body.cut(box(INNER_L, INNER_W, FRONT_H, RADIUS - WALL)
                    .translate((WALL, WALL, FRONT_T)))
    # the window in front of the lit part, with a chamfer towards the viewer
    wl, ww = VIEW_L + 2 * WINDOW_MARGIN, VIEW_W + 2 * WINDOW_MARGIN
    wx = PCB_X0 + GLASS_FROM_EDGE + (GLASS_L - VIEW_L) / 2 - WINDOW_MARGIN
    wy = (OUT_W - ww) / 2
    window = (cq.Workplane("XY").box(wl, ww, FRONT_T + 0.2, centered=False)
              .translate((wx, wy, -0.1)))
    body = body.cut(window)
    body = body.faces("<Z").edges(cq.selectors.BoxSelector(
        (wx - 1, wy - 1, -1), (wx + wl + 1, wy + ww + 1, 0.5))).chamfer(0.8)
    # stops at both ends of the board, and the screw posts
    for x in (PCB_X0 - CLEAR - 1.2, PCB_X0 + PCB_L + CLEAR):
        for y0, y1 in ((WALL, WALL + 14), (OUT_W - WALL - 14, OUT_W - WALL)):
            body = body.union(cq.Workplane("XY").box(1.2, y1 - y0, MODULE_T, centered=False)
                              .translate((x, y0, FRONT_T)))
    for x, y in SCREWS:
        post = (cq.Workplane("XY").circle(BOSS_D / 2).circle(SCREW_PILOT / 2)
                .extrude(MODULE_T).translate((x, y, FRONT_T)))
        body = body.union(post)
    return body


def back():
    """Lies on its back when printed: z = 0 is the back of the case."""
    body = box(OUT_L, OUT_W, BACK_H)
    body = body.cut(box(INNER_L, INNER_W, BACK_H, RADIUS - WALL).translate((WALL, WALL, BACK_T)))
    for x, y in SCREWS:                     # the screws go in from the back
        post = (cq.Workplane("XY").circle(BOSS_D / 2).extrude(BACK_H).translate((x, y, 0)))
        body = body.union(post)
        body = body.cut(cq.Workplane("XY").circle(SCREW_HOLE / 2).extrude(BACK_H + 1)
                        .translate((x, y, -0.5)))
        body = body.cut(cq.Workplane("XY").circle(3.2).extrude(2.0).translate((x, y, -0.01)))
    # four pins press the board into the front, 4 mm in from its corners
    # (where its mounting holes are), 0.2 mm short so nothing bends
    x0 = PCB_X0 + 4
    for x in (x0, x0 + PCB_L - 8):
        for y in (WALL + CLEAR + 4, OUT_W - WALL - CLEAR - 4):
            body = body.union(cq.Workplane("XY").circle(2.0).extrude(BACK_H - BACK_T - 0.2)
                              .translate((x, OUT_W - y, BACK_T)))
    # the cable leaves at the end with the white plug, at the open edge
    sw, sh = PLUG_SLOT
    body = body.cut(cq.Workplane("XY").box(WALL + 2, sw, sh, centered=False)
                    .translate((OUT_L - WALL - 1, (OUT_W - sw) / 2, BACK_H - sh)))
    # two keyholes to hang it on screws in the wall, 60 mm apart
    for x in (OUT_L / 2 - 30, OUT_L / 2 + 30):
        y = OUT_W * 0.62
        body = body.cut(cq.Workplane("XY").circle(4.2).extrude(BACK_T + 1).translate((x, y, -0.5)))
        body = body.cut(cq.Workplane("XY").box(4.4, 8, BACK_T + 1, centered=False)
                        .translate((x - 2.2, y - 8, -0.5)))
    return body


def stand():
    """A foot with a slot: the case stands in it, leaning back TILT degrees."""
    depth = FRONT_H + BACK_H                # how thick the closed case is
    l, d, h = 90.0, 55.0, 16.0
    base = box(l, d, h, 6)
    slot = (cq.Workplane("XY").box(l + 2, depth + 0.6, 40, centered=False)
            .translate((-1, -(depth + 0.6) / 2, 0))
            .rotate((0, 0, 0), (1, 0, 0), -TILT)
            .translate((0, d / 2 - 4, h - 9)))
    return base.cut(slot)


def preview(parts, path):
    """Shaded pictures of the parts (matplotlib)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure(figsize=(12, 8.4), dpi=100)
    for k, (name, part, view) in enumerate(parts):
        verts, tris = part.val().tessellate(0.1, 0.2)
        pts = np.array([(v.x, v.y, v.z) for v in verts])
        faces = pts[np.array(tris)]
        normal = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
        normal /= np.linalg.norm(normal, axis=1, keepdims=True) + 1e-9
        elev, azim = np.radians(view[0]), np.radians(view[1])
        eye = np.array([np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)])
        light = eye + np.array([0.3, 0.2, 0.5])
        light /= np.linalg.norm(light)
        shade = 0.35 + 0.65 * np.abs(normal @ light)
        base = np.array([0.80, 0.86, 0.82])
        ax = fig.add_subplot(2, 2, k + 1, projection="3d")
        ax.add_collection3d(Poly3DCollection(faces, facecolors=shade[:, None] * base,
                                             edgecolor="none"))
        span = (pts.max(0) - pts.min(0)).max() / 2
        for setter, mid in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), (pts.max(0) + pts.min(0)) / 2):
            setter(mid - span, mid + span)
        ax.view_init(*view)
        ax.set_axis_off()
        ax.set_title(name)
    fig.tight_layout()
    fig.savefig(path)


def main():
    # Seen from the front the white plug is on the right: mirror the parts so
    # the plug end (built at the high x end) comes out right.
    mirror = lambda part: part.mirror("YZ", (OUT_L / 2, 0, 0))
    parts = {"voorkant": mirror(front()), "achterkant": mirror(back()), "voet": stand()}
    for name, part in parts.items():
        cq.exporters.export(part, os.path.join(HERE, f"{name}.stl"), tolerance=0.02,
                            angularTolerance=0.1)
    assy = cq.Assembly()
    assy.add(parts["voorkant"], name="voorkant", color=cq.Color(0.85, 0.87, 0.85))
    assy.add(parts["achterkant"].rotate((0, 0, 0), (1, 0, 0), 180)
             .translate((0, OUT_W, FRONT_H + BACK_H)), name="achterkant",
             color=cq.Color(0.6, 0.65, 0.62))
    assy.add(parts["voet"].translate((OUT_L / 2 - 45, -80, 0)), name="voet",
             color=cq.Color(0.18, 0.42, 0.31))
    assy.export(os.path.join(HERE, "houder.step"))
    preview([("voorkant: voorzijde", parts["voorkant"], (-55, -75)),
             ("voorkant: binnenkant", parts["voorkant"], (55, -105)),
             ("achterkant: binnenkant", parts["achterkant"], (55, -75)),
             ("voet", parts["voet"], (30, -60))], os.path.join(HERE, "preview.png"))
    print(f"case {OUT_L:.1f} x {OUT_W:.1f} x {FRONT_H + BACK_H:.1f} mm; "
          "houder.step, voorkant.stl, achterkant.stl, voet.stl, preview.png")


if __name__ == "__main__":
    main()
