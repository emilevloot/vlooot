"""
Looot - a browser version of the Viking board game by Charles Chevallier
and Laurent Escoffier (Gigamic, 2023).

This module holds the whole game: board, rules, scoring and a simple
computer opponent. It has no browser code in it, so you can run and test
it with plain Python:

    python looot.py            # plays a full 3-player game between computers

The browser page (index.html) loads this file with Pyodide and talks to it
through the `Api` class at the bottom, which returns JSON strings.

Coordinates: all boards use "axial" hex coordinates (q, r) with flat-top
hexes, like the printed boards: q is the column, and a cell's height on
screen is r + q/2. The six neighbours of (q, r) are listed in DIRS.
"""

import copy
import json
import random

# ---------------------------------------------------------------------------
# Rule data (from the official rulebook)
# ---------------------------------------------------------------------------

DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]

RESOURCES = ["wood", "sheep", "gold", "axe"]
BUILDINGS = ["house", "watchtower", "castle"]

# Landscape terrain -> resource you gain when you place a Viking there
TERRAIN_RESOURCE = {"forest": "wood", "field": "sheep",
                    "mountain": "gold", "battlefield": "axe"}

# Starting (base) victory-point values, printed on the fjord board
BASE_VALUE = {"castle": 4, "watchtower": 2, "house": 1,
              "gold": 2, "sheep": 1, "wood": 1}
SCORE_ORDER = ["castle", "watchtower", "house", "gold", "sheep", "wood"]

# At setup a stack of tiles goes on every building space of the gameboard:
# 2 on each House, 2 on each Watchtower and 3 on each Castle. Capturing a
# building takes tiles from its own stack; an empty stack gives nothing.
BUILDING_STACK = {"house": 2, "watchtower": 2, "castle": 3}

# (axes needed, victory points)
TROPHIES = [(2, 3), (3, 6), (4, 10), (5, 15), (6, 21)]

SITE_VP = {"port": 5, "altar": 7, "jarl": 9}
SITE_NAMES = {"port": "Port", "altar": "Altar", "jarl": "Jarl Palace"}

# One tile of each type is shown in the rulebook (the first entry of each
# list). The other four of each type are not printed in the rules, so they
# are made up here with the same size and mix.
SITES = {
    "port": [
        ["house", "axe", "gold"],
        ["house", "sheep", "wood"],
        ["house", "gold", "sheep"],
        ["watchtower", "axe", "wood"],
        ["house", "axe", "sheep"],
    ],
    "altar": [
        ["watchtower", "house", "axe", "house"],
        ["house", "house", "watchtower", "wood"],
        ["house", "watchtower", "watchtower", "sheep"],
        ["house", "watchtower", "watchtower", "gold"],
        ["house", "house", "watchtower", "sheep"],
    ],
    "jarl": [
        ["house", "wood", "watchtower", "castle", "watchtower"],
        ["house", "house", "castle", "watchtower", "gold"],
        ["house", "castle", "watchtower", "sheep", "axe"],
        ["house", "house", "castle", "watchtower", "wood"],
        ["watchtower", "castle", "watchtower", "house", "sheep"],
    ],
}

# The 30 longships, from the last page of the rulebook:
# (the 3 resources needed, category that gets the bonus, bonus value)
LONGSHIPS = [
    (("gold", "gold", "axe"), "castle", 4),
    (("sheep", "sheep", "axe"), "castle", 3),
    (("wood", "wood", "wood"), "castle", 3),
    (("axe", "axe", "wood"), "castle", 3),
    (("gold", "gold", "sheep"), "sheep", 2),
    (("gold", "wood", "sheep"), "sheep", 2),
    (("gold", "axe", "sheep"), "sheep", 1),
    (("wood", "wood", "sheep"), "sheep", 1),
    (("wood", "axe", "sheep"), "sheep", 1),
    (("axe", "axe", "sheep"), "sheep", 1),
    (("sheep", "sheep", "gold"), "gold", 3),
    (("sheep", "wood", "gold"), "gold", 2),
    (("wood", "wood", "gold"), "gold", 2),
    (("axe", "wood", "gold"), "gold", 2),
    (("gold", "gold", "wood"), "wood", 2),
    (("gold", "sheep", "wood"), "wood", 1),
    (("sheep", "sheep", "wood"), "wood", 1),
    (("axe", "gold", "wood"), "wood", 1),
    (("axe", "sheep", "wood"), "wood", 1),
    (("axe", "axe", "wood"), "wood", 1),
    (("sheep", "sheep", "gold"), "watchtower", 2),
    (("wood", "wood", "gold"), "watchtower", 2),
    (("wood", "wood", "sheep"), "watchtower", 1),
    (("axe", "axe", "gold"), "watchtower", 1),
    (("wood", "wood", "axe"), "watchtower", 1),
    (("sheep", "sheep", "sheep"), "house", 2),
    (("sheep", "wood", "sheep"), "house", 1),
    (("axe", "axe", "gold"), "house", 1),
    (("axe", "axe", "sheep"), "house", 1),
    (("axe", "axe", "axe"), "house", 1),
]

VIKINGS_PER_PLAYER = 13
COLORS = ["red", "blue", "yellow", "grey"]

# Rules that can be switched off, for teaching the neural network the game
# step by step ("curriculum"). A normal game uses all of them. Without
# longships, the ocean still counts as a place to start placing Vikings.
FULL_RULES = {"buildings": True, "sites": True, "longships": True,
              "shields": True, "trophies": True}
RULE_STAGES = [
    ("resources", {"buildings": False, "sites": False, "longships": False,
                   "shields": False, "trophies": False}),
    ("buildings", {"sites": False, "longships": False,
                   "shields": False, "trophies": False}),
    ("sites", {"longships": False, "shields": False, "trophies": False}),
    ("longships", {"shields": False, "trophies": False}),
    ("full", {}),
]


def rules_for(stage):
    """The rules dict of a stage name from RULE_STAGES (None = full game)."""
    if stage is None:
        return dict(FULL_RULES)
    for name, off in RULE_STAGES:
        if name == stage:
            return dict(FULL_RULES, **off)
    raise ValueError("unknown rule stage %r" % stage)

# Each landscape board is a triangle of 25 flat-top hexes (measured from
# the rulebook art), with a front (A) and back (B) side. BOARD_SHAPE lists
# the spaces column by column, left to right and top to bottom; a board
# side in boards.json is a list of 25 space types in this same order.
BOARD_SHAPE = [(-3, 5), (-3, 6),
               (-2, 3), (-2, 4), (-2, 5),
               (-1, 1), (-1, 2), (-1, 3), (-1, 4), (-1, 5),
               (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
               (1, 0), (1, 1), (1, 2), (1, 3), (1, 4),
               (2, 1), (2, 2), (2, 3),
               (3, 2), (3, 3)]
BOARD_SIZE = len(BOARD_SHAPE)
N_BOARDS = 4
# How each board lies in the gameboard: (turns of 60 degrees, q shift,
# r shift). Boards 1-3 are the 3-player setup picture of the rulebook
# (pointing up, down, up); board 4 continues the strip pointing down.
BOARD_PLACEMENT = [(2, 6, 3), (1, 10, -4), (0, 12, -5), (1, 19, -9)]
# The 5 spaces of the Ocean board, along the bottom edge of board 1
OCEAN_CELLS = [(1, 7), (2, 6), (3, 6), (4, 5), (5, 5)]
OCEAN_SLOT = {c: i for i, c in enumerate(OCEAN_CELLS)}

# Every board side printed in the rulebook has this mix of spaces
BOARD_BUILDINGS = ["castle", "watchtower", "watchtower", "house", "house"]
BOARD_TERRAIN = (["forest"] * 6 + ["field"] * 5 +
                 ["mountain"] * 3 + ["battlefield"] * 6)
LAND_TYPES = ["forest", "field", "mountain", "battlefield"] + BUILDINGS

# Fjord board: 7 columns of flat-top hexes (34 empty spaces) plus the
# three construction sites, measured from the rulebook art.
FJORD_COLUMNS = {0: range(0, 5), 1: range(0, 5), 2: range(-1, 4),
                 3: range(-3, 4), 4: range(-3, 3), 5: range(-3, 2),
                 6: range(-3, 1)}
FJORD_SITE_SPOTS = {"altar": (1, 0), "port": (5, -3), "jarl": (4, 1)}


def key(c):
    return "%d,%d" % c


def unkey(s):
    q, r = s.split(",")
    return (int(q), int(r))


_NEIGHBOURS = {}


def neighbours(c):
    """The 6 neighbouring positions of c (worked out once, then remembered)."""
    nb = _NEIGHBOURS.get(c)
    if nb is None:
        nb = _NEIGHBOURS[c] = tuple((c[0] + dq, c[1] + dr) for dq, dr in DIRS)
    return nb


def rotate(c, turns):
    """Turn a hex position by 60 degrees `turns` times around (0, 0)."""
    x, z = c
    y = -x - z
    for _ in range(turns % 6):
        x, y, z = -z, -x, -y
    return (x, z)


def sub_multiset(need, have):
    """True if every item in `need` is in `have` (counting repeats)."""
    pool = list(have)
    for n in need:
        if n in pool:
            pool.remove(n)
        else:
            return False
    return True


def matched_count(need, have):
    pool = list(have)
    m = 0
    for n in need:
        if n in pool:
            pool.remove(n)
            m += 1
    return m


# ---------------------------------------------------------------------------
# Board generation
# ---------------------------------------------------------------------------

def board_cells(n_boards):
    """Return a list of lists: the gameboard cells of each landscape board,
    in BOARD_SHAPE order, turned and shifted as in BOARD_PLACEMENT."""
    boards = []
    for turns, dq, dr in BOARD_PLACEMENT[:n_boards]:
        boards.append([(rotate(c, turns)[0] + dq, rotate(c, turns)[1] + dr)
                       for c in BOARD_SHAPE])
    return boards


def random_board(rng):
    """A random board side (BOARD_SHAPE order) with the printed mix."""
    spots = []
    for _ in range(300):
        spots = rng.sample(BOARD_SHAPE, len(BOARD_BUILDINGS))
        # keep buildings apart, like on the printed boards
        if all(n not in spots for s in spots for n in neighbours(s)):
            break
    blds = BOARD_BUILDINGS[:]
    rng.shuffle(blds)
    terr = BOARD_TERRAIN[:]
    rng.shuffle(terr)
    return [blds.pop() if c in spots else terr.pop() for c in BOARD_SHAPE]


def random_layout(rng):
    return {"boards": [{"A": random_board(rng), "B": random_board(rng)}
                       for _ in range(N_BOARDS)]}


def load_layout(text):
    """Parse and check a boards.json text. Raises ValueError if it's wrong."""
    try:
        data = json.loads(text)
    except ValueError:
        raise ValueError("boards.json is not valid JSON.")
    boards = data.get("boards") if isinstance(data, dict) else None
    if not isinstance(boards, list) or len(boards) != N_BOARDS:
        raise ValueError("boards.json needs %d boards." % N_BOARDS)
    for i, sides in enumerate(boards):
        if not isinstance(sides, dict) or not sides:
            raise ValueError("Board %d has no sides." % (i + 1))
        for name, terr in sides.items():
            if name not in ("A", "B"):
                raise ValueError("Board %d: sides must be A or B." % (i + 1))
            if not isinstance(terr, list) or len(terr) != BOARD_SIZE:
                raise ValueError("Board %d%s needs %d spaces."
                                 % (i + 1, name, BOARD_SIZE))
            bad = [t for t in terr if t not in LAND_TYPES]
            if bad:
                raise ValueError("Board %d%s: unknown space '%s'."
                                 % (i + 1, name, bad[0]))
    return {"boards": boards}


def generate_land(n_players, rng, layout=None):
    """Lay out n of the 4 boards: which boards, in which places and on which
    side is random. (Always the same boards would let a computer player learn
    them by heart: with 2 players that gave only 4 landscapes, now 48.)"""
    if layout is None:
        layout = random_layout(rng)
    land = {}
    boards = rng.sample(layout["boards"], n_players)
    for cells, sides in zip(board_cells(n_players), boards):
        terr = sides[rng.choice(sorted(sides))]
        land.update(zip(cells, terr))
    return land


def fjord_cells():
    cells = [(q, r) for q, rs in sorted(FJORD_COLUMNS.items()) for r in rs]
    return cells, dict(FJORD_SITE_SPOTS)


# ---------------------------------------------------------------------------
# Game state
# ---------------------------------------------------------------------------

class Player:
    def __init__(self, idx, name, color, ai, sites):
        self.idx = idx
        self.name = name
        self.color = color
        self.ai = ai
        self.vikings_left = VIKINGS_PER_PLAYER
        self.shields = {"extra": True, "occupy": True, "double": True}
        cells, spots = fjord_cells()
        self.fjord_cells = cells
        # fjord content: cell -> item dict (or missing if empty)
        self.fjord = {}
        for s, c in spots.items():
            if s not in sites:          # construction sites switched off
                continue
            self.fjord[c] = {"kind": "site", "type": s, "need": sites[s],
                             "done": False}
        self.trophy = None           # index into TROPHIES
        self.castle_taken = {}       # castle cell key -> tiles taken
        self.wt_pairs = set()        # "a|b" keys of connected tower pairs

    def __deepcopy__(self, memo):
        """Fast copy: only what can change during a game is copied; the
        rest (like fjord_cells and the needs of ships and sites) is shared."""
        new = copy.copy(self)
        new.shields = dict(self.shields)
        new.fjord = {c: dict(it) for c, it in self.fjord.items()}
        new.castle_taken = dict(self.castle_taken)
        new.wt_pairs = set(self.wt_pairs)
        return new

    # --- fjord helpers ---
    def empty_cells(self):
        return [c for c in self.fjord_cells if c not in self.fjord]

    def count(self, kind_type):
        return sum(1 for it in self.fjord.values()
                   if it["kind"] in ("res", "bld") and it["type"] == kind_type)

    def adjacent_items(self, c):
        out = []
        for n in neighbours(c):
            it = self.fjord.get(n)
            if it and it["kind"] in ("res", "bld"):
                out.append(it["type"])
        return out

    def update_fills(self):
        """Flip any longship / construction site whose needs are now met.
        Returns a list of messages."""
        msgs = []
        for c, it in self.fjord.items():
            if it["kind"] == "ship" and not it["filled"]:
                if sub_multiset(it["need"], self.adjacent_items(c)):
                    it["filled"] = True
                    msgs.append("filled a longship (+%d %s)" %
                                (it["bonus"], it["cat"]))
            elif it["kind"] == "site" and not it["done"]:
                if sub_multiset(it["need"], self.adjacent_items(c)):
                    it["done"] = True
                    msgs.append("completed the %s (+%d VP)" %
                                (SITE_NAMES[it["type"]], SITE_VP[it["type"]]))
        return msgs

    def score(self):
        rows = {}
        total = 0
        for cat in SCORE_ORDER:
            bonus = sum(it["bonus"] for it in self.fjord.values()
                        if it["kind"] == "ship" and it["filled"]
                        and it["cat"] == cat)
            n = self.count(cat)
            v = (BASE_VALUE[cat] + bonus) * n
            rows[cat] = {"value": BASE_VALUE[cat] + bonus, "count": n,
                         "points": v}
            total += v
        sites = sum(SITE_VP[it["type"]] for it in self.fjord.values()
                    if it["kind"] == "site" and it["done"])
        trophy = TROPHIES[self.trophy][1] if self.trophy is not None else 0
        unfilled = sum(1 for it in self.fjord.values()
                       if it["kind"] == "ship" and not it["filled"])
        total += sites + trophy - 5 * unfilled
        return {"rows": rows, "sites": sites, "trophy": trophy,
                "unfilled": unfilled, "penalty": -5 * unfilled,
                "total": total}


class Game:
    def __init__(self, players, seed=None, layout=None, rules=None):
        """players: list of (name, is_ai) tuples, 2 to 4 of them.
        layout: board layouts from load_layout(), or None for random.
        rules: dict of FULL_RULES to switch some off (see RULE_STAGES)."""
        assert 2 <= len(players) <= 4
        self.rules = dict(FULL_RULES, **(rules or {}))
        self.seed = seed if seed is not None else random.randrange(10 ** 9)
        self.rng = random.Random(self.seed)
        n = len(players)
        self.land = generate_land(n, self.rng, layout)
        self.vikings = {}                     # land cell -> [player idx...]
        self.ocean = list(OCEAN_CELLS)
        bag = list(range(len(LONGSHIPS)))
        self.rng.shuffle(bag)
        self.bag = bag
        self.ocean_ships = [self.bag.pop() for _ in self.ocean]
        # tiles left on each building space of the gameboard
        self.stock = {c: BUILDING_STACK[t] for c, t in self.land.items()
                      if t in BUILDING_STACK}
        self.trophy_owner = [None] * len(TROPHIES)
        site_decks = {}
        for s, lst in SITES.items():
            deck = list(range(len(lst)))
            self.rng.shuffle(deck)
            site_decks[s] = deck
        self.players = []
        for i, (name, ai) in enumerate(players):
            sites = {s: SITES[s][site_decks[s][i]] for s in SITES}
            if not self.rules["sites"]:
                sites = {}
            p = Player(i, name, COLORS[i], ai, sites)
            if not self.rules["shields"]:
                p.shields = {k: False for k in p.shields}
            self.players.append(p)
        if not self.rules["longships"]:
            self.ocean_ships = [None] * len(self.ocean)
            self.bag = []
        self.current = 0
        self.log = []
        self.game_over = False
        self.new_turn()

    def __deepcopy__(self, memo):
        """Fast copy for the computer player, which tries moves on copies.
        The landscape, ocean spaces and random generator never change after
        setup, so they are shared instead of copied."""
        new = copy.copy(self)
        new.vikings = {c: list(v) for c, v in self.vikings.items()}
        new.bag = list(self.bag)
        new.ocean_ships = list(self.ocean_ships)
        new.stock = dict(self.stock)
        new.trophy_owner = list(self.trophy_owner)
        new.pending = [dict(it) for it in self.pending]
        new.log = list(self.log)
        new.players = [copy.deepcopy(p, memo) for p in self.players]
        return new

    # ------------------------------------------------------------------
    # turn bookkeeping
    # ------------------------------------------------------------------
    def new_turn(self):
        self.turn_no = getattr(self, "turn_no", 0) + 1   # 1 = the first turn of the game
        self.phase = "place"       # place -> tiles -> actions -> (ship)
        self.pending = []          # tiles waiting to go on the fjord
        self.placed_this_turn = 0
        self.took_ship = False
        self.held_ship = None      # ocean slot picked, waiting for a cell
        self.extra_active = False  # +Viking shield in use
        p = self.player()
        if p.vikings_left > 0 and not self.has_room_for_viking(p):
            # No legal space at all: this Viking is lost
            p.vikings_left -= 1
            self.say("%s has nowhere to place a Viking and loses one." % p.name)
            self.phase = "actions"

    def player(self):
        return self.players[self.current]

    def say(self, msg):
        self.log.append(msg)
        self.log = self.log[-60:]

    # ------------------------------------------------------------------
    # Viking placement
    # ------------------------------------------------------------------
    def is_anchor(self, c):
        """A space counts as next to 'a Viking or a Longship'."""
        for n in neighbours(c):
            if self.vikings.get(n):
                return True
            slot = OCEAN_SLOT.get(n)
            if slot is not None:
                if self.ocean_ships[slot] is not None or not self.rules["longships"]:
                    return True
        return False

    def legal_cells(self, occupy):
        """Resource spaces where a Viking may go. With occupy=True, only
        the occupied ones (those need the shield)."""
        out = []
        for c, t in self.land.items():
            if t not in TERRAIN_RESOURCE:
                continue
            occupied = bool(self.vikings.get(c))
            if occupied != occupy:
                continue
            if self.is_anchor(c):
                out.append(c)
        return out

    def has_room_for_viking(self, p):
        return bool(self.legal_cells(False) or
                    (p.shields["occupy"] and self.legal_cells(True)))

    def can_use_extra(self, p):
        return (self.phase == "actions" and not self.took_ship and
                p.shields["extra"] and p.vikings_left > 0 and
                self.placed_this_turn > 0 and self.has_room_for_viking(p))

    def place_viking(self, c, occupy=False, double=False):
        p = self.player()
        if self.phase != "place":
            raise ValueError("You can't place a Viking right now.")
        if p.vikings_left <= 0:
            raise ValueError("No Vikings left.")
        occupied = bool(self.vikings.get(c))
        if occupied and not occupy:
            raise ValueError("That space is taken.")
        if occupy and not (occupied and p.shields["occupy"]):
            raise ValueError("That shield can't be used here.")
        if double and not p.shields["double"]:
            raise ValueError("Double shield already used.")
        if c not in self.legal_cells(occupied):
            raise ValueError("A Viking must go next to a Viking or a Longship.")

        self.vikings.setdefault(c, []).append(p.idx)
        p.vikings_left -= 1
        self.placed_this_turn += 1
        if occupy:
            p.shields["occupy"] = False
        if double:
            p.shields["double"] = False
        res = TERRAIN_RESOURCE[self.land[c]]
        gained = [res] * (2 if double else 1)
        caps = self.captures(p, c)
        self.pending += [{"kind": "res", "type": r} for r in gained]
        # "from" remembers the stack, so an unused tile can go back on it
        self.pending += [{"kind": "bld", "type": self.land[a], "from": key(a)}
                         for a in caps]
        kinds = [self.land[a] for a in caps]
        parts = ["%d %s" % (gained.count(res), res)]
        for b in BUILDINGS:
            if kinds.count(b):
                parts.append("%d %s" % (kinds.count(b), b))
        extra = []
        if occupy:
            extra.append("stacked with the shield")
        if double:
            extra.append("double shield")
        self.say("%s placed a Viking%s: %s." % (
            p.name, " (" + ", ".join(extra) + ")" if extra else "",
            ", ".join(parts)))
        self.extra_active = False
        self._drop_if_full()
        self.phase = "tiles" if self.pending else "actions"

    def _drop_if_full(self):
        """You can't take tiles you have no room for. When there are more
        tiles than empty spaces, the player chooses which to keep by placing
        them; whatever is left once the fjord is full is lost."""
        p = self.player()
        if self.pending and not p.empty_cells():
            for it in self.pending:
                if it["kind"] == "bld":
                    self.stock[unkey(it["from"])] += 1
            self.say("%s's fjord is full: %d tile(s) lost." %
                     (p.name, len(self.pending)))
            self.pending = []

    def chain(self, p, start):
        seen = {start}
        todo = [start]
        while todo:
            c = todo.pop()
            for n in neighbours(c):
                if n not in seen and p.idx in self.vikings.get(n, []):
                    seen.add(n)
                    todo.append(n)
        return seen

    def tower_tiles(self, p, towers, c=None):
        """Watchtowers that pay 1 tile when a chain of p's Vikings (with the
        new Viking on c) touches `towers`.

        Towers p has linked before form a group. Every separate group this
        chain joins pays 1 tile: linking two new towers gives 2, and linking
        a new tower to a pair you already had gives 2 as well (1 for the new
        tower, 1 for the tower it connects to - the Ragnar example). Towers
        that were already linked pay nothing more. In a group, the tower next
        to the new Viking pays, otherwise the one with most tiles left."""
        def pair(a, b):
            return "|".join(sorted([key(a), key(b)]))
        groups = []
        left = sorted(towers)          # a fixed order: fastgame.py does the same
        while left:
            grp = [left.pop(0)]
            todo = list(grp)
            while todo:
                a = todo.pop()
                for b in [b for b in left if pair(a, b) in p.wt_pairs]:
                    left.remove(b)
                    todo.append(b)
                    grp.append(b)
            groups.append(grp)
        for a in towers:
            for b in towers:
                if a != b:
                    p.wt_pairs.add(pair(a, b))
        if len(groups) < 2:
            return []
        near = set(neighbours(c)) if c else set()
        return [max(g, key=lambda a: (a in near, self.stock.get(a, 0), a))
                for g in groups]

    def take_building(self, a, n):
        """Take up to n tiles from the stack on building space a."""
        got = min(n, self.stock.get(a, 0))
        if got:
            self.stock[a] -= got
        return [a] * got

    def supply(self):
        """Tiles left to claim on the gameboard, per building type."""
        tot = {b: 0 for b in BUILDINGS}
        for a, n in self.stock.items():
            tot[self.land[a]] += n
        return tot

    def captures(self, p, c):
        """Building spaces captured by player p placing a Viking on c:
        one entry per tile taken."""
        got = []
        if not self.rules["buildings"]:
            return got
        # Houses: each House next to the new Viking
        for n in neighbours(c):
            if self.land.get(n) == "house":
                got += self.take_building(n, 1)
        ch = self.chain(p, c)
        adj = set()
        for cc in ch:
            for n in neighbours(cc):
                if self.land.get(n) in ("watchtower", "castle"):
                    adj.add(n)
        # (sorted, so the tiles always come in the same order)
        adj = sorted(adj)
        # Watchtowers: 1 tile per group of towers this chain links together
        towers = [a for a in adj if self.land[a] == "watchtower"]
        for a in self.tower_tiles(p, towers, c):
            got += self.take_building(a, 1)
        # Castles: 1 tile per 4 Vikings in the chain (max 3 per castle)
        level = min(3, len(ch) // 4)
        for a in adj:
            if self.land[a] == "castle":
                k = key(a)
                have = p.castle_taken.get(k, 0)
                if level > have:
                    p.castle_taken[k] = level
                    got += self.take_building(a, level - have)
        return got

    # ------------------------------------------------------------------
    # Fjord tiles
    # ------------------------------------------------------------------
    def place_tile(self, idx, c):
        p = self.player()
        if self.phase != "tiles":
            raise ValueError("No tile to place.")
        if c not in p.fjord_cells or c in p.fjord:
            raise ValueError("Pick an empty fjord space.")
        it = self.pending.pop(idx)
        p.fjord[c] = {"kind": it["kind"], "type": it["type"]}
        for m in p.update_fills():
            self.say("%s %s." % (p.name, m))
        self._drop_if_full()
        if not self.pending:
            self.phase = "actions"

    def helpful_cells(self, p, item_type):
        """Empty fjord cells next to a longship/site still needing this."""
        out = []
        for c in p.empty_cells():
            for n in neighbours(c):
                it = p.fjord.get(n)
                if not it or it["kind"] not in ("ship", "site"):
                    continue
                if it.get("filled") or it.get("done"):
                    continue
                have = p.adjacent_items(n)
                missing = list(it["need"])
                for h in have:
                    if h in missing:
                        missing.remove(h)
                if item_type in missing:
                    out.append(c)
                    break
        return out

    # ------------------------------------------------------------------
    # Optional actions
    # ------------------------------------------------------------------
    def use_extra_shield(self):
        p = self.player()
        if self.phase != "actions" or self.took_ship:
            raise ValueError("Use this right after placing a Viking.")
        if not p.shields["extra"] or p.vikings_left <= 0:
            raise ValueError("Can't place another Viking.")
        if self.placed_this_turn == 0:
            raise ValueError("Place your first Viking first.")
        if not self.has_room_for_viking(p):
            raise ValueError("There is no space left for a second Viking.")
        p.shields["extra"] = False
        self.extra_active = True
        self.phase = "place"
        self.say("%s uses the shield to place a second Viking." % p.name)

    def pick_ship(self, slot):
        if self.phase != "actions" or self.took_ship:
            raise ValueError("You can take only 1 longship per turn.")
        if self.ocean_ships[slot] is None:
            raise ValueError("No longship there.")
        if not self.player().empty_cells():
            raise ValueError("Your fjord is full.")
        self.held_ship = slot
        self.phase = "ship"

    def cancel_ship(self):
        if self.phase == "ship":
            self.held_ship = None
            self.phase = "actions"

    def place_ship(self, c):
        p = self.player()
        if self.phase != "ship":
            raise ValueError("Pick a longship first.")
        if c not in p.fjord_cells or c in p.fjord:
            raise ValueError("Pick an empty fjord space.")
        sid = self.ocean_ships[self.held_ship]
        need, cat, bonus = LONGSHIPS[sid]
        p.fjord[c] = {"kind": "ship", "id": sid, "need": list(need),
                      "cat": cat, "bonus": bonus, "filled": False}
        self.ocean_ships[self.held_ship] = self.bag.pop() if self.bag else None
        self.held_ship = None
        self.took_ship = True
        self.phase = "actions"
        self.say("%s took a longship (%s -> +%d %s)." %
                 (p.name, "/".join(need), bonus, cat))
        for m in p.update_fills():
            self.say("%s %s." % (p.name, m))

    def axes(self, p):
        return p.count("axe")

    def claimable_trophies(self, p):
        if (p.trophy is not None or self.phase != "actions"
                or not self.rules["trophies"]):
            return []
        a = self.axes(p)
        return [i for i, (need, vp) in enumerate(TROPHIES)
                if self.trophy_owner[i] is None and a >= need]

    def claim_trophy(self, i):
        p = self.player()
        if i not in self.claimable_trophies(p):
            raise ValueError("You can't claim that trophy.")
        self.trophy_owner[i] = p.idx
        p.trophy = i
        self.say("%s claimed the %d-axe trophy (%d VP)." %
                 (p.name, TROPHIES[i][0], TROPHIES[i][1]))

    def end_turn(self):
        if self.phase not in ("actions",):
            raise ValueError("Finish placing first.")
        n = len(self.players)
        for step in range(1, n + 1):
            nxt = (self.current + step) % n
            if self.players[nxt].vikings_left > 0:
                self.current = nxt
                self.new_turn()
                return
        self.game_over = True
        self.phase = "over"
        w = self.winners()
        self.say("Game over! %s wins." % " & ".join(
            self.players[i].name for i in w))

    def winners(self):
        def rank(p):
            s = p.score()["total"]
            t = TROPHIES[p.trophy][1] if p.trophy is not None else -1
            return (s, t)
        best = max(rank(p) for p in self.players)
        return [p.idx for p in self.players if rank(p) == best]

    # ------------------------------------------------------------------
    # Saving the whole game (the web page sends it to server.py, which
    # lets the neural network play a turn on it and sends it back)
    # ------------------------------------------------------------------
    def save_state(self):
        """Everything needed to rebuild this game exactly, as plain JSON data."""
        return {
            "rules": self.rules, "seed": self.seed,
            "land": {key(c): t for c, t in self.land.items()},
            "vikings": {key(c): v for c, v in self.vikings.items()},
            "bag": self.bag, "ocean_ships": self.ocean_ships,
            "stock": {key(c): n for c, n in self.stock.items()},
            "trophy_owner": self.trophy_owner, "current": self.current,
            "turn_no": self.turn_no, "log": self.log, "game_over": self.game_over, "phase": self.phase,
            "pending": self.pending, "placed_this_turn": self.placed_this_turn,
            "took_ship": self.took_ship, "held_ship": self.held_ship,
            "extra_active": self.extra_active,
            "players": [{
                "idx": p.idx, "name": p.name, "color": p.color, "ai": p.ai,
                "nn": getattr(p, "nn", False), "vikings_left": p.vikings_left,
                "shields": p.shields, "fjord": {key(c): it for c, it in p.fjord.items()},
                "trophy": p.trophy, "castle_taken": p.castle_taken,
                "wt_pairs": sorted(p.wt_pairs)} for p in self.players],
        }

    @classmethod
    def load_state(cls, d):
        """The game saved by save_state()."""
        g = cls.__new__(cls)
        g.rules = dict(d["rules"])
        g.seed = d["seed"]
        g.rng = random.Random(g.seed)          # only used while setting up
        g.land = {unkey(k): t for k, t in d["land"].items()}
        g.vikings = {unkey(k): list(v) for k, v in d["vikings"].items()}
        g.ocean = list(OCEAN_CELLS)
        g.bag = list(d["bag"])
        g.ocean_ships = list(d["ocean_ships"])
        g.stock = {unkey(k): n for k, n in d["stock"].items()}
        g.trophy_owner = list(d["trophy_owner"])
        g.turn_no = d.get("turn_no", 0)
        for name in ("current", "game_over", "phase", "placed_this_turn",
                     "took_ship", "held_ship", "extra_active"):
            setattr(g, name, d[name])
        g.log = list(d["log"])
        g.pending = [dict(it) for it in d["pending"]]
        g.players = []
        cells, _ = fjord_cells()
        for pd in d["players"]:
            p = Player.__new__(Player)
            for name in ("idx", "name", "color", "ai", "nn", "vikings_left", "trophy"):
                setattr(p, name, pd[name])
            p.shields = dict(pd["shields"])
            p.fjord_cells = cells
            p.fjord = {unkey(k): dict(it) for k, it in pd["fjord"].items()}
            p.castle_taken = dict(pd["castle_taken"])
            p.wt_pairs = set(pd["wt_pairs"])
            g.players.append(p)
        return g

    # ------------------------------------------------------------------
    # Serialisation for the browser
    # ------------------------------------------------------------------
    def to_dict(self):
        p = self.player()
        state = {
            "seed": self.seed,
            "current": self.current,
            "turn_no": self.turn_no,
            "phase": self.phase,
            "game_over": self.game_over,
            "winners": self.winners() if self.game_over else [],
            "land": [{"q": c[0], "r": c[1], "t": t,
                      "v": self.vikings.get(c, []),
                      "left": self.stock.get(c)}
                     for c, t in self.land.items()],
            "ocean": [{"q": c[0], "r": c[1], "slot": i,
                       "ship": self._ship_dict(self.ocean_ships[i])}
                      for i, c in enumerate(self.ocean)],
            "bag": len(self.bag),
            "supply": self.supply(),
            "trophies": [{"axes": a, "vp": v, "owner": self.trophy_owner[i]}
                         for i, (a, v) in enumerate(TROPHIES)],
            "pending": self.pending,
            "took_ship": self.took_ship,
            "held_ship": self.held_ship,
            "extra_active": self.extra_active,
            "can_extra": self.can_use_extra(p),
            "room": len(p.empty_cells()),
            "log": self.log[-14:],
            "players": [],
        }
        if self.phase == "place":
            state["legal"] = [key(c) for c in self.legal_cells(False)]
            state["legal_occupy"] = ([key(c) for c in self.legal_cells(True)]
                                     if p.shields["occupy"] else [])
        else:
            state["legal"] = []
            state["legal_occupy"] = []
        state["helpful"] = {}
        if self.phase == "tiles":
            for t in set(it["type"] for it in self.pending):
                state["helpful"][t] = [key(c) for c in self.helpful_cells(p, t)]
        state["claimable"] = self.claimable_trophies(p)
        for pl in self.players:
            fj = []
            for c in pl.fjord_cells:
                it = pl.fjord.get(c)
                d = {"q": c[0], "r": c[1], "item": it}
                if it and it["kind"] in ("ship", "site"):
                    d["have"] = pl.adjacent_items(c)
                fj.append(d)
            state["players"].append({
                "name": pl.name, "color": pl.color, "ai": pl.ai,
                "nn": getattr(pl, "nn", False),
                "vikings_left": pl.vikings_left, "shields": pl.shields,
                "trophy": pl.trophy, "axes": self.axes(pl),
                "fjord": fj, "score": pl.score(),
            })
        return state

    def _ship_dict(self, sid):
        if sid is None:
            return None
        need, cat, bonus = LONGSHIPS[sid]
        return {"id": sid, "need": list(need), "cat": cat, "bonus": bonus}


# ---------------------------------------------------------------------------
# Computer opponent
# ---------------------------------------------------------------------------

# The computer player's judgement: every number it uses to guess how good a
# position is. The values were found by tune.py, which plays thousands of
# games in the arena and keeps changes that win more often.
BOT_WEIGHTS = {
    # unfilled longship: chance it still gets filled
    "ship_base": 0.901,          # chance with nothing missing
    "ship_per_missing": 0.22,   # minus this per missing resource
    "ship_late": 0.25,          # minus this when Vikings run short
    "ship_min": 0.1,           # lowest chance while still possible
    "ship_hopeless": 1.3,       # hopeless if missing > Vikings left * this
    "ship_hopeless_chance": 0.05,
    # expected extra tiles of a longship's category per Viking left
    "ship_future_res": 0.3,
    "ship_future_bld": 0.15,
    # unfinished construction site: VP * (share done ** power) * weight
    "site_power": 2.0,
    "site_early": 0.938,          # weight while more than site_late_left Vikings
    "site_late": 0.514,
    "site_late_left": 2,
    # trophy: share of its VP counted before claiming it
    "trophy_ready": 0.9,        # enough axes already
    "trophy_progress": 0.5,     # still collecting (times axes / needed)
    # each unused shield is worth this many points
    "shield_early": 1.915,
    "shield_late": 0.3,
    "shield_late_left": 5,
    # points per Viking of castle chain on the way to the next 4
    "chain": 0.8,
    # fjord tile placement
    "tile_need": 10.0,          # next to a ship/site that needs this tile
    "tile_block": 3.0,          # minus: next to one that doesn't need it
    "tile_open": 0.262,           # per empty neighbour (room for ships)
    # longship placement in the fjord
    "shipcell_match": 5.0,      # per needed resource already next to it
    "shipcell_room": 1.0,       # per empty space it still needs
    # decisions
    "extra_gain": 3.0,          # 2nd Viking shield: needed gain in points
    "extra_force_left": 3,      # ...or use it with this few Vikings left
    "take_ship_margin": 1.0,    # take a longship if it adds this much
    "trophy_wait_left": 0,      # settle for a lower trophy with this few left
    "noise": 0.3,               # random tie-breaking between placements
}


class Bot:
    """A greedy player: it tries every move one step ahead and keeps the
    one that raises its estimated final score the most. How it judges a
    position is set by BOT_WEIGHTS (or the weights you pass in)."""

    def __init__(self, rng, weights=None):
        self.rng = rng
        self.w = dict(BOT_WEIGHTS)
        if weights:
            self.w.update(weights)

    # --- how good is this position for player p? ---
    def evaluate(self, g, p):
        w = self.w
        s = p.score()
        val = s["total"]
        left = p.vikings_left
        # unfilled longships: count them as "probably filled" when there
        # is still time, instead of the flat -5 penalty
        for c, it in p.fjord.items():
            if it["kind"] == "ship" and not it["filled"]:
                have = p.adjacent_items(c)
                m = matched_count(it["need"], have)
                missing = 3 - m
                room = sum(1 for n in neighbours(c)
                           if n in p.fjord_cells and n not in p.fjord)
                if missing > left * w["ship_hopeless"] or room < missing:
                    chance = w["ship_hopeless_chance"]
                else:
                    late = w["ship_late"] if left < 2 * missing else 0
                    chance = max(w["ship_min"], w["ship_base"] -
                                 w["ship_per_missing"] * missing - late)
                chance = min(1.0, chance)
                per = (w["ship_future_res"] if it["cat"] in RESOURCES
                       else w["ship_future_bld"])
                future = p.count(it["cat"]) + left * per
                gain = it["bonus"] * max(future, 1)
                val += 5 + chance * gain - (1 - chance) * 5
        site_w = w["site_early"] if left > w["site_late_left"] else w["site_late"]
        for c, it in p.fjord.items():
            if it["kind"] == "site" and not it["done"]:
                m = matched_count(it["need"], p.adjacent_items(c))
                frac = m / float(len(it["need"]))
                val += SITE_VP[it["type"]] * frac ** w["site_power"] * site_w
        # trophy progress
        if p.trophy is None:
            a = g.axes(p)
            best = 0
            for i, (need, vp) in enumerate(TROPHIES):
                if g.trophy_owner[i] is None:
                    if a >= need:
                        best = max(best, vp * w["trophy_ready"])
                    elif need - a <= left:
                        best = max(best, vp * w["trophy_progress"] * (a / float(need)))
            val += best
        # shields still unused have some value while the game runs
        sw = w["shield_early"] if left > w["shield_late_left"] else w["shield_late"]
        val += sw * sum(1 for v in p.shields.values() if v)
        # longer chains mean more castle tiles later
        if left > 0:
            best_chain = 0
            done = set()
            for c, owners in g.vikings.items():
                if p.idx in owners and c not in done:
                    ch = g.chain(p, c)
                    done |= ch
                    near_castle = any(g.land.get(n) == "castle"
                                      for cc in ch for n in neighbours(cc))
                    if near_castle:
                        best_chain = max(best_chain, len(ch) % 4)
            val += best_chain * w["chain"]
        return val

    # --- fjord placement of one tile ---
    def best_tile_cell(self, g, p, item):
        cells = p.empty_cells()
        if not cells:
            return None
        best, best_v = None, None
        for c in cells:
            v = 0.0
            for n in neighbours(c):
                it = p.fjord.get(n)
                if it and it["kind"] in ("ship", "site") and \
                        not it.get("filled") and not it.get("done"):
                    have = p.adjacent_items(n)
                    missing = list(it["need"])
                    for h in have:
                        if h in missing:
                            missing.remove(h)
                    if item["type"] in missing:
                        v += self.w["tile_need"]
                    else:
                        v -= self.w["tile_block"]   # don't block a slot it still needs
                elif n in p.fjord_cells and n not in p.fjord:
                    v += self.w["tile_open"]        # keep room for new longships
            v += self.rng.random() * 0.1
            if best_v is None or v > best_v:
                best, best_v = c, v
        return best

    KEEP_ORDER = {"castle": 5, "watchtower": 3, "house": 2, "gold": 2,
                  "axe": 1.5, "sheep": 1, "wood": 1}

    def place_pending(self, g):
        p = g.player()
        while g.phase == "tiles":
            i = 0
            if len(g.pending) > len(p.empty_cells()):
                # not everything fits: keep the most valuable tiles
                i = max(range(len(g.pending)),
                        key=lambda j: self.KEEP_ORDER[g.pending[j]["type"]])
            c = self.best_tile_cell(g, p, g.pending[i])
            g.place_tile(i, c)

    def best_ship_cell(self, g, p, need):
        best, best_v = None, None
        for c in p.empty_cells():
            have = p.adjacent_items(c)
            m = matched_count(need, have)
            room = sum(1 for n in neighbours(c)
                       if n in p.fjord_cells and n not in p.fjord)
            v = (m * self.w["shipcell_match"] +
                 min(room, 3 - m) * self.w["shipcell_room"] +
                 self.rng.random() * 0.1)
            if room < 3 - m:
                v -= 20
            if best_v is None or v > best_v:
                best, best_v = c, v
        return best

    # --- choosing a Viking placement ---
    def placement_options(self, g):
        p = g.player()
        opts = [(c, False, False) for c in g.legal_cells(False)]
        if p.shields["double"]:
            opts += [(c, False, True) for c in g.legal_cells(False)]
        if p.shields["occupy"]:
            opts += [(c, True, False) for c in g.legal_cells(True)]
        return opts

    def simulate_place(self, g, opt):
        g2 = copy.deepcopy(g)
        c, occ, dbl = opt
        g2.place_viking(c, occ, dbl)
        self.place_pending(g2)
        return g2

    def choose_placement(self, g):
        p = g.player()
        opts = self.placement_options(g)
        if not opts:
            return None, None
        best, best_v = None, None
        for opt in opts:
            g2 = self.simulate_place(g, opt)
            v = self.evaluate(g2, g2.players[p.idx]) + self.rng.random() * self.w["noise"]
            if best_v is None or v > best_v:
                best, best_v = opt, v
        return best, best_v

    def play_turn(self, g):
        """Play a whole turn for the current player."""
        p = g.player()
        if g.phase == "place":
            opt, _ = self.choose_placement(g)
            if opt is not None:
                g.place_viking(*opt)
                self.place_pending(g)
        # Second Viking with the shield?
        if g.can_use_extra(p):
            base = self.evaluate(g, p)
            g2 = copy.deepcopy(g)
            g2.use_extra_shield()
            opt, v = self.choose_placement(g2)
            if opt is not None and (v - base > self.w["extra_gain"] or
                                    p.vikings_left <= self.w["extra_force_left"]):
                g.use_extra_shield()
                g.place_viking(*opt)
                self.place_pending(g)
        # Longship?
        if g.phase == "actions" and not g.took_ship and p.empty_cells():
            base = self.evaluate(g, p)
            best, best_v = None, base + self.w["take_ship_margin"]
            for slot, sid in enumerate(g.ocean_ships):
                if sid is None:
                    continue
                g2 = copy.deepcopy(g)
                g2.pick_ship(slot)
                c = self.best_ship_cell(g2, g2.player(), list(LONGSHIPS[sid][0]))
                if c is None:
                    continue
                g2.place_ship(c)
                v = self.evaluate(g2, g2.players[p.idx])
                if v > best_v:
                    best, best_v = (slot, c), v
            if best:
                g.pick_ship(best[0])
                g.place_ship(best[1])
        # Trophy?
        cl = g.claimable_trophies(p)
        if cl:
            top = max(cl)
            remaining = [i for i in range(len(TROPHIES))
                         if g.trophy_owner[i] is None]
            if top == max(remaining) or p.vikings_left <= self.w["trophy_wait_left"]:
                g.claim_trophy(top)
        if g.phase == "actions":
            g.end_turn()


# ---------------------------------------------------------------------------
# JSON interface used by the web page
# ---------------------------------------------------------------------------

class Api:
    def __init__(self):
        self.game = None
        self.layout = None
        self.bot = Bot(random.Random())

    def set_layout(self, text):
        """Use the board layouts from boards.json. Returns "" or an error."""
        try:
            self.layout = load_layout(text)
            return ""
        except ValueError as e:
            self.layout = None
            return str(e)

    def _state(self, error=None):
        d = self.game.to_dict() if self.game else {}
        if error:
            d["error"] = error
        return json.dumps(d)

    def _do(self, fn, *args):
        try:
            fn(*args)
            return self._state()
        except ValueError as e:
            return self._state(str(e))

    def new_game(self, players_json, seed=-1):
        seats = json.loads(players_json)
        players = [(p["name"], bool(p["ai"])) for p in seats]
        self.game = Game(players, None if seed < 0 else seed, self.layout)
        for p, s in zip(self.game.players, seats):
            p.nn = bool(s.get("nn")) and p.ai     # played by the neural network
        self.game.say("A new game begins. %s goes first." %
                      self.game.player().name)
        return self._state()

    def state(self):
        return self._state()

    def place_viking(self, q, r, occupy, double):
        return self._do(self.game.place_viking, (q, r), bool(occupy), bool(double))

    def place_tile(self, idx, q, r):
        return self._do(self.game.place_tile, idx, (q, r))

    def use_extra(self):
        return self._do(self.game.use_extra_shield)

    def pick_ship(self, slot):
        return self._do(self.game.pick_ship, slot)

    def cancel_ship(self):
        return self._do(self.game.cancel_ship)

    def place_ship(self, q, r):
        return self._do(self.game.place_ship, (q, r))

    def claim_trophy(self, i):
        return self._do(self.game.claim_trophy, i)

    def end_turn(self):
        return self._do(self.game.end_turn)

    def save_state(self):
        return json.dumps(self.game.save_state())

    def load_state(self, text):
        self.game = Game.load_state(json.loads(text))
        return self._state()

    # The network player INSIDE the browser (no server): nn_bot.py and the
    # network file must be next to this file (the web page puts them there).
    def nn_turn(self, model):
        """The network plays the current player's turn."""
        import nn_bot
        g = self.game
        if not g.game_over and g.player().ai:
            nn_bot.NNBot(random.Random(), model).play_turn(g)
        return self._state()

    def nn_view(self, model):
        """What the network thinks, from its own seat (see nn_bot.thoughts)."""
        import nn_bot
        seat = next((p.idx for p in self.game.players if getattr(p, "nn", False)), None)
        if seat is None:
            return json.dumps({"error": "No network player in this game."})
        out = nn_bot.thoughts(nn_bot.load_net(model), self.game, seat)
        out["model"] = model
        return json.dumps(out)

    def nn_review(self, model, before, after, search=True):
        """The coach's judgement of one finished turn (review.py): `before`
        and `after` are the game (save_state JSON) at its start and end."""
        import review
        return json.dumps(review.review_turn(model, json.loads(before), json.loads(after),
                                             search=bool(search)))

    def ai_turn(self):
        g = self.game
        if g.game_over or not g.player().ai:
            return self._state()
        self.bot.play_turn(g)
        return self._state()


api = Api()


def read_layout_file(path="boards.json"):
    """boards.json next to this file, or None if there is none."""
    import os
    full = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    if not os.path.exists(full):
        return None
    with open(full, encoding="utf-8") as f:
        return load_layout(f.read())


def dump_layout(layout):
    """boards.json text with one line per board side, easy to read."""
    out = ['{', '  "boards": [']
    for i, sides in enumerate(layout["boards"]):
        out.append('    {')
        names = sorted(sides)
        for j, s in enumerate(names):
            comma = "," if j < len(names) - 1 else ""
            out.append('      "%s": %s%s' % (s, json.dumps(sides[s]), comma))
        out.append('    }' + ("," if i < len(layout["boards"]) - 1 else ""))
    out += ['  ]', '}', '']
    return "\n".join(out)


def simulate(n=3, seed=1, verbose=True, layout=None):
    g = Game([("Bot %d" % (i + 1), True) for i in range(n)], seed, layout)
    bot = Bot(random.Random(seed))
    turns = 0
    while not g.game_over:
        bot.play_turn(g)
        turns += 1
        assert turns < 400
    if verbose:
        for line in g.log[-10:]:
            print(line)
        for p in g.players:
            s = p.score()
            print("%-6s %4d  (sites %d, trophy %d, unfilled ships %d)" %
                  (p.name, s["total"], s["sites"], s["trophy"], s["unfilled"]))
    return g


# Only from a terminal: the browser also runs this file as "__main__",
# but without a __file__.
if __name__ == "__main__" and "__file__" in globals():
    simulate(layout=read_layout_file())
