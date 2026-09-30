"""
Local web server for Looot. Serves the game and the board editor at
http://localhost:8000, lets editor.html save boards.json, and plays the
turns of the neural-network player (2-player games).

    python server.py

The game itself runs in the browser. For a turn of the "Network" player the
page sends the whole game (Game.save_state) to /ai-turn; the network (see
nn_bot.py, needs numpy and numba) plays the turn here and the new game goes
back to the page. /ai-view answers what the network thinks of a game (win
chance, and for the three-part network the value it gives every item and
how well each longship fits) - the page shows that next to the board.
/ai-review judges a finished turn like a chess trainer (review.py): the
points lost against the network's best move, a label and the best move.
"""

import glob
import http.server
import json
import os
import random
import threading

import looot

ROOT = os.path.dirname(os.path.abspath(__file__))
PORT = 8000
_nn_lock = threading.Lock()      # one network turn at a time (the net keeps a cache)
_models = {}                     # 2 -> the 2-player network, "mp" -> the 3-4-player one


def pick_model(multi=False):
    """The network that plays: of all curriculum runs
    (results/curriculum_<prefix>_results.json), the best full-game network that
    still loads with the current code - a 2-player network, or with
    multi=True a network for 3-4 players (made from a 2-player network by
    transfer_mp.py and trained further). Networks with final tests are
    ranked by their win rate against greedy players there; the three-part
    network wins ties, as the page can show what it thinks."""
    import nn_bot
    found = []
    for f in glob.glob(os.path.join(ROOT, "results", "curriculum_*_results.json")):
        res = json.load(open(f))
        best = res.get("stages", {}).get("full", {}).get("best")
        if not best:
            continue
        try:
            net = nn_bot.load_net(best)
        except (OSError, ValueError, KeyError):
            continue
        if hasattr(net.E, "PRESENT_COLS") != multi:
            continue
        test = res.get("final", {}).get("%s vs greedy" % best)
        found.append(((test is not None, test["win"] if test else 0,
                       isinstance(net, nn_bot.ThreeNet)), best))
    if not found:
        raise ValueError("No trained %s network fits the current code yet."
                         % ("3-4-player" if multi else "2-player"))
    return max(found)[1]


def model(n_players):
    key = 2 if n_players == 2 else "mp"
    if key not in _models:
        _models[key] = pick_model(key == "mp")
    return _models[key]


def nn_turn(state):
    """Let the network play the current player's turn of a saved game."""
    import nn_bot                # numpy + numba: only loaded when needed
    g = looot.Game.load_state(state)
    if not g.game_over and g.player().ai:
        with _nn_lock:
            # its last 2 turns worked out exactly (endgame.py)
            nn_bot.NNBot(random.Random(), model(len(g.players)), endgame="wide").play_turn(g)
    return g.save_state()


def nn_review(req):
    """The coach's judgement of one finished turn (see review.py)."""
    import review
    n = len(req["before"]["players"])
    with _nn_lock:
        return review.review_turn(model(n), req["before"], req["after"],
                                  search=bool(req.get("search", True)))


def nn_view(state):
    """What the network thinks of a saved game, seen from its own seat."""
    import nn_bot
    g = looot.Game.load_state(state)
    seat = next((p.idx for p in g.players if getattr(p, "nn", False)), None)
    if seat is None:
        raise ValueError("No network player in this game.")
    name = model(len(g.players))
    with _nn_lock:
        out = nn_bot.thoughts(nn_bot.load_net(name), g, seat)
    out["model"] = name
    return out


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT, **kwargs)

    def end_headers(self):
        # always serve the latest looot.py / boards.json
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def reply(self, code, data):
        body = json.dumps(data).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_body(self, limit):
        size = int(self.headers.get("Content-Length") or 0)
        if size <= 0 or size > limit:
            return None
        return self.rfile.read(size).decode("utf-8")

    def do_POST(self):
        if self.path == "/save-boards":
            self.save_boards()
        elif self.path in ("/ai-turn", "/ai-view", "/ai-review"):
            self.ai_turn(view=self.path[4:])
        else:
            self.reply(404, {"error": "Unknown address."})

    def save_boards(self):
        text = self.read_body(200000)
        if text is None:
            self.reply(400, {"error": "Nothing to save."})
            return
        try:
            layout = looot.load_layout(text)
        except ValueError as e:
            self.reply(400, {"error": str(e)})
            return
        with open(os.path.join(ROOT, "boards.json"), "w", encoding="utf-8") as f:
            f.write(looot.dump_layout(layout))
        self.reply(200, {"ok": True})

    def ai_turn(self, view="turn"):
        # view: "turn" (the network plays), "view" (what it thinks) or
        # "review" (the coach judges a finished turn)
        # Only our own page sends this header. A request with a custom header
        # from another web site would first need a permission check
        # ("preflight") that this server never gives, so other sites can't
        # use this address.
        if self.headers.get("X-Looot") != "1":
            self.reply(403, {"error": "Not allowed."})
            return
        text = self.read_body(2000000)
        if text is None:
            self.reply(400, {"error": "No game sent."})
            return
        try:
            if view == "view":
                self.reply(200, nn_view(json.loads(text)))
                return
            if view == "review":
                self.reply(200, nn_review(json.loads(text)))
                return
            state = nn_turn(json.loads(text))
        except (ValueError, KeyError, TypeError) as e:
            self.reply(400, {"error": str(e)})
            return
        except Exception as e:           # numpy/numba missing, no model, ...
            self.reply(500, {"error": "%s: %s" % (type(e).__name__, e)})
            return
        # the game as JSON text inside the answer, so the page can hand it to
        # Python unchanged (the order of the spaces stays the same)
        self.reply(200, {"state": json.dumps(state)})


if __name__ == "__main__":
    server = http.server.ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print("Looot is running at http://localhost:%d  (close this window to stop)" % PORT)
    server.serve_forever()
