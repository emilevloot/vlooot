"""
Local web server for Looot. Serves the game and the board editor at
http://localhost:8000, lets editor.html save boards.json, and plays the
turns of the neural-network player (2-player games).

    python server.py

The game itself runs in the browser. For a turn of the "Network" player the
page sends the whole game (Game.save_state) to /ai-turn; the network (see
nn_bot.py, needs numpy and numba) plays the turn here and the new game goes
back to the page.
"""

import http.server
import json
import os
import random
import threading

import looot

ROOT = os.path.dirname(os.path.abspath(__file__))
PORT = 8000
NN_MODEL = "c3_full_r8"          # models/<NN_MODEL>.npz: the network that plays

_nn_lock = threading.Lock()      # one network turn at a time (the net keeps a cache)


def nn_turn(state):
    """Let the network play the current player's turn of a saved game."""
    import nn_bot                # numpy + numba: only loaded when needed
    g = looot.Game.load_state(state)
    if len(g.players) != 2:
        raise ValueError("The neural network only plays 2-player games.")
    if not g.game_over and g.player().ai:
        with _nn_lock:
            nn_bot.NNBot(random.Random(), NN_MODEL).play_turn(g)
    return g.save_state()


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
        elif self.path == "/ai-turn":
            self.ai_turn()
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

    def ai_turn(self):
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
