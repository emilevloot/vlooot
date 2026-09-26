"""
Local web server for Looot. Serves the game and the board editor at
http://localhost:8000 and lets editor.html save boards.json.

    python server.py
"""

import http.server
import json
import os

import looot

ROOT = os.path.dirname(os.path.abspath(__file__))
PORT = 8000


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

    def do_POST(self):
        if self.path != "/save-boards":
            self.reply(404, {"error": "Unknown address."})
            return
        size = int(self.headers.get("Content-Length") or 0)
        if size <= 0 or size > 200000:
            self.reply(400, {"error": "Nothing to save."})
            return
        try:
            layout = looot.load_layout(self.rfile.read(size).decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as e:
            self.reply(400, {"error": str(e)})
            return
        with open(os.path.join(ROOT, "boards.json"), "w", encoding="utf-8") as f:
            f.write(looot.dump_layout(layout))
        self.reply(200, {"ok": True})


if __name__ == "__main__":
    server = http.server.ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print("Looot is running at http://localhost:%d  (close this window to stop)" % PORT)
    server.serve_forever()
