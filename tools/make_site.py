"""
Build site/: everything needed to play Looot on a web site without a server
(GitHub Pages), the network player included.

    python tools/make_site.py

The game runs in the visitor's browser (Python via Pyodide, py/). The network
player runs there too: numpy comes from Pyodide's own site, the network file
from models/. Only what the page needs is copied; site/ is its own git
repository, pushed to a PUBLIC GitHub repository (the code of this one, the
training and its data stay private).
"""

import os
import sys

# This script lives in tools/; the modules and files it uses are in the
# project folder above it.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import re
import shutil

HERE = ROOT
SITE = os.path.join(HERE, "site")
FILES = ["index.html", "editor.html", "looot.py", "nn_encode.py", "nn_bot.py", "review.py", "endgame.py", "boards.json"]


def web_model():
    """The network the page plays with (WEB_MODEL in index.html)."""
    m = re.search(r"const WEB_MODEL = '([^']+)'", open(os.path.join(HERE, "index.html"),
                                                          encoding="utf-8").read())
    return m.group(1)


def main():
    os.makedirs(SITE, exist_ok=True)
    # clear what a previous build copied (but keep site/.git)
    for name in os.listdir(SITE):
        if name == ".git":
            continue
        p = os.path.join(SITE, name)
        shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
    for f in FILES:
        shutil.copy2(os.path.join(HERE, f), SITE)
    shutil.copytree(os.path.join(HERE, "py"), os.path.join(SITE, "py"))
    model = web_model()
    os.makedirs(os.path.join(SITE, "models"))
    shutil.copy2(os.path.join(HERE, "models", model + ".npz"), os.path.join(SITE, "models"))
    # GitHub Pages: serve every file as it is (no Jekyll processing)
    open(os.path.join(SITE, ".nojekyll"), "w").close()
    with open(os.path.join(SITE, "README.md"), "w", encoding="utf-8") as f:
        f.write("# Looot\n\nPlay the Viking board game Looot in your browser: "
                "open index.html (GitHub Pages).\n\nThe computer players: *Computer* "
                "(a greedy player) and *Network* (a neural network trained by self-play, "
                "network `%s`), both running in the browser.\n" % model)
    size = sum(os.path.getsize(os.path.join(d, x)) for d, _, fs in os.walk(SITE)
               if ".git" not in d for x in fs)
    print("site/ built: %.1f MB, network %s" % (size / 1e6, model))


if __name__ == "__main__":
    main()
