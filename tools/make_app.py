"""
Build app/www: everything the Android app (app/, Capacitor) carries inside
it, so it plays without internet from the first start.

    python tools/make_app.py            (app/www)
    python tools/make_app.py --bump     (and the version number one up: a new Play Store version)
    python tools/make_app.py --version  (only show the version)
    then build the app: start/build_app_test.bat or start/build_app_release.bat (app/README.txt)

The web site loads a few files from elsewhere (fonts from Google Fonts,
PeerJS and the QR code from jsDelivr, numpy from Pyodide's site); the app
gets its own copies, downloaded once into app/vendor/ (numpy is checked
against the checksum in py/pyodide-lock.json). The page is the same
index.html as on the web site, with window.LOOOT_APP set: no service
worker, no "install" button, no local server, and invite links that point
to the web site (friends can join there without the app).
"""

import os
import sys

# This script lives in tools/; the modules and files it uses are in the
# project folder above it.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import hashlib
import json
import re
import shutil
import urllib.request

APP = os.path.join(ROOT, "app")
GRADLE = os.path.join(APP, "android", "app", "build.gradle")
WWW = os.path.join(APP, "www")
VENDOR = os.path.join(APP, "vendor")
FILES = ["index.html", "privacy.html", "boards.json", "looot.py", "nn_bot.py", "nn_encode.py",
         "mp_encode.py", "review.py", "endgame.py"]
PY = ["pyodide.js", "pyodide.asm.js", "pyodide.asm.wasm", "python_stdlib.b64.txt", "pyodide-lock.json"]
ICONS = ["favicon-32.png", "favicon-64.png", "icon-192.png"]
PEERJS = "https://cdn.jsdelivr.net/npm/peerjs@1.5.4/dist/peerjs.min.js"
QRJS = "https://cdn.jsdelivr.net/npm/qrcode-generator@1.4.4/qrcode.min.js"
FONTS = "https://fonts.googleapis.com/css2?family=Pirata+One&family=Alegreya+Sans:wght@400;500;700;800&display=swap"
# Google Fonts answers with the font format the browser asks for: say we are Chrome (woff2)
CHROME = ("Mozilla/5.0 (Linux; Android 14) AppleWebKit/537.36 (KHTML, like Gecko) "
          "Chrome/140.0 Mobile Safari/537.36")


def fetch(url, path, sha256=None):
    """url -> app/vendor/<path>, once; returns the local file."""
    out = os.path.join(VENDOR, path)
    if not os.path.exists(out):
        os.makedirs(os.path.dirname(out), exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": CHROME})
        data = urllib.request.urlopen(req, timeout=60).read()
        if sha256 and hashlib.sha256(data).hexdigest() != sha256:
            raise SystemExit("%s: wrong checksum, not used" % url)
        with open(out, "wb") as f:
            f.write(data)
        print("  downloaded %s (%d kB)" % (path, len(data) // 1024))
    return out


def fonts():
    """The two fonts as local files, and a style sheet that points to them."""
    css = open(fetch(FONTS, "fonts/fonts.src.css"), encoding="utf-8").read()
    files = re.findall(r"url\((https://fonts\.gstatic\.com/[^)]+)\)", css)
    for url in files:
        name = url.rsplit("/", 1)[1]
        fetch(url, "fonts/" + name)
        css = css.replace(url, name)
    with open(os.path.join(VENDOR, "fonts", "fonts.css"), "w", encoding="utf-8") as f:
        f.write(css)
    return len(files)


def page():
    """index.html for the app: its own copies instead of the internet."""
    s = open(os.path.join(ROOT, "index.html"), encoding="utf-8").read()

    def rep(a, b):
        nonlocal s
        if a not in s:
            raise SystemExit("index.html changed: can't find %r" % a[:60])
        s = s.replace(a, b)

    rep('<meta charset="utf-8">', '<meta charset="utf-8">\n'
        '<script>window.LOOOT_APP = true; document.documentElement.classList.add("app");</script>')
    rep('<link rel="manifest" href="manifest.webmanifest">\n', '')
    rep('<link rel="preconnect" href="https://fonts.googleapis.com">\n', '')
    rep('<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n', '')
    rep('href="%s"' % FONTS, 'href="fonts/fonts.css"')
    rep("'%s'" % PEERJS, "'lib/peerjs.min.js'")
    rep("'%s'" % QRJS, "'lib/qrcode.min.js'")
    return s


def version(bump=False):
    """The app's version (versionName in build.gradle); bump: one up. Every
    upload to the Play Store needs a higher versionCode."""
    s = open(GRADLE, encoding="utf-8").read()
    code = int(re.search(r"versionCode (\d+)", s).group(1))
    name = re.search(r'versionName "([^"]+)"', s).group(1)
    if bump:
        code += 1
        name = "1.0.%d" % code
        s = re.sub(r"versionCode \d+", "versionCode %d" % code, s)
        s = re.sub(r'versionName "[^"]+"', 'versionName "%s"' % name, s)
        with open(GRADLE, "w", encoding="utf-8") as f:
            f.write(s)
        print("version %s (versionCode %d)" % (name, code))
    return name


def main():
    if "--version" in sys.argv:
        print(version())
        return
    if "--bump" in sys.argv:
        version(bump=True)
    m = re.search(r"const WEB_MODEL = '([^']+)'", open(os.path.join(ROOT, "index.html"), encoding="utf-8").read())
    m2 = re.search(r"const WEB_MODEL_MP = '([^']+)'", open(os.path.join(ROOT, "index.html"), encoding="utf-8").read())
    models = [m.group(1), m2.group(1)]
    lock = json.load(open(os.path.join(ROOT, "py", "pyodide-lock.json"), encoding="utf-8"))
    numpy = lock["packages"]["numpy"]
    print("Downloads (once, into app/vendor/):")
    wheel = fetch("https://cdn.jsdelivr.net/pyodide/v%s/full/%s" % (lock["info"]["version"], numpy["file_name"]),
                  "py/" + numpy["file_name"], numpy.get("sha256"))
    fetch(PEERJS, "lib/peerjs.min.js")
    fetch(QRJS, "lib/qrcode.min.js")
    n_fonts = fonts()

    os.makedirs(WWW, exist_ok=True)
    for f in os.listdir(WWW):                 # empty it (the folder itself may be in use)
        p = os.path.join(WWW, f)
        shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
    for d in ("py", "models", "icons", "lib", "fonts"):
        os.makedirs(os.path.join(WWW, d))
    for f in FILES:
        if f == "index.html":
            with open(os.path.join(WWW, f), "w", encoding="utf-8") as out:
                out.write(page())
        else:
            shutil.copy(os.path.join(ROOT, f), os.path.join(WWW, f))
    for f in PY:
        shutil.copy(os.path.join(ROOT, "py", f), os.path.join(WWW, "py", f))
    shutil.copy(wheel, os.path.join(WWW, "py", numpy["file_name"]))
    for name in models:
        shutil.copy(os.path.join(ROOT, "models", name + ".npz"), os.path.join(WWW, "models", name + ".npz"))
    for f in ICONS:
        shutil.copy(os.path.join(ROOT, "icons", f), os.path.join(WWW, "icons", f))
    for f in ("peerjs.min.js", "qrcode.min.js"):
        shutil.copy(os.path.join(VENDOR, "lib", f), os.path.join(WWW, "lib", f))
    for f in os.listdir(os.path.join(VENDOR, "fonts")):
        if f != "fonts.src.css":
            shutil.copy(os.path.join(VENDOR, "fonts", f), os.path.join(WWW, "fonts", f))
    size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(WWW) for f in fs)
    print("app/www: %d files, %.1f MB (networks %s; %d font files)"
          % (sum(len(fs) for _, _, fs in os.walk(WWW)), size / 2 ** 20, " and ".join(models), n_fonts))


if __name__ == "__main__":
    main()
