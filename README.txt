LOOOT - how to play and edit
============================

PROJECT MAP
  index.html, editor.html      the game and the board editor (web pages)
  dashboard.html               how the network plays against itself: every
                               game clickable, mistakes, replays
  dashboard_greedy.html        the network against the greedy player
  history.html                 every run and network so far, and what changed
  boards.json                  the landscape boards (made with the editor)
  manifest.webmanifest, sw.js  the installable web app: name, icon, and the
                               offline copy (see PLAY ON YOUR PHONE)
  icons\                       the app icons (tools\make_icons.py draws them)

  The game, the coach and the AI (these import each other, so they stay
  together in this folder; the web page loads some of them too):
    looot.py        the rules, scoring and the greedy computer player
    review.py       the coach: every move judged like a chess trainer
    analysis.py     the coach's deep analysis: a move played out many times
    endgame.py      the last 2 turns of a player, counted exactly
    server.py       the local web server (start\start_game.bat)
    nn_bot.py       the network player
    nn_encode.py, mp_encode.py     positions as numbers (2 / 2-4 players)
    nn_model.py, gpu_net.py        the networks (PyTorch)
    fastgame.py, mp_game.py        the fast (Numba) game for training
  Training:
    curriculum.py, curriculum_report.py, selfplay_train.py,
    gen_data.py, train_nn.py, arena.py

  start\      double-click these (start_game.bat plays; the rest train)
  tools\      scripts run by hand: the 8-hour training, dashboards,
              comparisons, tuning, 4-player start, code generators
  tests\      checks that the fast engines play exactly like looot.py
  templates\  the dashboards' page templates
  results\    logs and results of every training run, comparisons,
              tuning, and the reports (curriculum_report.html, ...)
  models\     trained networks      data\     training data (big)
  replays\    recorded games for the dashboard's replays
  py\         Pyodide: the Python that runs in the browser. Don't edit.

PLAY
  Double-click start\start_game.bat. Your browser opens the game at
  http://localhost:8000. Keep the black window open while you play;
  close it to stop.
  (It needs Python installed. If "python" isn't found, try replacing
  "python" with "py" in start\start_game.bat.)
  Computer players: "Computer" is the greedy player, "Network" the neural
  network (2-4 players). The network's turns are played by
  server.py (it needs numpy and numba), with the best full-game network of
  the newest training run. Without them, the greedy player takes its turns.
  With a Network player in the game, the panel "What the network thinks"
  shows its win chance and, for the three-part network, how much it wants
  each item, how well each longship in the ocean fits its fjord, and (dotted
  blue rings) where a Viking is worth most to it.

PLAY WITH FRIENDS ONLINE
  The game opens with a menu. "Play with friends" makes a lobby with a
  code, a link and a QR code; friends open the link (or choose "Join a
  game" and type the code). Every seat can be a friend, you, the computer
  or the network; then Start. After the game, "Back to the lobby" plays
  again with the same friends.
  The host's browser runs the game, the network and the coach; the
  friends' browsers only draw the board and send their moves, so they
  need no Python and join quickly. The browsers find each other through
  PeerJS (a free matchmaking service) and then talk directly (WebRTC).
  It needs internet, and the host must keep the game open: when the host
  leaves, the game ends. A friend who loses the connection opens the link
  again and gets the seat back; meanwhile the host can let the computer
  play for them.

PLAY ON YOUR PHONE (OR ANY COMPUTER), ALSO WITHOUT INTERNET
  Open https://emilevloot.github.io/vlooot/ and install it as an app:
    Android (Chrome): the "Install app" button in the game, or the menu
                      (three dots) -> Install app / Add to Home screen.
    iPhone / iPad (Safari): Share -> Add to Home Screen.
    Windows / Mac (Chrome, Edge): the install icon in the address bar.
  The first time it downloads about 32 MB (Python for the browser, numpy
  and the two networks); after that it starts from the phone's own copy,
  also offline (sw.js). A new version of the game arrives by itself the
  next time you open it with internet. Changed WEB_MODEL in index.html?
  Raise the version in sw.js (looot-v1 -> looot-v2) so old copies go.

EDIT THE LANDSCAPE BOARDS
  Click "Board editor" in the game (or open
  http://localhost:8000/editor.html while start\start_game.bat is running).
  Pick a space type (keys 1-7), click or drag over the boards, switch
  between side A and B per board, then "Save boards". Start a new game
  to play on them. Ctrl+Z undoes, Ctrl+S saves.
  Every game picks which of the 4 boards are used (2 for 2 players, 3 for
  3, all 4 for 4), in random places, each on a random side. For training,
  gen_data.py plays half of the games on newly made random boards
  (--random-share), so a network can't learn the boards by heart. Layouts are stored in boards.json;
  delete that file to go back to random boards.
  Boards 1A, 2A and 3A are the three board sides printed in the
  rulebook. The rulebook doesn't show the other five sides, so they
  are random (with the same mix of spaces). Paint them to match your
  own boards.

EDIT
  looot.py    All the game rules, scoring and the computer player.
              Open it in VS Code or Spyder. After saving, reload the
              browser page (F5) and start a new game to see the change.
  index.html  The board drawing, buttons and colours (HTML/JavaScript).
  editor.html The landscape board editor.
  server.py   The small local web server (also saves boards.json).
  py\         Pyodide, the Python that runs inside the browser. Don't edit.

TEST WITHOUT THE BROWSER
  In a terminal in this folder:   python looot.py
  This plays a full 3-player game between computer players on your
  boards.json and prints the scores. Handy for checking rule changes.

TEST ARENA (compare computer players)
  In a terminal in this folder:
      python arena.py greedy random random
      python arena.py greedy greedy --games 300
  Plays many games (seats rotated, same boards for each rotation) and
  prints win rates, scores, where the points come from and thinking time.
  The "±" is the 95% margin: if two win rates' ranges don't overlap, the
  difference is real. New players are added to PLAYERS in arena.py.
  Players: random, greedy (the game's computer player, with the tuned
  BOT_WEIGHTS) and original (greedy with the numbers from before tuning).

TUNING THE COMPUTER PLAYER
      python tools\tune.py --minutes 60
  Tries variations of BOT_WEIGHTS (in looot.py) against the current best
  in the arena and keeps a variation only when it clearly scores more.
  Stop any time with Ctrl+C; running it again continues. Progress is in
  results\tune_log.txt, the best weights in results\tune_best.json.
      python tools\tune.py --apply
  writes the best weights into looot.py, so the game uses them.

NEURAL NETWORK PLAYER (2-player games, needs PyTorch + an NVIDIA GPU)
  nn_encode.py    turns a position into numbers for the network
  nn_model.py     the networks (hex convolutions over the boards):
                    value  land, fjords and the rest combined at the end
                    attn   the value network's hex layers, then attention
                           (a small transformer) over all spaces of land,
                           both fjords and the 5 longships
                    three  three parts passing each other messages:
                           fjord part   -> a value for every item
                           board part   -> what a Viking on each space is worth
                           ship part    -> how well each longship fits
  gpu_net.py      the network on the GPU for self-play
  gen_data.py     plays games and saves positions + how they ended
  train_nn.py     trains a network on that data (on the GPU)
  nn_bot.py       a player that uses a trained network
  curriculum.py   the whole training: rules added step by step
                  (resources, buildings, sites, longships, full game),
                  with self-play rounds and TD learning in every step
  curriculum_report.py / results\curriculum_report.html   graphs of the runs
  3-4 PLAYERS
  mp_encode.py    the encoding for 2-4 players: all 4 boards (spaces not
                  in the game are switched off), "the opponent" = the
                  strongest opponent / all opponents together
  mp_game.py      fastgame.py for 2-4 players (made by tools\make_mp_game.py;
                  tests\test_mp.py checks it plays exactly like looot.py)
  tools\transfer_mp.py  turns a 2-player network into a 2-4-player one that
                  starts with everything it learned
  tools\run_4p.py  the 4-player training: the best 2-player network (from
                  the results\compare_*.json matches), transferred, then trained
                  on 4-player games (prefix m4)
  gen_data.py / curriculum.py take --players 3 or 4.
  selfplay_train.py    training by playing only against itself: a new
                       network must beat the champion to replace it
  start\start_experiments.bat  6 hours: the attention network (c7a) and
                       c5t_full_r10 trained further against itself (s1),
                       then everything against each other
  start\start_compare.bat    6 hours: the value network (c5v) and the three-part
                       network (c5t) 3 hours each, then they play each
                       other (tools\compare_runs.py, result in results\compare_log.txt)
  start\start_training.bat   the long run (prefix c6, the three-part network,
                       about 6 hours; the last
                       stage keeps training until the time is used up)
  Try a trained network in the arena:
      python arena.py nn:c4_full_r8 greedy
      python arena.py nn:c4_full_r8+2 greedy     (looks 2 turns ahead)
      python arena.py nn:c4_full_r8@2 greedy     (only takes a longship
                                                   if it looks 2 points better)
  Play with only some rules:  python arena.py greedy random --rules sites
  data\ and models\ hold the training data and networks. Networks from
  before encoding version 3 (cur_*, c2_*) no longer load. Version 3 tells
  the network per landscape space what a Viking there would give each
  player (tiles, chain size, items on their "shopping list": what their
  unfinished longships and construction sites still miss).

FAST TRAINING ENGINE (needs: pip install numba)
  fastgame.py     the game for 2 players as numpy arrays, compiled with
                  Numba, plus the network player on top of it. Only for
                  training: looot.py stays the real game.
  gen_data.py uses it by itself for network self-play: with a GPU, a few
  processes each play 64 games in step and let the GPU judge all their
  positions in one batch (about 100-150 games/s); without a GPU it runs
  on the CPU. Options: --cpu, --gpu-procs 6, --parallel 64.
  Set LOOOT_ENGINE=python to use the old (slow) way.
  Checks that the fast versions give exactly the same results:
      python tests\test_fastgame.py   rules, encoding and player vs looot.py
      python tests\test_mp.py         the same for 2-4 players (mp_game.py)
      python tests\test_speedups.py   fast encoder / network vs the simple ones

GOOD PLACES TO START IN looot.py
  BASE_VALUE        starting points for castles, gold, sheep...
  BUILDING_STACK    tiles stacked on each house / watchtower / castle
  TROPHIES          axes needed and points
  SITES             what the construction sites need
  LONGSHIPS         all 30 longships
  Game.captures()   the house / watchtower / castle capture rules
  Bot               the computer opponent

THE COACH, THE DASHBOARDS AND THE 8-HOUR TRAINING
  review.py       after every turn: how many points the move lost against
                  the network's best move -> Brilliant ... Blunder, accuracy,
                  and why: what the better move does (fills a longship, gets
                  an item your Altar needs, keeps a shield ...) or, for a good
                  move, what it does better than the next best
                  (the game page's "Coach", server.py /ai-review)
  analysis.py     "Play it out": the network plays the rest of the game from
                  the move played and from the best few others, 32 or 96
                  times each, every move the same futures (bag order, dice);
                  the average end score shows what the move really cost, with
                  a margin (server.py /ai-analyze; only the local game, it
                  needs the fast engine: a turn takes about 5-15 seconds)
  endgame.py      a player's last 2 turns counted exactly instead of guessed
                  (the player uses it; the coach and the dashboard check with it)
  tools\selfplay_record.py   the network against itself, every game recorded
                  (replays\, results\selfplay_data.json)
  tools\dashboard_data.py    the network against greedy (results\dashboard_data.json)
  tools\make_dashboard.py    dashboard.html and dashboard_greedy.html from those
  tools\history_data.py     all runs, networks and training data summed up in
                  results\history.json (run it before cleaning up models\ and data\)
  tools\make_history.py     history.html from that
  tools\train_8h.py          (start\start_8h.bat) 2 players 4.5 hours, then 3-4
                  players, then new dashboard games; logs in results\