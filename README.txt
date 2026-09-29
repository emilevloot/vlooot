LOOOT - how to play and edit
============================

PLAY
  Double-click start_game.bat. Your browser opens the game at
  http://localhost:8000. Keep the black window open while you play;
  close it to stop.
  (It needs Python installed. If "python" isn't found, try replacing
  "python" with "py" in start_game.bat.)
  Computer players: "Computer" is the greedy player, "Network" the neural
  network (2-player games only). The network's turns are played by
  server.py (it needs numpy and numba), with the best full-game network of
  the newest training run. Without them, the greedy player takes its turns.
  With a Network player in the game, the panel "What the network thinks"
  shows its win chance and, for the three-part network, how much it wants
  each item, how well each longship in the ocean fits its fjord, and (dotted
  blue rings) where a Viking is worth most to it.

EDIT THE LANDSCAPE BOARDS
  Click "Board editor" in the game (or open
  http://localhost:8000/editor.html while start_game.bat is running).
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
      python tune.py --minutes 60
  Tries variations of BOT_WEIGHTS (in looot.py) against the current best
  in the arena and keeps a variation only when it clearly scores more.
  Stop any time with Ctrl+C; running it again continues. Progress is in
  tune_log.txt, the best weights in tune_best.json.
      python tune.py --apply
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
  curriculum_report.py / curriculum_report.html   graphs of the runs
  3-4 PLAYERS
  mp_encode.py    the encoding for 2-4 players: all 4 boards (spaces not
                  in the game are switched off), "the opponent" = the
                  strongest opponent / all opponents together
  mp_game.py      fastgame.py for 2-4 players (test_mp.py checks it plays
                  exactly like looot.py)
  transfer_mp.py  turns a 2-player network into a 2-4-player one that
                  starts with everything it learned
  run_4p.py       the 4-player training: the best 2-player network (from
                  the compare_*.json matches), transferred, then trained
                  on 4-player games (prefix m4)
  gen_data.py / curriculum.py take --players 3 or 4.
  selfplay_train.py    training by playing only against itself: a new
                       network must beat the champion to replace it
  start_experiments.bat  6 hours: the attention network (c7a) and
                       c5t_full_r10 trained further against itself (s1),
                       then everything against each other
  start_compare.bat    6 hours: the value network (c5v) and the three-part
                       network (c5t) 3 hours each, then they play each
                       other (compare_runs.py, result in compare_log.txt)
  start_training.bat   the long run (prefix c6, the three-part network,
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
      python test_fastgame.py      rules, encoding and player vs looot.py
      python test_speedups.py      fast encoder / network vs the simple ones

GOOD PLACES TO START IN looot.py
  BASE_VALUE        starting points for castles, gold, sheep...
  BUILDING_STACK    tiles stacked on each house / watchtower / castle
  TROPHIES          axes needed and points
  SITES             what the construction sites need
  LONGSHIPS         all 30 longships
  Game.captures()   the house / watchtower / castle capture rules
  Bot               the computer opponent
