@echo off
rem Run 6: the three-part network, the whole curriculum in about 6 hours
rem (see curriculum.py). Progress: (in results\) curriculum_c6_log.txt. Stopped? Start this
rem again: finished stages are skipped (a stage that was busy starts over).
cd /d "%~dp0.."
python curriculum.py --prefix c6 --arch three --hours 6 ^
  --greedy-games 8000 --selfplay-games 8000 --window 3 ^
  --rounds resources=2,buildings=3,sites=3,longships=6 ^
  --arena-games 300 --arena-games-long 500
pause
