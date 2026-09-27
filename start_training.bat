@echo off
rem Run 3: the whole curriculum in about 6 hours (see curriculum.py).
rem Progress: curriculum_c3_log.txt. Stopped? Start this again: finished
rem stages are skipped (a stage that was busy starts over).
cd /d "%~dp0"
python curriculum.py --prefix c3 --hours 6 ^
  --greedy-games 8000 --selfplay-games 8000 --window 3 ^
  --rounds resources=2,buildings=3,sites=3,longships=10 ^
  --arena-games 300 --arena-games-long 500
pause
