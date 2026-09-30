@echo off
rem Run 5: the run-4 network (c5v) and the three-part network (c5t), each
rem trained for 3 hours with the same curriculum, then they play each other.
rem Progress: (in results\) curriculum_c5v_log.txt, curriculum_c5t_log.txt, compare_log.txt.
rem Stopped? Start this again: finished stages and runs are skipped.
cd /d "%~dp0.."
python curriculum.py --prefix c5v --arch value --hours 2.95 ^
  --greedy-games 8000 --selfplay-games 8000 --window 3 ^
  --rounds resources=2,buildings=3,sites=3,longships=6 ^
  --arena-games 300 --arena-games-long 500
python curriculum.py --prefix c5t --arch three --hours 2.95 ^
  --greedy-games 8000 --selfplay-games 8000 --window 3 ^
  --rounds resources=2,buildings=3,sites=3,longships=6 ^
  --arena-games 300 --arena-games-long 500
python tools\compare_runs.py c5v c5t --games 500
pause
