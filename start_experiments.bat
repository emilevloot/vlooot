@echo off
rem Two experiments, about 2.5 hours each, then all against each other:
rem   c7a  the network with attention (transformer), the whole curriculum
rem   s1   the three-part network c5t_full_r10, trained further by playing
rem        only against itself, with a champion to beat (selfplay_train.py)
rem Progress: curriculum_c7a_log.txt, curriculum_s1_log.txt, compare_log.txt.
rem Stopped? Start this again: finished parts are skipped.
cd /d "%~dp0"
python curriculum.py --prefix c7a --arch attn --hours 2.5 ^
  --greedy-games 8000 --selfplay-games 8000 --window 3 ^
  --rounds resources=2,buildings=3,sites=3,longships=6 ^
  --arena-games 300 --arena-games-long 500
python selfplay_train.py --prefix s1 --init c5t_full_r10 --hours 2.5
python compare_runs.py c5t c7a --games 500
python compare_runs.py c5t s1 --games 500
python compare_runs.py s1 c7a --games 500
pause
