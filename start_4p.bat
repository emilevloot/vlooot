@echo off
rem 4 players: the best 2-player network (from the compare_*.json matches),
rem transferred to 4 players and trained for 2 hours. Progress: curriculum_m4_log.txt
cd /d "%~dp0"
python run_4p.py --hours 2
pause
