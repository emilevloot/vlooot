@echo off
rem Self-play with a champion (with the network's proposals): run s2
rem continues from its champion, at most 4 hours. Progress: curriculum_s2_log.txt
cd /d "%~dp0"
python selfplay_train.py --prefix s2 --init s2_r7 --hours 3.8
pause
