@echo off
rem 3 and 4 players: s2_r19, transferred (t4_start), trained by self-play with a
rem champion on 3- and 4-player games, at most 4 hours. Progress: curriculum_t4_log.txt
cd /d "%~dp0"
python selfplay_train.py --prefix t4 --init t4_start --players 3,4 --games 1500 --window 2 --batch 2048 --match-games 300 --arena-games 200 --hours 3.8
pause
