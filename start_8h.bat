@echo off
rem 8 hours of training: 2 players (4.5 h), then 3-4 players (3 h), then new dashboard games.
rem Progress: train_8h_log.txt, curriculum_s2_log.txt, curriculum_t4_log.txt
cd /d "%~dp0"
python train_8h.py
pause
