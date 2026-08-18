@echo off
REM Double-click this to rebuild every artifact from scratch.
REM Takes roughly 45-90 minutes; the NLP stage is the slow one.
cd /d "%~dp0"
set PYTHONPATH=%~dp0
set PYTHONIOENCODING=utf-8
set TOKENIZERS_PARALLELISM=false
"%USERPROFILE%\anaconda3\envs\influencer\python.exe" -u run_pipeline.py --continue-on-error
pause
