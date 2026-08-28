@echo off
REM Double-click this to open the dashboard in your browser.
cd /d "%~dp0"
set PYTHONPATH=%~dp0
"%USERPROFILE%\anaconda3\envs\influencer\python.exe" -m streamlit run app\Home.py --server.port 8502
pause
