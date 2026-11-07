@echo off
REM ============================================================
REM SmartWrangle: one-command launcher for Windows
REM ============================================================
REM How to use: double-click this file, or type  run  in a
REM terminal opened in this folder.
REM
REM What it does, in order:
REM   1. Moves into the folder this file lives in
REM   2. Creates the virtual environment (venv) the first time
REM   3. Installs or updates the packages in requirements.txt
REM   4. Starts the app in your browser at http://localhost:8501
REM
REM To stop the app, press Ctrl+C in this window.
REM ============================================================

REM Always run from the project folder, even if started elsewhere
cd /d "%~dp0"

REM Step 2: create the venv only if it does not exist yet.
REM "py -3.12" uses the Python 3.12 install; if the py launcher
REM is missing, fall back to whatever "python" points to.
if not exist "venv\Scripts\python.exe" (
    echo Creating virtual environment...
    py -3.12 -m venv venv || python -m venv venv
)

REM Stop here with a clear message if the venv still is not there
if not exist "venv\Scripts\python.exe" (
    echo Could not create the virtual environment. Is Python 3.12 installed?
    pause
    exit /b 1
)

REM Step 3: install packages using the venv's own Python.
REM -q keeps the output short; already-installed packages are skipped.
echo Checking packages...
"venv\Scripts\python.exe" -m pip install -q -r requirements.txt
if errorlevel 1 (
    echo Package install failed. Check your internet connection and try again.
    pause
    exit /b 1
)

REM Step 4: start the app with the venv's Python
echo Starting SmartWrangle...
"venv\Scripts\python.exe" -m streamlit run app.py

REM Keep the window open if the app exits with an error
if errorlevel 1 pause
