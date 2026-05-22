@echo off
python -m PyInstaller --noconfirm --clean --onedir --windowed --name LiveDIC --icon "C:/Users/yamas/AppData/Local/Temp/LiveDIC.ico" --exclude-module PyQt5 --add-data=dic_core.py:. main.py
pause
