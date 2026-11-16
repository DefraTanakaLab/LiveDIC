@echo off
python -m PyInstaller --noconfirm --clean --onedir --windowed --name LiveDIC --icon "C:/Users/yamas/AppData/Local/Temp/LiveDIC.ico" --exclude-module PyQt5 --add-data=dic_core.py:. --add-data=dic_icgn.py:. --add-data=dic_rt.py:. --add-data=rt_source.py:. --add-data=rt_panel.py:. --add-data=rt_record.py:. main.py
pause
