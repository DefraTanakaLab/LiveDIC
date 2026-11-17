@echo off
python -m PyInstaller --noconfirm --clean --onedir --windowed --name LiveDIC --icon "%TEMP%\LiveDIC.ico" --exclude-module PyQt5 --add-data=dic_core.py:. --add-data=dic_icgn.py:. --add-data=dic_rt.py:. --add-data=rt_source.py:. --add-data=rt_panel.py:. --add-data=rt_record.py:. --add-data=settings_store.py:. main.py
pause
