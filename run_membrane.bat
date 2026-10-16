@echo off
call C:\Users\ballm\anaconda3\Scripts\activate.bat streamlit_apps
python -m streamlit run src\membrane_app.py
pause
