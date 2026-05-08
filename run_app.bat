@echo off
REM 启动LDA文本分析系统
echo 正在启动LDA文本分析系统...
echo.

REM 激活虚拟环境并运行Streamlit
call .venv\Scripts\activate.bat
python -m streamlit run app.py

pause
