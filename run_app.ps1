# PowerShell启动脚本
Write-Host "正在启动LDA文本分析系统..." -ForegroundColor Green
Write-Host ""

# 激活虚拟环境
& .\.venv\Scripts\Activate.ps1

# 运行Streamlit
python -m streamlit run app.py
