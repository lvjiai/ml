@echo off
chcp 65001 >nul
cd /d "%~dp0"

rem 双击本文件即可启动 demo；关闭窗口即停止服务。
rem 浏览器会自动打开 http://localhost:8501，若没有请手动访问。
python -m streamlit run app\streamlit_app.py --server.headless false --browser.gatherUsageStats false

echo.
echo 服务已停止。按任意键关闭窗口。
pause >nul
