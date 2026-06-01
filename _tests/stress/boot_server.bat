@echo off
cd /d D:\PROJECT\ComfyUI_windows_portable
del /Q D:\PROJECT\Custom_Nodes\_AUDIT\stress_test\server_stdout.log 2>nul
del /Q D:\PROJECT\Custom_Nodes\_AUDIT\stress_test\server_stderr.log 2>nul
start "ComfyUI" /MIN cmd /c "D:\PROJECT\ComfyUI_windows_portable\comfy_env\python.exe -s D:\PROJECT\ComfyUI_windows_portable\ComfyUI\main.py --listen 127.0.0.1 --port 8188 --disable-auto-launch 1> D:\PROJECT\Custom_Nodes\_AUDIT\stress_test\server_stdout.log 2> D:\PROJECT\Custom_Nodes\_AUDIT\stress_test\server_stderr.log"
echo launched
