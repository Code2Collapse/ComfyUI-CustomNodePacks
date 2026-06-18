"""Poll ComfyUI /system_stats until ready or timeout."""
import sys, time, urllib.request
URL = "http://127.0.0.1:8188/system_stats"
for i in range(90):  # up to 180s
    try:
        with urllib.request.urlopen(URL, timeout=2) as r:
            if r.status == 200:
                print(f"READY after {i*2}s", flush=True)
                sys.exit(0)
    except Exception:
        pass
    time.sleep(2)
print("TIMEOUT", flush=True)
sys.exit(1)
