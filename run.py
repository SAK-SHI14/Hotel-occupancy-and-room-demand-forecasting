"""Single-command launcher for Hotel Room Demand Forecasting.

Starts both the FastAPI REST API (port 8000) and the Streamlit Dashboard (port 8501)
together, opens the browser automatically, and cleanly shuts down on Ctrl+C.

Usage:
    python run.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import webbrowser


def main() -> None:
    print("=" * 65)
    print("🏨 HOTEL OCCUPANCY & ROOM-DEMAND FORECASTING SYSTEM")
    print("=" * 65)
    print("Starting services...")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    # 1. Start FastAPI server
    api_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "hotel_forecast.api.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
    ]
    api_proc = subprocess.Popen(api_cmd, env=env)
    print("  [✓] FastAPI REST API running at:     http://localhost:8000")
    print("  [✓] Interactive Swagger Docs at:     http://localhost:8000/docs")

    # 2. Start Streamlit dashboard
    dashboard_cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app/dashboard.py",
        "--server.port",
        "8501",
        "--server.headless",
        "true",
    ]
    dashboard_proc = subprocess.Popen(dashboard_cmd, env=env)
    print("  [✓] Streamlit Dashboard running at:  http://localhost:8501")
    print("=" * 65)
    print("Opening dashboard in your default browser...")
    print("Press Ctrl+C anytime to stop all servers.")
    print("=" * 65)

    # Allow servers a moment to initialize then open dashboard
    time.sleep(2)
    try:
        webbrowser.open("http://localhost:8501")
    except Exception:
        pass

    try:
        # Keep launcher alive and monitor processes
        while True:
            time.sleep(1)
            if api_proc.poll() is not None:
                print("\n[!] FastAPI server exited.")
                break
            if dashboard_proc.poll() is not None:
                print("\n[!] Streamlit dashboard exited.")
                break
    except KeyboardInterrupt:
        print("\nStopping all services...")
    finally:
        for proc, _name in [(dashboard_proc, "Streamlit"), (api_proc, "FastAPI")]:
            if proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=3)
                except Exception:
                    proc.kill()
        print("All services stopped cleanly. Goodbye!")


if __name__ == "__main__":
    main()
