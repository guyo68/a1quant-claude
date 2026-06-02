import subprocess, sys
subprocess.run([sys.executable, "-m", "streamlit", "run",
                "energy_dashboard/app.py", "--server.port", "8501"])
