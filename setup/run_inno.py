"""Invoke Inno Setup's ISCC compiler with metadata from pyproject.toml.

Reads the live app name/version/publisher from setup/_meta.py (which in turn
reads pyproject.toml) and passes them to ISCC as /D preprocessor defines,
so setup_generator.iss stays agnostic of the specific values.

Assumes the project is invoked from its root directory.
"""
import subprocess
import sys
from pathlib import Path
from _meta import NAME, VERSION, PUBLISHER

ISCC_PATH = r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
ISS_FILE = r"setup\setup_generator.iss"
BUILD_DIR = r"exe.win-amd64-3.12"

defines = {
    "MyAppName": NAME,
    "MyAppVersion": VERSION,
    "MyAppPublisher": PUBLISHER,
    "MyBuildDir": BUILD_DIR,
}

cmd = [ISCC_PATH] + [f"/D{k}={v}" for k, v in defines.items()] + [ISS_FILE]
print("Running:", " ".join(cmd))
result = subprocess.run(cmd, capture_output=True, text=True)

print("STDOUT:\n", result.stdout)
if result.stderr:
    print("STDERR:\n", result.stderr, file=sys.stderr)

sys.exit(result.returncode)
