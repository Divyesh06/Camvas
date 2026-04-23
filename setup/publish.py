"""Create a GitHub release and upload the installer via the gh CLI.

Reads NAME/VERSION from _meta.py (the same source of truth as the build and
installer scripts), then tags the release `v<VERSION>` and attaches the
installer produced by `poe release`.

Requires: gh CLI installed and authenticated (`gh auth login`).
Assumes the project is invoked from its root directory.
"""
import subprocess
import sys
from pathlib import Path
from _meta import NAME, VERSION

tag = f"v{VERSION}"
installer = Path("build") / "installer" / f"{NAME}-{VERSION}.exe"

if not installer.exists():
    sys.exit(f"Installer not found at {installer} — run `poe release` first.")

cmd = [
    "gh", "release", "create", tag, str(installer),
    "--title", f"{NAME} {VERSION}",
    "--generate-notes",
]
print("Running:", " ".join(cmd))
sys.exit(subprocess.run(cmd).returncode)
