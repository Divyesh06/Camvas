"""Packaging metadata, read once from pyproject.toml.

Single source of truth for app name, version, description, and publisher.
Every setup script (cx_Freeze, Inno Setup) consumes these values so bumping
the version in pyproject.toml flows through to the installer automatically.

Assumes the project is invoked from its root directory.
"""
import tomllib  # Python 3.11+ stdlib

with open("pyproject.toml", "rb") as _f:
    _data = tomllib.load(_f)

_project = _data["project"]
_authors = _project.get("authors") or [{}]

NAME = _project["name"]
VERSION = _project["version"]
DESCRIPTION = _project.get("description", NAME)
PUBLISHER = _authors[0].get("name", NAME)
