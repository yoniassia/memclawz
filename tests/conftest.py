import os
import sys
import json
import tempfile
import pytest

# Add parent to path so zvec package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

QMD_SCHEMA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "qmd", "schema.json")
WORKSPACE = os.path.expanduser("~/.openclaw/workspace")
QMD_PATH = os.path.join(WORKSPACE, "memory/qmd/current.json")
ZVEC_URL = "http://localhost:4010"
SQLITE_PATH = os.path.expanduser("~/.openclaw/memory/main.sqlite")


@pytest.fixture
def tmp_qmd(tmp_path):
    """Create a temporary QMD file for testing."""
    qmd_path = tmp_path / "current.json"
    qmd_path.write_text(json.dumps({
        "session_id": "test",
        "started_at": "2026-01-01T00:00:00Z",
        "tasks": [],
        "entities_seen": {},
        "updated_at": "2026-01-01T00:00:00Z"
    }, indent=2))
    return str(qmd_path)


@pytest.fixture
def qmd_schema():
    with open(QMD_SCHEMA_PATH) as f:
        return json.load(f)
