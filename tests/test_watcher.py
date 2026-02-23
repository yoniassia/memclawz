"""Auto-indexing watcher tests."""
import json
import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# We test watcher functions directly
import zvec.watcher as watcher_mod
# But watcher.py is a script, not in zvec package — import from scripts
WATCHER_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "zvec", "watcher.py")


def _load_watcher():
    """Load watcher module from file."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("watcher", WATCHER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def watcher():
    return _load_watcher()


class TestWatcherSync:
    def test_sync_state_persistence(self, tmp_path, watcher):
        """Verify state file tracks last_sync_id."""
        state_file = str(tmp_path / "state.json")
        orig = watcher.STATE_FILE
        watcher.STATE_FILE = state_file
        try:
            state = watcher.load_state()
            assert state["last_sync_id"] == 0
            state["last_sync_id"] = 42
            state["total_synced"] = 10
            watcher.save_state(state)
            state2 = watcher.load_state()
            assert state2["last_sync_id"] == 42
            assert state2["total_synced"] == 10
        finally:
            watcher.STATE_FILE = orig

    def test_sync_idempotent(self, watcher):
        """Run sync_once twice — second run should sync 0 if nothing changed."""
        try:
            import urllib.request
            urllib.request.urlopen("http://localhost:4010/health", timeout=2)
        except:
            pytest.skip("Zvec server not running")
        
        # First sync
        n1 = watcher.sync_once()
        # Second sync immediately — nothing new
        n2 = watcher.sync_once()
        assert n2 == 0, f"Expected 0 new chunks on second sync, got {n2}"

    def test_get_max_rowid(self, watcher):
        """Verify we can read max rowid from SQLite."""
        if not os.path.exists(watcher.SQLITE_PATH):
            pytest.skip("SQLite not found")
        max_id = watcher.get_max_rowid()
        assert max_id > 0
