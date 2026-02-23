"""QMD working memory tests."""
import json
import os
import tempfile
import threading
import time
from datetime import datetime, timezone

import pytest
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

QMD_SCHEMA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "qmd", "schema.json")


def load_qmd(path):
    with open(path) as f:
        return json.load(f)

def save_qmd(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")

def make_task(id, title, status="active"):
    return {"id": id, "status": status, "title": title, "progress": [], "entities": [], "decisions": [], "blockers": [], "next": ""}


class TestQMDReadWrite:
    def test_qmd_read_write(self, tmp_path):
        p = str(tmp_path / "qmd.json")
        qmd = {"session_id": "test", "tasks": [make_task("t1", "Test task")], "updated_at": "2026-01-01T00:00:00Z"}
        save_qmd(p, qmd)
        loaded = load_qmd(p)
        assert loaded["tasks"][0]["id"] == "t1"
        assert loaded["tasks"][0]["title"] == "Test task"

    def test_qmd_add_task(self, tmp_path):
        p = str(tmp_path / "qmd.json")
        qmd = {"session_id": "test", "tasks": [], "updated_at": ""}
        save_qmd(p, qmd)
        qmd = load_qmd(p)
        qmd["tasks"].append(make_task("t2", "New task"))
        save_qmd(p, qmd)
        qmd = load_qmd(p)
        assert len(qmd["tasks"]) == 1
        assert qmd["tasks"][0]["id"] == "t2"

    def test_qmd_complete_task(self, tmp_path):
        p = str(tmp_path / "qmd.json")
        qmd = {"session_id": "test", "tasks": [make_task("t1", "Do it")]}
        save_qmd(p, qmd)
        qmd = load_qmd(p)
        qmd["tasks"][0]["status"] = "done"
        qmd["tasks"][0]["outcome"] = "Did it"
        save_qmd(p, qmd)
        qmd = load_qmd(p)
        assert qmd["tasks"][0]["status"] == "done"
        assert qmd["tasks"][0]["outcome"] == "Did it"

    def test_qmd_compact(self, tmp_path):
        """Simulate compaction: done tasks move to daily log."""
        qmd_path = str(tmp_path / "qmd.json")
        log_path = str(tmp_path / "2026-01-01.md")
        
        tasks = [
            make_task("done1", "Finished task", "done"),
            make_task("active1", "Still going", "active"),
        ]
        tasks[0]["outcome"] = "Completed successfully"
        qmd = {"session_id": "test", "tasks": tasks}
        save_qmd(qmd_path, qmd)
        
        # Compact
        qmd = load_qmd(qmd_path)
        done = [t for t in qmd["tasks"] if t["status"] == "done"]
        active = [t for t in qmd["tasks"] if t["status"] != "done"]
        
        # Write done to log
        with open(log_path, "w") as f:
            for t in done:
                f.write(f"## ✅ {t['title']}\n{t.get('outcome','')}\n\n")
        
        qmd["tasks"] = active
        save_qmd(qmd_path, qmd)
        
        qmd = load_qmd(qmd_path)
        assert len(qmd["tasks"]) == 1
        assert qmd["tasks"][0]["id"] == "active1"
        assert "Finished task" in open(log_path).read()

    def test_qmd_schema_validation(self, tmp_path):
        """Validate QMD against JSON schema."""
        try:
            import jsonschema
        except ImportError:
            pytest.skip("jsonschema not installed")
        
        with open(QMD_SCHEMA_PATH) as f:
            schema = json.load(f)
        
        qmd = {
            "session_id": "test",
            "started_at": "2026-01-01T00:00:00Z",
            "tasks": [{"id": "t1", "status": "active", "title": "Test"}],
            "entities_seen": {"people": ["Yoni"]},
            "updated_at": "2026-01-01T00:00:00Z"
        }
        jsonschema.validate(qmd, schema)  # Should not raise

    def test_qmd_max_size(self, tmp_path):
        """Write 50KB+ of tasks, verify FIFO eviction works."""
        p = str(tmp_path / "qmd.json")
        MAX_SIZE = 50 * 1024  # 50KB
        
        tasks = []
        for i in range(200):
            t = make_task(f"task-{i}", f"Task number {i} with some padding text " * 5)
            t["progress"] = [f"Step {j}" for j in range(10)]
            tasks.append(t)
        
        qmd = {"session_id": "test", "tasks": tasks}
        save_qmd(p, qmd)
        
        # Check size exceeds 50KB
        size = os.path.getsize(p)
        assert size > MAX_SIZE, f"Expected >50KB, got {size}"
        
        # FIFO eviction: keep only newest tasks that fit under limit
        qmd = load_qmd(p)
        while True:
            data = json.dumps(qmd, indent=2)
            if len(data) <= MAX_SIZE:
                break
            qmd["tasks"].pop(0)  # Remove oldest
        
        save_qmd(p, qmd)
        assert os.path.getsize(p) <= MAX_SIZE + 100  # small margin for final write

    def test_qmd_concurrent_access(self, tmp_path):
        """Simulate two writers with file locking, verify no corruption."""
        import fcntl
        p = str(tmp_path / "qmd.json")
        save_qmd(p, {"session_id": "test", "tasks": []})
        lock_path = p + ".lock"
        
        errors = []
        
        def writer(writer_id, count):
            for i in range(count):
                try:
                    with open(lock_path, "w") as lf:
                        fcntl.flock(lf, fcntl.LOCK_EX)
                        qmd = load_qmd(p)
                        qmd["tasks"].append(make_task(f"w{writer_id}-{i}", f"Writer {writer_id} task {i}"))
                        save_qmd(p, qmd)
                        fcntl.flock(lf, fcntl.LOCK_UN)
                except Exception as e:
                    errors.append(str(e))
        
        t1 = threading.Thread(target=writer, args=(1, 20))
        t2 = threading.Thread(target=writer, args=(2, 20))
        t1.start(); t2.start()
        t1.join(); t2.join()
        
        # Should be valid JSON with all 40 tasks (no corruption, no lost writes)
        qmd = load_qmd(p)
        assert isinstance(qmd["tasks"], list)
        assert len(qmd["tasks"]) == 40, f"Expected 40 tasks, got {len(qmd['tasks'])}"
        assert len(errors) == 0, f"Errors: {errors}"
