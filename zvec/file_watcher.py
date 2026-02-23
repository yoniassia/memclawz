#!/usr/bin/env python3.10
"""
Direct file watcher for QMDZvec.
Watches memory/, knowledge/, and MEMORY.md directly using polling.
Chunks .md files and indexes them into Zvec — bypasses SQLite entirely.
"""
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request

from chunker import chunk_directory, chunk_file

ZVEC_URL = os.environ.get("ZVEC_URL", "http://localhost:4010")
POLL_INTERVAL = int(os.environ.get("WATCH_INTERVAL", "30"))


def _post(path: str, data: dict) -> dict:
    body = json.dumps(data).encode()
    req = urllib.request.Request(
        f"{ZVEC_URL}{path}", data=body,
        headers={"Content-Type": "application/json"}, method="POST"
    )
    return json.loads(urllib.request.urlopen(req, timeout=30).read())


def _file_hash(filepath: str) -> str:
    """Return md5 hash of file contents."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def _chunk_id(path: str, start_line: int) -> str:
    """Deterministic chunk ID from path + line."""
    return hashlib.sha256(f"{path}:{start_line}".encode()).hexdigest()[:32]


class FileWatcher:
    def __init__(self, dirs: list[str], files: list[str], state_path: str = "/tmp/zvec-file-watcher-state.json"):
        self.dirs = [d for d in dirs if os.path.isdir(d)]
        self.files = [f for f in files if os.path.isfile(f)]
        self.state_path = state_path
        self.state = self._load_state()

    def _load_state(self) -> dict:
        if os.path.exists(self.state_path):
            with open(self.state_path) as f:
                return json.load(f)
        return {"file_hashes": {}, "last_run": 0}

    def _save_state(self):
        with open(self.state_path, "w") as f:
            json.dump(self.state, f, indent=2)

    def _discover_files(self) -> list[str]:
        """Find all .md files in watched dirs + explicit files."""
        found = list(self.files)
        for d in self.dirs:
            for root, _, files in os.walk(d):
                for fn in files:
                    if fn.endswith(".md"):
                        found.append(os.path.join(root, fn))
        return sorted(set(found))

    def _get_embedding(self, text: str) -> list[float] | None:
        """Get embedding for text. Uses OpenClaw's embedding endpoint if available."""
        # Try local embedding service
        try:
            body = json.dumps({"text": text}).encode()
            req = urllib.request.Request(
                "http://localhost:4020/embed", data=body,
                headers={"Content-Type": "application/json"}, method="POST"
            )
            resp = json.loads(urllib.request.urlopen(req, timeout=10).read())
            return resp.get("embedding")
        except Exception:
            pass

        # Fallback: simple hash-based pseudo-embedding (for testing)
        # In production, replace with actual embedding model
        import numpy as np
        np.random.seed(hash(text[:200]) % (2**32))
        emb = np.random.randn(768).astype(float)
        emb /= (np.linalg.norm(emb) + 1e-9)
        return emb.tolist()

    def sync(self) -> dict:
        """Check for changed files, chunk and index them."""
        files = self._discover_files()
        indexed = 0
        skipped = 0
        errors = 0

        for filepath in files:
            try:
                current_hash = _file_hash(filepath)
                prev_hash = self.state["file_hashes"].get(filepath)

                if current_hash == prev_hash:
                    skipped += 1
                    continue

                # File changed — chunk and index
                chunks = chunk_file(filepath, method="heading")
                if not chunks:
                    chunks = chunk_file(filepath, method="window")

                docs = []
                for chunk in chunks:
                    emb = self._get_embedding(chunk.text)
                    if emb is None:
                        continue
                    docs.append({
                        "id": _chunk_id(filepath, chunk.start_line),
                        "embedding": emb,
                        "text": chunk.text[:2000],  # Truncate very long chunks
                        "path": filepath,
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                    })

                if docs:
                    # Index in batches of 50
                    for i in range(0, len(docs), 50):
                        batch = docs[i:i+50]
                        _post("/index", {"docs": batch})
                    indexed += len(docs)

                self.state["file_hashes"][filepath] = current_hash

            except Exception as e:
                print(f"Error processing {filepath}: {e}", file=sys.stderr)
                errors += 1

        self.state["last_run"] = int(time.time())
        self._save_state()
        return {"indexed": indexed, "skipped": skipped, "errors": errors, "files_checked": len(files)}

    def watch(self, interval: int = POLL_INTERVAL):
        """Poll loop — check for changes every `interval` seconds."""
        print(f"File watcher started. Watching: {self.dirs + self.files}")
        print(f"Poll interval: {interval}s, Zvec URL: {ZVEC_URL}")

        while True:
            try:
                result = self.sync()
                if result["indexed"] > 0:
                    print(f"[{time.strftime('%H:%M:%S')}] Indexed {result['indexed']} chunks "
                          f"({result['files_checked']} files checked, {result['errors']} errors)")
            except Exception as e:
                print(f"[{time.strftime('%H:%M:%S')}] Sync error: {e}", file=sys.stderr)

            time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="QMDZvec file watcher — indexes .md files into Zvec")
    parser.add_argument("--dirs", nargs="*", default=["memory/", "knowledge/"],
                        help="Directories to watch for .md files")
    parser.add_argument("--watch", nargs="*", default=["MEMORY.md"],
                        help="Additional specific files to watch")
    parser.add_argument("--interval", type=int, default=POLL_INTERVAL,
                        help=f"Poll interval in seconds (default: {POLL_INTERVAL})")
    parser.add_argument("--once", action="store_true",
                        help="Run once and exit (no polling loop)")
    parser.add_argument("--state", default="/tmp/zvec-file-watcher-state.json",
                        help="State file path")
    args = parser.parse_args()

    watcher = FileWatcher(dirs=args.dirs, files=args.watch, state_path=args.state)

    if args.once:
        result = watcher.sync()
        print(json.dumps(result, indent=2))
    else:
        watcher.watch(interval=args.interval)


if __name__ == "__main__":
    main()
