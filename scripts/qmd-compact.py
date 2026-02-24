#!/usr/bin/env python3
"""QMD Compaction Script — moves completed tasks to daily log, trims QMD."""

import json
import os
from datetime import datetime, timezone

WORKSPACE = os.path.expanduser("~/.openclaw/workspace")
QMD_PATH = os.path.join(WORKSPACE, "memory/qmd/current.json")
MEMORY_DIR = os.path.join(WORKSPACE, "memory")


def load_qmd():
    if not os.path.exists(QMD_PATH):
        return None
    with open(QMD_PATH) as f:
        return json.load(f)


def save_qmd(qmd):
    with open(QMD_PATH, "w") as f:
        json.dump(qmd, f, indent=2)
        f.write("\n")


def today_log_path():
    return os.path.join(MEMORY_DIR, datetime.now(timezone.utc).strftime("%Y-%m-%d") + ".md")


def compact():
    qmd = load_qmd()
    if not qmd:
        print("No QMD found, nothing to compact.")
        return

    tasks = qmd.get("tasks", [])
    done = [t for t in tasks if t.get("status") == "done"]
    active = [t for t in tasks if t.get("status") != "done"]

    if not done:
        print(f"No completed tasks to compact. {len(active)} active tasks remain.")
        return

    # Build summary for daily log
    lines = [
        "",
        "## QMD Compaction Summary",
        f"*Compacted at {datetime.now(timezone.utc).isoformat()}*",
        "",
    ]

    for t in done:
        lines.append(f"### ✅ {t.get('title', t.get('id', 'unknown'))}")
        if t.get("outcome"):
            lines.append(f"**Outcome:** {t['outcome']}")
        if t.get("progress"):
            for p in t["progress"]:
                lines.append(f"- {p}")
        if t.get("decisions"):
            lines.append("**Decisions:**")
            for d in t["decisions"]:
                lines.append(f"- {d}")
        if t.get("entities"):
            lines.append(f"**Entities:** {', '.join(t['entities'])}")
        lines.append("")

    # Append to daily log
    log_path = today_log_path()
    with open(log_path, "a") as f:
        f.write("\n".join(lines) + "\n")

    # Update QMD — keep only active tasks
    qmd["tasks"] = active
    qmd["updated_at"] = datetime.now(timezone.utc).isoformat()
    save_qmd(qmd)

    print(f"Compacted {len(done)} completed tasks to {log_path}")
    print(f"{len(active)} active tasks remain in QMD")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="QMD Compaction")
    parser.add_argument("--auto", action="store_true",
                        help="Silent mode for cron/heartbeat — no output unless work done")
    args = parser.parse_args()

    if args.auto:
        # Silent mode: suppress output when nothing to do
        import io, sys
        buf = io.StringIO()
        _orig_print = print
        def print(*a, **kw):
            _orig_print(*a, file=buf, **kw)
        compact()
        output = buf.getvalue()
        if "Compacted" in output:
            sys.stdout.write(output)
    else:
        compact()
