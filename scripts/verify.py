#!/usr/bin/env python3
"""
QMDZvec Post-Install Verification
Validates that the memory system is working correctly.
"""
import json
import sys
import time
import urllib.request
import random
import os

ZVEC_PORT = os.environ.get("ZVEC_PORT", "4010")
BASE = f"http://localhost:{ZVEC_PORT}"


def api(path, data=None):
    """Simple HTTP helper."""
    if data:
        req = urllib.request.Request(
            f"{BASE}{path}",
            data=json.dumps(data).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
    else:
        req = urllib.request.Request(f"{BASE}{path}")
    resp = urllib.request.urlopen(req, timeout=10)
    return json.loads(resp.read())


def main():
    errors = 0

    # 1. Health check
    try:
        h = api("/health")
        assert h.get("status") == "ok", f"Bad health: {h}"
        print(f"  ✅ Health: {h['status']} (engine: {h.get('engine', '?')})")
    except Exception as e:
        print(f"  ❌ Health check failed: {e}")
        errors += 1
        sys.exit(1)

    # 2. Stats
    try:
        s = api("/stats")
        total = s.get("total_docs", 0)
        dim = s.get("dimension", s.get("dim", "?"))
        print(f"  ✅ Stats: {total} chunks, {dim}-dim")
    except Exception as e:
        print(f"  ❌ Stats failed: {e}")
        errors += 1
        total = 0

    if total == 0:
        print("  ⚠️  No chunks indexed — skipping search verification")
        print(f"\n{'✅' if errors == 0 else '❌'} Verification {'passed' if errors == 0 else 'failed'}")
        sys.exit(errors)

    # 3. Search test — use a random embedding from the index
    latencies = []
    try:
        # Get a sample embedding by searching with a random vector
        import numpy as np
        dim_val = int(dim) if str(dim).isdigit() else 768
        
        # Do 3 test searches
        for i in range(3):
            rand_emb = np.random.randn(dim_val).astype(float).tolist()
            # Normalize
            norm = sum(x*x for x in rand_emb) ** 0.5
            rand_emb = [x / norm for x in rand_emb]
            
            t0 = time.time()
            results = api("/search", {"embedding": rand_emb, "topk": 5})
            latency = (time.time() - t0) * 1000
            latencies.append(latency)
            
            count = results.get("count", len(results.get("results", [])))
            if count == 0 and i == 0:
                print(f"  ⚠️  Search returned 0 results (may need index rebuild)")

        avg_latency = sum(latencies) / len(latencies)
        print(f"  ✅ Search: {avg_latency:.0f}ms avg ({len(latencies)} queries)")
    except Exception as e:
        print(f"  ❌ Search test failed: {e}")
        errors += 1

    # 4. QMD check
    qmd_path = os.path.expanduser("~/.openclaw/workspace/memory/qmd/current.json")
    if os.path.exists(qmd_path):
        try:
            with open(qmd_path) as f:
                qmd = json.load(f)
            assert "tasks" in qmd, "Missing 'tasks' field"
            print(f"  ✅ QMD: valid ({len(qmd.get('tasks', []))} active tasks)")
        except Exception as e:
            print(f"  ❌ QMD invalid: {e}")
            errors += 1
    else:
        print(f"  ⚠️  QMD not found at {qmd_path}")

    # Summary
    print("")
    if errors == 0:
        avg = sum(latencies) / len(latencies) if latencies else 0
        print(f"✅ Memory verification passed: {total} chunks indexed, search working ({avg:.0f}ms avg)")
    else:
        print(f"❌ Verification failed with {errors} error(s)")
    
    sys.exit(errors)


if __name__ == "__main__":
    main()
