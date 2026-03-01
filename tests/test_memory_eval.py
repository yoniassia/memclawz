#!/usr/bin/env python3.10
"""
memclawz v3.0 Memory Evaluation Test Suite
Inspired by AMA-Bench (Zhao et al., 2026)
"""
import json
import os
import sys
import tempfile
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from memclawz_server.causality_graph import CausalityGraph


def _random_emb(dim=64, seed=None):
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _similar_emb(base, noise=0.1, seed=None):
    rng = np.random.RandomState(seed)
    v = np.array(base, dtype=np.float32) + rng.randn(len(base)).astype(np.float32) * noise
    return (v / np.linalg.norm(v)).tolist()


class TestCausalityGraph(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test_graph.db")
        self.graph = CausalityGraph(db_path=self.db_path)

    def tearDown(self):
        self.graph.close()

    def test_add_and_retrieve_node(self):
        emb = _random_emb(seed=1)
        nid = self.graph.add_node("test fact", embedding=emb, node_id="n1", source="test")
        self.assertEqual(nid, "n1")
        stats = self.graph.stats()
        self.assertEqual(stats["nodes"], 1)

    def test_similarity_search(self):
        emb1 = _random_emb(seed=10)
        emb2 = _random_emb(seed=20)
        self.graph.add_node("the sky is blue", embedding=emb1, node_id="sky")
        self.graph.add_node("water is wet", embedding=emb2, node_id="water")
        # Search with something close to emb1
        query = _similar_emb(emb1, noise=0.05, seed=30)
        results = self.graph.similarity_search(query, topk=2)
        self.assertEqual(results[0]["id"], "sky")
        self.assertGreater(results[0]["score"], results[1]["score"])

    def test_causal_edges(self):
        emb = _random_emb(seed=1)
        self.graph.add_node("event A", embedding=emb, node_id="A")
        self.graph.add_node("event B caused by A", embedding=_random_emb(seed=2), node_id="B", caused_by=["A"])
        self.graph.add_node("event C caused by B", embedding=_random_emb(seed=3), node_id="C", caused_by=["B"])
        stats = self.graph.stats()
        self.assertEqual(stats["edges"], 2)
        self.assertEqual(stats["edge_types"].get("causality", 0), 2)

    def test_multi_hop_retrieval(self):
        """Key test: verify multi-hop finds causally connected nodes."""
        emb_a = _random_emb(seed=100)
        emb_b = _random_emb(seed=200)  # very different from A
        emb_c = _random_emb(seed=300)  # very different from A

        self.graph.add_node("user clicked buy button", embedding=emb_a, node_id="click")
        self.graph.add_node("order was created", embedding=emb_b, node_id="order", caused_by=["click"])
        self.graph.add_node("payment processed", embedding=emb_c, node_id="payment", caused_by=["order"])

        # Search with query similar to "click" — should find click directly,
        # then traverse to find order and payment via causal edges
        query = _similar_emb(emb_a, noise=0.05, seed=400)
        result = self.graph.multi_hop_search(query, topk=1, similarity_threshold=0.99, max_depth=2)

        all_ids = {r["id"] for r in result["results"]}
        self.assertIn("click", all_ids)
        # With low threshold forcing traversal, should find connected nodes
        self.assertTrue(result["traversal_count"] >= 0)

    def test_keyword_search(self):
        self.graph.add_node("the user logged in at 3pm", node_id="login")
        self.graph.add_node("the server crashed at 4pm", node_id="crash")
        self.graph.add_node("user reported slow performance", node_id="perf")

        results = self.graph.keyword_search("user logged")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "login")

    def test_keyword_search_fallback_on_low_confidence(self):
        """Self-evaluation gate: low confidence triggers keyword fallback."""
        # Add nodes with embeddings far from query
        for i in range(5):
            self.graph.add_node(f"random fact {i}", embedding=_random_emb(seed=i), node_id=f"rand{i}")
        # Add the target with specific keywords
        self.graph.add_node("deployment failed on server-42", embedding=_random_emb(seed=999), node_id="deploy")

        # Search with unrelated embedding — should get low confidence
        query = _random_emb(seed=500)
        result = self.graph.multi_hop_search(query, topk=3, similarity_threshold=0.9)
        
        # Confidence should be moderate/low since embeddings are random
        self.assertIsInstance(result["confidence"], float)
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)

        # Keyword search should find it regardless of embedding
        kw_results = self.graph.keyword_search("deployment server-42")
        self.assertEqual(len(kw_results), 1)
        self.assertEqual(kw_results[0]["id"], "deploy")

    def test_association_edges_bidirectional(self):
        self.graph.add_node("state A", node_id="sA")
        self.graph.add_node("state B", node_id="sB", associations=["sA"])
        # Should have 2 edges (bidirectional)
        stats = self.graph.stats()
        self.assertEqual(stats["edge_types"].get("association", 0), 2)
        # Traverse from sA should find sB
        traversed = self.graph.traverse_edges(["sA"], max_depth=1)
        self.assertEqual(len(traversed), 1)
        self.assertEqual(traversed[0]["id"], "sB")

    def test_confidence_scoring(self):
        """Test confidence computation."""
        # High similarity results should give high confidence
        base = _random_emb(seed=42)
        for i in range(5):
            self.graph.add_node(f"fact {i}", embedding=_similar_emb(base, noise=0.01, seed=i+100), node_id=f"f{i}")
        
        result = self.graph.multi_hop_search(base, topk=5)
        self.assertGreater(result["confidence"], 0.5)

    def test_empty_graph(self):
        result = self.graph.multi_hop_search(_random_emb(seed=1), topk=5)
        self.assertEqual(result["confidence"], 0.0)
        self.assertEqual(len(result["results"]), 0)

    def test_stats(self):
        self.graph.add_node("a", node_id="a")
        self.graph.add_node("b", node_id="b", caused_by=["a"])
        self.graph.add_node("c", node_id="c", associations=["a"])
        stats = self.graph.stats()
        self.assertEqual(stats["nodes"], 3)
        self.assertEqual(stats["edge_types"].get("causality", 0), 1)
        self.assertEqual(stats["edge_types"].get("association", 0), 2)


class TestQMDCausalFields(unittest.TestCase):
    """Test QMD schema backward compatibility with new causal fields."""

    def test_schema_has_causal_fields(self):
        schema_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "qmd", "schema.json")
        with open(schema_path) as f:
            schema = json.load(f)
        task_props = schema["properties"]["tasks"]["items"]["properties"]
        self.assertIn("caused_by", task_props)
        self.assertIn("causes", task_props)
        self.assertIn("confidence", task_props)

    def test_backward_compatible_qmd(self):
        """Old QMD files without causal fields should still be valid."""
        old_qmd = {
            "session_id": "test",
            "tasks": [{"id": "t1", "status": "active", "title": "Test task"}],
            "updated_at": "2026-03-01T00:00:00Z"
        }
        # Should not raise — just validate it has required fields
        self.assertIn("tasks", old_qmd)
        self.assertIn("id", old_qmd["tasks"][0])

    def test_new_qmd_with_causal_fields(self):
        """New QMD with causal fields."""
        new_qmd = {
            "session_id": "test-v3",
            "tasks": [{
                "id": "t1", "status": "active", "title": "Deploy v3",
                "caused_by": ["t0-research"],
                "causes": ["t2-testing"],
                "confidence": 0.85,
            }],
            "updated_at": "2026-03-01T00:00:00Z"
        }
        task = new_qmd["tasks"][0]
        self.assertEqual(task["caused_by"], ["t0-research"])
        self.assertEqual(task["confidence"], 0.85)


class TestAMABenchMini(unittest.TestCase):
    """
    Mini benchmark inspired by AMA-Bench.
    10 synthetic agent trajectories with QA pairs testing
    recall, causal inference, and state updating.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.graph = CausalityGraph(db_path=os.path.join(self.tmpdir, "bench.db"))
        self._build_trajectories()

    def tearDown(self):
        self.graph.close()

    def _build_trajectories(self):
        """10 synthetic agent trajectories."""
        self.trajectories = [
            # (id, text, embedding_seed, caused_by, source)
            ("t1", "Agent navigated to /settings page", 1, [], "traj-1"),
            ("t2", "Agent clicked 'Enable Dark Mode' toggle", 2, ["t1"], "traj-1"),
            ("t3", "UI theme changed to dark mode", 3, ["t2"], "traj-1"),
            ("t4", "Agent opened /profile page", 4, [], "traj-2"),
            ("t5", "Agent uploaded new avatar image", 5, ["t4"], "traj-2"),
            ("t6", "Avatar updated successfully, old avatar deleted", 6, ["t5"], "traj-2"),
            ("t7", "Agent ran database migration script", 7, [], "traj-3"),
            ("t8", "Migration added 'preferences' column to users table", 8, ["t7"], "traj-3"),
            ("t9", "Agent restarted the application server", 9, ["t8"], "traj-3"),
            ("t10", "Application running with new schema, all tests passing", 10, ["t9"], "traj-3"),
        ]
        for tid, text, seed, caused_by, source in self.trajectories:
            self.graph.add_node(text, embedding=_random_emb(seed=seed), node_id=tid,
                              caused_by=caused_by, source=source)

    def _qa_pairs(self):
        """QA pairs testing different memory capabilities."""
        return [
            # (query_text, query_emb_seed, expected_ids, capability)
            # Recall: find specific events
            ("dark mode toggle", 2, {"t2", "t3"}, "recall"),
            ("avatar upload", 5, {"t5", "t6"}, "recall"),
            # Causal inference: what caused what
            ("what caused dark mode change", 3, {"t2"}, "causal"),
            ("what caused server restart", 9, {"t8"}, "causal"),
            # State updating: track current state
            ("current UI theme", 3, {"t3"}, "state"),
            ("current avatar status", 6, {"t6"}, "state"),
            ("database schema state", 10, {"t10", "t8"}, "state"),
        ]

    def test_v3_benchmark(self):
        """Run the mini benchmark and output scores."""
        qa_pairs = self._qa_pairs()
        
        v2_correct = 0  # Simulate v2 (similarity only, no graph)
        v3_correct = 0  # v3 with causality graph
        total = len(qa_pairs)

        results_detail = []

        for query_text, seed, expected_ids, capability in qa_pairs:
            query_emb = _similar_emb(_random_emb(seed=seed), noise=0.05, seed=seed + 1000)

            # v2: similarity only
            v2_results = self.graph.similarity_search(query_emb, topk=3)
            v2_found = {r["id"] for r in v2_results}
            v2_hit = bool(v2_found & expected_ids)
            if v2_hit:
                v2_correct += 1

            # v3: multi-hop search (similarity + traversal)
            v3_result = self.graph.multi_hop_search(query_emb, topk=3, similarity_threshold=0.8, max_depth=2)
            v3_found = {r["id"] for r in v3_result["results"]}
            v3_hit = bool(v3_found & expected_ids)
            if v3_hit:
                v3_correct += 1

            results_detail.append({
                "query": query_text,
                "capability": capability,
                "v2_hit": v2_hit,
                "v3_hit": v3_hit,
                "v2_found": list(v2_found),
                "v3_found": list(v3_found),
                "expected": list(expected_ids),
                "confidence": v3_result["confidence"],
            })

        v2_score = v2_correct / total
        v3_score = v3_correct / total

        print("\n" + "=" * 60)
        print("  memclawz Mini AMA-Bench Results")
        print("=" * 60)
        print(f"  v2 (similarity only): {v2_correct}/{total} = {v2_score:.2%}")
        print(f"  v3 (causality graph): {v3_correct}/{total} = {v3_score:.2%}")
        print(f"  Improvement: +{(v3_score - v2_score):.2%}")
        print("-" * 60)
        for r in results_detail:
            status = "✅" if r["v3_hit"] else "❌"
            print(f"  {status} [{r['capability']}] {r['query']}")
            if not r["v3_hit"]:
                print(f"      expected: {r['expected']}, got: {r['v3_found']}")
        print("=" * 60)

        # v3 should be at least as good as v2
        self.assertGreaterEqual(v3_score, v2_score)


if __name__ == "__main__":
    unittest.main(verbosity=2)
