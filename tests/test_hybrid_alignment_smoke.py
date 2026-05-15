import unittest

from ddrg_v1.alignment import (
    build_alignment_candidates,
    canonicalize_claim_text,
    deterministic_alignment_clusters,
    hybrid_align_support_graphs,
)


def make_support_graph(graph_id: int, answer: str, nodes: list[dict]) -> dict:
    return {
        "graph": graph_id,
        "predicted_answer": answer,
        "nodes": nodes,
    }


class HybridAlignmentSmokeTest(unittest.TestCase):
    def test_canonicalization_is_stable(self) -> None:
        left = "The total, number of apples!"
        right = "total number of apples"
        self.assertEqual(canonicalize_claim_text(left), canonicalize_claim_text(right))

    def test_deterministic_alignment_groups_similar_claims(self) -> None:
        graphs = [
            make_support_graph(
                1,
                "A",
                [
                    {"id": "N1", "content": "Total number of apples is 5.", "edge_label": "count", "distance_to_ans": 1},
                    {"id": "N2", "content": "Total number of apples is 5.", "edge_label": "count", "distance_to_ans": 2},
                ],
            ),
            make_support_graph(
                2,
                "A",
                [
                    {"id": "N8", "content": "the total number of apples is 5", "edge_label": "count", "distance_to_ans": 1},
                ],
            ),
            make_support_graph(
                3,
                "B",
                [
                    {"id": "N3", "content": "The train arrives at noon.", "edge_label": "schedule", "distance_to_ans": 1},
                ],
            ),
        ]
        result = deterministic_alignment_clusters(graphs)
        self.assertTrue(result["parse_ok"])
        self.assertGreaterEqual(result["diagnostics"]["deterministic_cluster_count"], 1)
        claims = result["alignments"][0]["claims"]
        self.assertEqual({claim["graph"] for claim in claims}, {1, 2})
        self.assertEqual(len([claim for claim in claims if claim["graph"] == 1]), 1)

    def test_same_graph_claims_are_prevented_from_forming_one_cluster(self) -> None:
        graphs = [
            make_support_graph(
                1,
                "A",
                [
                    {"id": "N1", "content": "Total number of apples is 5.", "edge_label": "count", "distance_to_ans": 1},
                    {"id": "N2", "content": "The total number of apples is 5.", "edge_label": "count", "distance_to_ans": 1},
                ],
            ),
            make_support_graph(
                2,
                "A",
                [
                    {"id": "N3", "content": "Total number of apples is 5.", "edge_label": "count", "distance_to_ans": 1},
                ],
            ),
        ]
        result = deterministic_alignment_clusters(graphs)
        self.assertEqual(len(result["alignments"]), 1)
        claims = result["alignments"][0]["claims"]
        self.assertEqual(len(claims), 2)
        self.assertEqual(len([claim for claim in claims if claim["graph"] == 1]), 1)

    def test_low_overlap_claims_are_not_aligned(self) -> None:
        graphs = [
            make_support_graph(1, "A", [{"id": "N1", "content": "Total number of apples is 5.", "edge_label": "count"}]),
            make_support_graph(2, "A", [{"id": "N2", "content": "Blue cars parked beside the lake.", "edge_label": "scene"}]),
        ]
        candidates = build_alignment_candidates(graphs)
        self.assertEqual(candidates, [])

    def test_hybrid_alignment_reports_diagnostics(self) -> None:
        graphs = [
            make_support_graph(1, "A", [{"id": "N1", "content": "Total number of apples is 5.", "edge_label": "count"}]),
            make_support_graph(2, "A", [{"id": "N2", "content": "Total number of apples is 5.", "edge_label": "count"}]),
        ]
        result = hybrid_align_support_graphs(
            client=None,
            args=None,
            question="Dummy question",
            support_graphs=graphs,
            llm_align_fn=lambda: {"parse_ok": True, "alignments": []},
        )
        self.assertTrue(result["parse_ok"])
        self.assertEqual(result["mode"], "hybrid")
        self.assertIn("diagnostics", result)
        self.assertIn("deterministic_cluster_count", result["diagnostics"])
        self.assertIn("llm_fallback_used", result["diagnostics"])

    def test_hybrid_alignment_falls_back_for_ambiguous_candidates(self) -> None:
        graphs = [
            make_support_graph(
                1,
                "A",
                [{"id": "N1", "content": "red blue green apples", "edge_label": "count", "distance_to_ans": 1}],
            ),
            make_support_graph(
                2,
                "A",
                [{"id": "N2", "content": "red blue yellow apples", "edge_label": "count", "distance_to_ans": 1}],
            ),
        ]
        llm_result = {
            "parse_ok": True,
            "alignments": [
                {
                    "id": "A1",
                    "topic": "fallback",
                    "claims": [
                        {"graph": 1, "node": "N1", "claim": "red blue green apples", "answer": "A"},
                        {"graph": 2, "node": "N2", "claim": "red blue yellow apples", "answer": "A"},
                    ],
                }
            ],
        }
        result = hybrid_align_support_graphs(
            client=None,
            args=None,
            question="Dummy question",
            support_graphs=graphs,
            llm_align_fn=lambda: llm_result,
        )
        self.assertTrue(result["diagnostics"]["llm_fallback_used"])
        self.assertEqual(result["diagnostics"]["llm_fallback_reason"], "ambiguous_candidates")
        self.assertGreaterEqual(len(result["alignments"]), 1)


if __name__ == "__main__":
    unittest.main()
