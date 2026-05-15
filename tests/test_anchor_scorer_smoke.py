import tempfile
import unittest
from pathlib import Path

from ddrg_v1.anchor_scorer import (
    FEATURE_NAMES,
    LogisticAnchorScorer,
    build_training_examples_from_row,
    load_anchor_scorer,
)
from ddrg_v1.core import select_verified_anchor


def make_graph(
    graph_id: str,
    answer: str,
    graph_score: float,
    node_count: int,
    verified_nodes: int,
    verified_edges: int,
    answer_valid: bool = True,
) -> dict:
    nodes = [
        {
            "id": "ANS" if idx == 0 else f"N{idx}",
            "parents": [f"N{idx + 1}"] if idx == 0 and node_count > 1 else ([] if idx == node_count - 1 else [f"N{idx + 1}"]),
            "repair": {"status": "verified" if idx < verified_nodes else "untouched"},
            "distance_to_ans": idx,
            "content": f"claim-{idx}",
        }
        for idx in range(node_count)
    ]
    edges = [
        {
            "source": f"N{idx + 1}",
            "target": "ANS" if idx == 0 else f"N{idx}",
            "label": "supports",
            "repair": {"status": "verified" if idx < verified_edges else "untouched"},
        }
        for idx in range(max(node_count - 1, 0))
    ]
    return {
        "graph": graph_id,
        "predicted_answer": answer,
        "graph_score": graph_score,
        "parse_ok": True,
        "issues": [],
        "nodes": nodes,
        "edges": edges,
        "omitted_parent_edges": [],
        "repaired_nodes": nodes,
        "repaired_edges": edges,
        "support_node_ids": [node["id"] for node in nodes if node["id"] != "ANS"],
        "local_claims": [{"node": node["id"], "claim": node["content"]} for node in nodes if node["id"] != "ANS"],
        "repair_summary": {
            "answer_node_valid": answer_valid,
            "invalid_node_ids": [] if answer_valid else ["ANS"],
        },
    }


class AnchorScorerSmokeTest(unittest.TestCase):
    def test_default_selection_preserves_heuristic_mode(self) -> None:
        graphs = [
            make_graph("G1", "A", graph_score=5.0, node_count=3, verified_nodes=2, verified_edges=2),
            make_graph("G2", "B", graph_score=0.2, node_count=8, verified_nodes=0, verified_edges=0),
        ]

        selected, selected_graph, info = select_verified_anchor(
            repaired_support_graphs=graphs,
            fallback_pred="C",
            probe_results=[],
        )

        self.assertEqual(selected, "A")
        self.assertEqual(selected_graph["graph"], "G1")
        self.assertEqual(info["answer_source"], "verified_anchor")
        self.assertEqual(info["score_mode"], "heuristic_verified_anchor")
        self.assertEqual(len(info["anchor_diagnostics"]), 2)

    def test_optional_learned_selection_can_override_heuristic(self) -> None:
        graphs = [
            make_graph("G1", "A", graph_score=5.0, node_count=3, verified_nodes=2, verified_edges=2),
            make_graph("G2", "B", graph_score=0.2, node_count=8, verified_nodes=0, verified_edges=0),
        ]
        weights = [0.0 for _ in FEATURE_NAMES]
        weights[FEATURE_NAMES.index("num_nodes")] = 4.0
        scorer = LogisticAnchorScorer(
            feature_names=list(FEATURE_NAMES),
            weights=weights,
            bias=0.0,
            means=[0.0 for _ in FEATURE_NAMES],
            scales=[1.0 for _ in FEATURE_NAMES],
            metadata={"model_type": "test_logreg"},
        )

        selected, selected_graph, info = select_verified_anchor(
            repaired_support_graphs=graphs,
            fallback_pred="C",
            probe_results=[],
            anchor_scorer=scorer,
        )

        self.assertEqual(selected, "B")
        self.assertEqual(selected_graph["graph"], "G2")
        self.assertEqual(info["answer_source"], "learned_anchor")
        self.assertEqual(info["score_mode"], "learned_anchor_logreg")
        self.assertEqual(
            info["selected_graph_score_components"]["mode"],
            "learned_anchor_logreg",
        )

    def test_training_row_build_save_and_load_smoke(self) -> None:
        row = {
            "id": "row-1",
            "benchmark": "logiqa",
            "question": "Dummy question",
            "gold": "B",
            "model": "dummy",
            "ddrg_v1": {
                "alignment": {
                    "alignments": [
                        {"claims": [{"answer": "B"}, {"answer": "B"}]},
                    ]
                },
                "probe_results": [{"supported": "B", "opposed": "A"}],
                "repaired_support_graphs": [
                    make_graph("G1", "B", graph_score=1.0, node_count=4, verified_nodes=2, verified_edges=2),
                    make_graph("G2", "A", graph_score=1.5, node_count=3, verified_nodes=1, verified_edges=1),
                ],
            },
        }
        examples = build_training_examples_from_row(row)
        self.assertEqual(len(examples), 2)
        self.assertEqual(sum(example.label for example in examples), 1)
        self.assertIn("heuristic_anchor_score", examples[0].features)
        self.assertIn("alignment_answer_support_sum", examples[0].features)

        scorer = LogisticAnchorScorer.fit(examples, epochs=25)
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "anchor_scorer.json"
            scorer.save(model_path)
            loaded = load_anchor_scorer(model_path)

        probability = loaded.predict_proba(examples[0].features)
        self.assertGreaterEqual(probability, 0.0)
        self.assertLessEqual(probability, 1.0)


if __name__ == "__main__":
    unittest.main()
