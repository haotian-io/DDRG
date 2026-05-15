"""Learned anchor scorer utilities for DDRG v1.

This module provides:

- graph-level feature extraction for repaired support graphs
- JSONL -> training-example conversion
- a small logistic-regression scorer implemented with NumPy
- model save/load helpers for runtime anchor selection
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from . import utils as ddrg


FEATURE_NAMES: List[str] = [
    "heuristic_anchor_score",
    "graph_score",
    "num_nodes",
    "num_edges",
    "num_repaired_nodes",
    "num_repaired_edges",
    "num_support_nodes",
    "num_omitted_parent_edges",
    "num_local_claims",
    "num_multi_parent_nodes",
    "max_parent_count",
    "mean_parent_count",
    "distance_to_ans_mean",
    "distance_to_ans_min",
    "distance_to_ans_max",
    "parse_ok",
    "issue_count",
    "answer_node_valid",
    "verified_nodes",
    "verified_edges",
    "refuted_nodes",
    "refuted_edges",
    "invalid_nodes",
    "dropped_nodes",
    "blocked_edges",
    "probe_answer_delta",
    "answer_vote_count",
    "answer_vote_share",
    "answer_is_majority",
    "supported_answer",
    "opposed_answer",
    "supported_minus_opposed",
    "alignment_cluster_count",
    "alignment_answer_support_sum",
    "alignment_answer_support_max",
]


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_mean(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    return sum(items) / len(items) if items else 0.0


def _count_status(items: Iterable[Dict[str, Any]], status: str) -> int:
    return sum(1 for item in items if item.get("repair", {}).get("status") == status)


def supported_answers_from_probes(probe_results: Sequence[Dict[str, Any]]) -> set[str]:
    return {
        ddrg.normalize_answer(result.get("supported", ""))
        for result in probe_results
        if ddrg.normalize_answer(result.get("supported", ""))
    }


def opposed_answers_from_probes(probe_results: Sequence[Dict[str, Any]]) -> set[str]:
    return {
        ddrg.normalize_answer(result.get("opposed", ""))
        for result in probe_results
        if ddrg.normalize_answer(result.get("opposed", ""))
    }


def answer_probe_delta(answer: str, probe_results: Sequence[Dict[str, Any]]) -> int:
    normalized = ddrg.normalize_answer(answer)
    delta = 0
    for result in probe_results:
        if normalized and normalized == ddrg.normalize_answer(result.get("supported", "")):
            delta += 1
        if normalized and normalized == ddrg.normalize_answer(result.get("opposed", "")):
            delta -= 1
    return delta


def heuristic_anchor_score(
    graph: Dict[str, Any],
    probe_results: Sequence[Dict[str, Any]],
) -> Tuple[float, Dict[str, Any]]:
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", []) + graph.get("omitted_parent_edges", [])
    summary = graph.get("repair_summary", {})
    verified_nodes = _count_status(nodes, "verified")
    verified_edges = _count_status(edges, "verified")
    refuted_nodes = _count_status(nodes, "refuted")
    refuted_edges = _count_status(edges, "refuted")
    invalid_nodes = len(summary.get("invalid_node_ids", []))
    valid_ans_bonus = 2.0 if summary.get("answer_node_valid", False) else -3.0
    probe_delta = answer_probe_delta(graph.get("predicted_answer", ""), probe_results)
    score = (
        _safe_float(graph.get("graph_score", 0.0))
        + 0.60 * verified_nodes
        + 0.35 * verified_edges
        - 1.25 * refuted_nodes
        - 0.85 * refuted_edges
        - 0.50 * invalid_nodes
        + valid_ans_bonus
        + 1.00 * probe_delta
    )
    return score, {
        "base_graph_score": _safe_float(graph.get("graph_score", 0.0)),
        "verified_nodes": verified_nodes,
        "verified_edges": verified_edges,
        "refuted_nodes": refuted_nodes,
        "refuted_edges": refuted_edges,
        "invalid_nodes": invalid_nodes,
        "valid_ans_bonus": valid_ans_bonus,
        "probe_answer_delta": probe_delta,
        "score": score,
    }


def alignment_answer_support(
    answer: str,
    alignment: Optional[Dict[str, Any]],
) -> Tuple[int, int, int]:
    normalized = ddrg.normalize_answer(answer)
    if not normalized or not isinstance(alignment, dict):
        return 0, 0, 0
    counts: List[int] = []
    for item in alignment.get("alignments", []):
        if not isinstance(item, dict):
            continue
        cluster_count = sum(
            1
            for claim in item.get("claims", [])
            if isinstance(claim, dict) and ddrg.normalize_answer(claim.get("answer", "")) == normalized
        )
        if cluster_count:
            counts.append(cluster_count)
    return len(counts), sum(counts), max(counts, default=0)


def extract_graph_features(
    graph: Dict[str, Any],
    all_graphs: Sequence[Dict[str, Any]],
    probe_results: Sequence[Dict[str, Any]],
    alignment: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    nodes = list(graph.get("nodes", []))
    edges = list(graph.get("edges", []))
    all_edges = edges + list(graph.get("omitted_parent_edges", []))
    repaired_nodes = list(graph.get("repaired_nodes", []))
    repaired_edges = list(graph.get("repaired_edges", []))
    summary = graph.get("repair_summary", {})

    normalized_answer = ddrg.normalize_answer(graph.get("predicted_answer", ""))
    answer_counts = Counter(
        ddrg.normalize_answer(item.get("predicted_answer", ""))
        for item in all_graphs
        if ddrg.normalize_answer(item.get("predicted_answer", ""))
    )
    answer_vote_count = int(answer_counts.get(normalized_answer, 0))
    total_answer_votes = sum(answer_counts.values())
    majority_count = max(answer_counts.values(), default=0)

    non_root_nodes = [node for node in nodes if node.get("id") not in {"N0", "Q"}]
    claim_distances = [
        _safe_float(node.get("distance_to_ans"))
        for node in non_root_nodes
        if node.get("distance_to_ans") is not None
    ]
    parent_counts = [len(node.get("parents", [])) for node in non_root_nodes]

    heuristic_score, _ = heuristic_anchor_score(graph, probe_results)
    supported_answers = supported_answers_from_probes(probe_results)
    opposed_answers = opposed_answers_from_probes(probe_results)
    alignment_cluster_count, alignment_support_sum, alignment_support_max = alignment_answer_support(
        normalized_answer,
        alignment,
    )

    features = {
        "heuristic_anchor_score": heuristic_score,
        "graph_score": _safe_float(graph.get("graph_score", 0.0)),
        "num_nodes": float(len(nodes)),
        "num_edges": float(len(edges)),
        "num_repaired_nodes": float(len(repaired_nodes)),
        "num_repaired_edges": float(len(repaired_edges)),
        "num_support_nodes": float(len(graph.get("support_node_ids", []))),
        "num_omitted_parent_edges": float(len(graph.get("omitted_parent_edges", []))),
        "num_local_claims": float(len(graph.get("local_claims", []))),
        "num_multi_parent_nodes": float(sum(1 for count in parent_counts if count > 1)),
        "max_parent_count": float(max(parent_counts, default=0)),
        "mean_parent_count": _safe_mean(parent_counts),
        "distance_to_ans_mean": _safe_mean(claim_distances),
        "distance_to_ans_min": float(min(claim_distances, default=0.0)),
        "distance_to_ans_max": float(max(claim_distances, default=0.0)),
        "parse_ok": 1.0 if graph.get("parse_ok") else 0.0,
        "issue_count": float(len(graph.get("issues", []))),
        "answer_node_valid": 1.0 if summary.get("answer_node_valid", False) else 0.0,
        "verified_nodes": float(_count_status(nodes, "verified")),
        "verified_edges": float(_count_status(all_edges, "verified")),
        "refuted_nodes": float(_count_status(nodes, "refuted")),
        "refuted_edges": float(_count_status(all_edges, "refuted")),
        "invalid_nodes": float(len(summary.get("invalid_node_ids", []))),
        "dropped_nodes": float(_count_status(nodes, "dropped")),
        "blocked_edges": float(_count_status(all_edges, "blocked")),
        "probe_answer_delta": float(answer_probe_delta(normalized_answer, probe_results)),
        "answer_vote_count": float(answer_vote_count),
        "answer_vote_share": float(answer_vote_count / total_answer_votes) if total_answer_votes else 0.0,
        "answer_is_majority": 1.0 if answer_vote_count and answer_vote_count == majority_count else 0.0,
        "supported_answer": 1.0 if normalized_answer in supported_answers else 0.0,
        "opposed_answer": 1.0 if normalized_answer in opposed_answers else 0.0,
        "supported_minus_opposed": float(
            (1 if normalized_answer in supported_answers else 0)
            - (1 if normalized_answer in opposed_answers else 0)
        ),
        "alignment_cluster_count": float(alignment_cluster_count),
        "alignment_answer_support_sum": float(alignment_support_sum),
        "alignment_answer_support_max": float(alignment_support_max),
    }
    return {name: float(features.get(name, 0.0)) for name in FEATURE_NAMES}


@dataclass
class AnchorTrainingExample:
    example_id: str
    graph_index: int
    gold_answer: str
    graph_answer: str
    label: int
    features: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


def graph_label(graph: Dict[str, Any], gold_answer: Any) -> int:
    summary = graph.get("repair_summary", {})
    answer = ddrg.normalize_answer(graph.get("predicted_answer", ""))
    gold = ddrg.normalize_answer(gold_answer)
    if not answer or not gold:
        return 0
    if not summary.get("answer_node_valid", False):
        return 0
    return int(ddrg.answers_match(answer, gold))


def build_training_examples_from_row(row: Dict[str, Any]) -> List[AnchorTrainingExample]:
    info = row.get("ddrg_v1", {})
    graphs = info.get("repaired_support_graphs", [])
    if not isinstance(graphs, list) or not graphs:
        return []
    gold = row.get("gold")
    gold_normalized = ddrg.normalize_answer(gold)
    if not gold_normalized:
        return []
    alignment = info.get("alignment", {})
    probe_results = info.get("probe_results", [])
    example_id = f"{row.get('benchmark', 'unknown')}::{row.get('id', '')}"

    examples: List[AnchorTrainingExample] = []
    for graph in graphs:
        graph_index = int(ddrg.parse_graph_index(graph.get("graph")) or 0)
        features = extract_graph_features(
            graph=graph,
            all_graphs=graphs,
            probe_results=probe_results,
            alignment=alignment,
        )
        examples.append(
            AnchorTrainingExample(
                example_id=example_id,
                graph_index=graph_index,
                gold_answer=gold_normalized,
                graph_answer=ddrg.normalize_answer(graph.get("predicted_answer", "")),
                label=graph_label(graph, gold_normalized),
                features=features,
                metadata={
                    "benchmark": row.get("benchmark", ""),
                    "row_id": row.get("id", ""),
                    "model": row.get("model", ""),
                    "question": row.get("question", ""),
                    "source_file": row.get("_source_file", ""),
                },
            )
        )
    return examples


def load_result_rows(paths: Sequence[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                row["_source_file"] = str(path)
                rows.append(row)
    return rows


def build_training_examples(rows: Sequence[Dict[str, Any]]) -> List[AnchorTrainingExample]:
    examples: List[AnchorTrainingExample] = []
    for row in rows:
        examples.extend(build_training_examples_from_row(row))
    return examples


def feature_rows_from_examples(examples: Sequence[AnchorTrainingExample]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in examples:
        row: Dict[str, Any] = {
            "example_id": item.example_id,
            "graph_index": item.graph_index,
            "gold_answer": item.gold_answer,
            "graph_answer": item.graph_answer,
            "label": item.label,
        }
        row.update(item.metadata)
        row.update(item.features)
        rows.append(row)
    return rows


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


@dataclass
class LogisticAnchorScorer:
    feature_names: List[str]
    weights: List[float]
    bias: float
    means: List[float]
    scales: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def fit(
        cls,
        examples: Sequence[AnchorTrainingExample],
        learning_rate: float = 0.05,
        epochs: int = 400,
        l2: float = 1e-3,
    ) -> "LogisticAnchorScorer":
        if not examples:
            raise ValueError("No training examples available for anchor scorer.")

        feature_names = list(FEATURE_NAMES)
        X = np.array(
            [[example.features.get(name, 0.0) for name in feature_names] for example in examples],
            dtype=float,
        )
        y = np.array([float(example.label) for example in examples], dtype=float)

        means = X.mean(axis=0)
        scales = X.std(axis=0)
        scales = np.where(scales < 1e-8, 1.0, scales)
        Xn = (X - means) / scales

        weights = np.zeros(Xn.shape[1], dtype=float)
        bias = 0.0

        positive = max(float(y.sum()), 1.0)
        negative = max(float(len(y) - y.sum()), 1.0)
        pos_weight = negative / positive
        sample_weights = np.where(y > 0.5, pos_weight, 1.0)

        for _ in range(max(int(epochs), 1)):
            logits = Xn @ weights + bias
            probs = sigmoid(logits)
            errors = (probs - y) * sample_weights
            grad_w = (Xn.T @ errors) / len(Xn) + l2 * weights
            grad_b = float(errors.mean())
            weights -= float(learning_rate) * grad_w
            bias -= float(learning_rate) * grad_b

        metadata = {
            "train_examples": len(examples),
            "positive_examples": int(y.sum()),
            "negative_examples": int(len(y) - y.sum()),
            "learning_rate": learning_rate,
            "epochs": epochs,
            "l2": l2,
            "model_type": "logistic_regression_numpy",
        }
        return cls(
            feature_names=feature_names,
            weights=weights.tolist(),
            bias=float(bias),
            means=means.tolist(),
            scales=scales.tolist(),
            metadata=metadata,
        )

    def _prepare_vector(self, feature_map: Dict[str, float]) -> np.ndarray:
        vector = np.array([float(feature_map.get(name, 0.0)) for name in self.feature_names], dtype=float)
        means = np.array(self.means, dtype=float)
        scales = np.array(self.scales, dtype=float)
        return (vector - means) / scales

    def predict_logit(self, feature_map: Dict[str, float]) -> float:
        vector = self._prepare_vector(feature_map)
        weights = np.array(self.weights, dtype=float)
        return float(vector @ weights + float(self.bias))

    def predict_proba(self, feature_map: Dict[str, float]) -> float:
        return float(sigmoid(np.array([self.predict_logit(feature_map)], dtype=float))[0])

    def score_graph(
        self,
        graph: Dict[str, Any],
        all_graphs: Sequence[Dict[str, Any]],
        probe_results: Sequence[Dict[str, Any]],
        alignment: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        features = extract_graph_features(
            graph=graph,
            all_graphs=all_graphs,
            probe_results=probe_results,
            alignment=alignment,
        )
        logit = self.predict_logit(features)
        probability = self.predict_proba(features)
        return probability, {
            "mode": "learned_anchor_logreg",
            "probability": probability,
            "logit": logit,
            "features": features,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "ddrg_anchor_scorer_v1",
            "feature_names": self.feature_names,
            "weights": self.weights,
            "bias": self.bias,
            "means": self.means,
            "scales": self.scales,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "LogisticAnchorScorer":
        return cls(
            feature_names=list(payload.get("feature_names", [])),
            weights=[float(value) for value in payload.get("weights", [])],
            bias=float(payload.get("bias", 0.0)),
            means=[float(value) for value in payload.get("means", [])],
            scales=[float(value) for value in payload.get("scales", [])],
            metadata=dict(payload.get("metadata", {})),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "LogisticAnchorScorer":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(payload)


def load_anchor_scorer(path: str | Path) -> LogisticAnchorScorer:
    return LogisticAnchorScorer.load(Path(path).resolve())


def grouped_anchor_accuracy(
    examples: Sequence[AnchorTrainingExample],
    scorer: LogisticAnchorScorer,
) -> float:
    by_example: Dict[str, List[AnchorTrainingExample]] = {}
    for item in examples:
        by_example.setdefault(item.example_id, []).append(item)
    if not by_example:
        return math.nan

    correct = 0
    total = 0
    for group in by_example.values():
        pool = [
            item
            for item in group
            if item.features.get("answer_node_valid", 0.0) > 0.5 and item.graph_answer
        ] or list(group)
        scored = []
        for item in pool:
            probability = scorer.predict_proba(item.features)
            scored.append((probability, item))
        if not scored:
            continue
        selected = max(scored, key=lambda pair: pair[0])[1]
        correct += int(selected.label == 1)
        total += 1
    return correct / total if total else math.nan


def grouped_heuristic_anchor_accuracy(examples: Sequence[AnchorTrainingExample]) -> float:
    by_example: Dict[str, List[AnchorTrainingExample]] = {}
    for item in examples:
        by_example.setdefault(item.example_id, []).append(item)
    if not by_example:
        return math.nan

    correct = 0
    total = 0
    for group in by_example.values():
        pool = [
            item
            for item in group
            if item.features.get("answer_node_valid", 0.0) > 0.5 and item.graph_answer
        ] or list(group)
        selected = max(pool, key=lambda item: item.features.get("heuristic_anchor_score", 0.0))
        correct += int(selected.label == 1)
        total += 1
    return correct / total if total else math.nan
