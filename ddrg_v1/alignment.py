from __future__ import annotations

import copy
import re
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import utils as ddrg


SAFE_STOP_WORDS = {"a", "an", "the"}
NON_WORD_RE = re.compile(r"[^\w\s]")
WHITESPACE_RE = re.compile(r"\s+")


def canonicalize_claim_text(text: str) -> str:
    normalized = str(text or "").lower()
    normalized = normalized.replace("\u2019", "'").replace("\u2018", "'")
    normalized = normalized.replace("\u201c", '"').replace("\u201d", '"')
    normalized = NON_WORD_RE.sub(" ", normalized)
    tokens = [
        token
        for token in WHITESPACE_RE.sub(" ", normalized).strip().split(" ")
        if token and token not in SAFE_STOP_WORDS
    ]
    return " ".join(tokens)


def node_alignment_key(node: Dict[str, Any], graph_answer: str, graph_id: int) -> Dict[str, Any]:
    node_id = str(node.get("id", "")).strip()
    claim = str(node.get("content", "") or "").strip()
    edge_label = str(node.get("edge_label", "") or "").strip()
    canonical_claim = canonicalize_claim_text(claim)
    canonical_edge = canonicalize_claim_text(edge_label)
    token_source = canonical_claim or canonical_edge
    tokens = sorted(set(token_source.split())) if token_source else []
    return {
        "graph": graph_id,
        "node": node_id,
        "answer": ddrg.normalize_answer(graph_answer),
        "claim": claim,
        "canonical_claim": canonical_claim,
        "edge_label": edge_label,
        "canonical_edge_label": canonical_edge,
        "distance_to_ans": node.get("distance_to_ans"),
        "is_answer_node": bool(node.get("is_answer_node")) or node_id == "ANS",
        "tokens": tokens,
    }


def _node_records(support_graphs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for graph in support_graphs:
        graph_id = ddrg.parse_graph_index(graph.get("graph"))
        if graph_id is None:
            continue
        graph_answer = ddrg.normalize_answer(graph.get("predicted_answer", ""))
        for node in graph.get("nodes", []):
            record = node_alignment_key(node, graph_answer=graph_answer, graph_id=graph_id)
            if not record["node"] or record["node"] in {"N0", "Q"}:
                continue
            if record["is_answer_node"]:
                continue
            if not record["canonical_claim"] and not record["canonical_edge_label"]:
                continue
            records.append(record)
    return records


def _jaccard_similarity(left_tokens: Sequence[str], right_tokens: Sequence[str]) -> float:
    left = set(left_tokens)
    right = set(right_tokens)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _distance_similarity(left_distance: Any, right_distance: Any) -> float:
    if left_distance is None or right_distance is None:
        return 0.0
    try:
        gap = abs(int(left_distance) - int(right_distance))
    except (TypeError, ValueError):
        return 0.0
    if gap == 0:
        return 1.0
    if gap == 1:
        return 0.5
    return 0.0


def build_alignment_candidates(
    support_graphs: Sequence[Dict[str, Any]],
    min_token_overlap: float = 0.35,
) -> List[Dict[str, Any]]:
    records = _node_records(support_graphs)
    candidates: List[Dict[str, Any]] = []
    for idx, left in enumerate(records):
        for right in records[idx + 1 :]:
            if left["graph"] == right["graph"]:
                continue
            overlap = _jaccard_similarity(left["tokens"], right["tokens"])
            same_edge = bool(left["canonical_edge_label"]) and left["canonical_edge_label"] == right["canonical_edge_label"]
            same_answer = bool(left["answer"]) and left["answer"] == right["answer"]
            distance_score = _distance_similarity(left["distance_to_ans"], right["distance_to_ans"])
            exact_claim = bool(left["canonical_claim"]) and left["canonical_claim"] == right["canonical_claim"]
            if overlap < min_token_overlap and not (same_edge and overlap >= 0.20) and not exact_claim:
                continue

            score = overlap
            if same_edge:
                score += 0.15
            if same_answer:
                score += 0.05
            score += 0.10 * distance_score

            confidence = "ambiguous"
            if exact_claim or (overlap >= 0.90) or (overlap >= 0.75 and same_edge and distance_score > 0.0):
                confidence = "deterministic"
            elif overlap < min_token_overlap + 0.10 and not same_edge:
                confidence = "low"

            candidates.append(
                {
                    "left": left,
                    "right": right,
                    "similarity": round(score, 4),
                    "token_overlap": round(overlap, 4),
                    "same_edge_label": same_edge,
                    "same_answer": same_answer,
                    "distance_similarity": round(distance_score, 4),
                    "confidence": confidence,
                }
            )
    candidates.sort(key=lambda item: (item["confidence"] == "deterministic", item["similarity"]), reverse=True)
    return candidates


def _claim_from_record(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "graph": record["graph"],
        "node": record["node"],
        "answer": record["answer"],
        "claim": record["claim"],
        "distance_to_ans": record["distance_to_ans"],
    }


def deterministic_alignment_clusters(support_graphs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    candidates = build_alignment_candidates(support_graphs)
    deterministic_pairs = [item for item in candidates if item.get("confidence") == "deterministic"]
    clusters: List[Dict[str, Any]] = []
    claim_to_cluster: Dict[Tuple[int, str], int] = {}

    def can_add(cluster: Dict[str, Any], record: Dict[str, Any]) -> bool:
        return record["graph"] not in cluster["graphs"]

    for item in deterministic_pairs:
        left = item["left"]
        right = item["right"]
        left_key = (left["graph"], left["node"])
        right_key = (right["graph"], right["node"])
        left_idx = claim_to_cluster.get(left_key)
        right_idx = claim_to_cluster.get(right_key)

        if left_idx is None and right_idx is None:
            cluster = {
                "records": [left, right],
                "graphs": {left["graph"], right["graph"]},
                "similarities": [float(item["similarity"])],
            }
            clusters.append(cluster)
            cluster_idx = len(clusters) - 1
            claim_to_cluster[left_key] = cluster_idx
            claim_to_cluster[right_key] = cluster_idx
            continue

        if left_idx is not None and right_idx is None:
            cluster = clusters[left_idx]
            if can_add(cluster, right):
                cluster["records"].append(right)
                cluster["graphs"].add(right["graph"])
                cluster["similarities"].append(float(item["similarity"]))
                claim_to_cluster[right_key] = left_idx
            continue

        if right_idx is not None and left_idx is None:
            cluster = clusters[right_idx]
            if can_add(cluster, left):
                cluster["records"].append(left)
                cluster["graphs"].add(left["graph"])
                cluster["similarities"].append(float(item["similarity"]))
                claim_to_cluster[left_key] = right_idx
            continue

        if left_idx is None or right_idx is None or left_idx == right_idx:
            continue
        left_cluster = clusters[left_idx]
        right_cluster = clusters[right_idx]
        if left_cluster["graphs"] & right_cluster["graphs"]:
            continue
        merged_records = left_cluster["records"] + right_cluster["records"]
        merged_graphs = set(left_cluster["graphs"]) | set(right_cluster["graphs"])
        merged_scores = left_cluster["similarities"] + right_cluster["similarities"] + [float(item["similarity"])]
        left_cluster["records"] = merged_records
        left_cluster["graphs"] = merged_graphs
        left_cluster["similarities"] = merged_scores
        right_cluster["records"] = []
        right_cluster["graphs"] = set()
        right_cluster["similarities"] = []
        for record in merged_records:
            claim_to_cluster[(record["graph"], record["node"])] = left_idx

    alignments = []
    for idx, cluster in enumerate(clusters, start=1):
        records = cluster.get("records", [])
        if len(records) < 2:
            continue
        similarities = cluster.get("similarities", [])
        alignments.append(
            {
                "id": f"H{idx}",
                "topic": records[0].get("claim", "") or records[0].get("edge_label", ""),
                "claims": [_claim_from_record(record) for record in records],
                "source": "deterministic",
                "avg_similarity": round(sum(similarities) / len(similarities), 4) if similarities else 1.0,
            }
        )

    ambiguous_candidates = [item for item in candidates if item.get("confidence") == "ambiguous"]
    low_candidates = [item for item in candidates if item.get("confidence") == "low"]
    return {
        "parse_ok": True,
        "alignments": alignments,
        "diagnostics": {
            "candidate_count": len(candidates),
            "deterministic_candidate_count": len(deterministic_pairs),
            "ambiguous_candidate_count": len(ambiguous_candidates),
            "low_confidence_candidate_count": len(low_candidates),
            "deterministic_cluster_count": len(alignments),
        },
        "candidates": candidates,
    }


def filter_alignment_with_constraints(
    alignment: Dict[str, Any],
    support_graphs: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    lookup = {
        (ddrg.parse_graph_index(graph.get("graph")), str(node.get("id", ""))): node
        for graph in support_graphs
        if ddrg.parse_graph_index(graph.get("graph")) is not None
        for node in graph.get("nodes", [])
    }
    filtered_alignments = []
    dropped_same_graph = 0
    dropped_low_similarity = 0
    dropped_contradictory_answers = 0
    for idx, item in enumerate(alignment.get("alignments", []), start=1):
        if not isinstance(item, dict):
            continue
        seen_graphs = set()
        claims = []
        for claim in item.get("claims", []):
            if not isinstance(claim, dict):
                continue
            graph_id = ddrg.parse_graph_index(claim.get("graph"))
            node_id = str(claim.get("node", ""))
            if graph_id is None or not node_id:
                continue
            if graph_id in seen_graphs:
                dropped_same_graph += 1
                continue
            seen_graphs.add(graph_id)
            node = lookup.get((graph_id, node_id), {})
            enriched_claim = copy.deepcopy(claim)
            if not enriched_claim.get("claim"):
                enriched_claim["claim"] = str(node.get("content", "")).strip()
            if enriched_claim.get("distance_to_ans") is None:
                enriched_claim["distance_to_ans"] = node.get("distance_to_ans")
            claims.append(enriched_claim)
        if len(claims) < 2:
            continue
        is_final_only = all(str(claim.get("node", "")) == "ANS" for claim in claims)
        answer_labels = {
            ddrg.normalize_answer(claim.get("answer", ""))
            for claim in claims
            if ddrg.normalize_answer(claim.get("answer", ""))
        }
        if is_final_only and len(answer_labels) > 1:
            dropped_contradictory_answers += 1
            continue
        avg_similarity = float(item.get("avg_similarity", 1.0))
        llm_confidence = float(item.get("llm_confidence", 0.0))
        if item.get("source") == "deterministic" and avg_similarity < 0.75 and llm_confidence < 0.5:
            dropped_low_similarity += 1
            continue
        filtered_item = {
            "id": str(item.get("id") or f"A{idx}"),
            "topic": str(item.get("topic", "")).strip(),
            "claims": claims,
        }
        if item.get("source"):
            filtered_item["source"] = item.get("source")
        if "avg_similarity" in item:
            filtered_item["avg_similarity"] = avg_similarity
        filtered_alignments.append(filtered_item)

    diagnostics = dict(alignment.get("diagnostics", {}))
    diagnostics.update(
        {
            "filtered_cluster_count": len(filtered_alignments),
            "dropped_same_graph_claims": dropped_same_graph,
            "dropped_low_similarity_clusters": dropped_low_similarity,
            "dropped_contradictory_final_answer_clusters": dropped_contradictory_answers,
        }
    )
    result = dict(alignment)
    result["alignments"] = filtered_alignments
    result["diagnostics"] = diagnostics
    result["parse_ok"] = bool(alignment.get("parse_ok", False) or filtered_alignments)
    return result


def _merge_alignment_sets(
    deterministic_alignments: Sequence[Dict[str, Any]],
    llm_alignments: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    merged = [copy.deepcopy(item) for item in deterministic_alignments]
    seen_sets = {
        _alignment_claim_key(item.get("claims", []))
        for item in merged
        if _alignment_claim_key(item.get("claims", []))
    }
    for item in llm_alignments:
        claim_key = _alignment_claim_key(item.get("claims", []))
        if not claim_key or claim_key in seen_sets:
            continue
        merged_item = copy.deepcopy(item)
        merged_item.setdefault("source", "llm")
        merged.append(merged_item)
        seen_sets.add(claim_key)
    return merged


def _alignment_claim_key(claims: Sequence[Dict[str, Any]]) -> Tuple[Tuple[int, str], ...]:
    normalized_claims: List[Tuple[int, str]] = []
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        graph_id = ddrg.parse_graph_index(claim.get("graph"))
        node_id = str(claim.get("node", ""))
        if graph_id is None or not node_id:
            continue
        normalized_claims.append((graph_id, node_id))
    return tuple(sorted(normalized_claims))


def hybrid_align_support_graphs(
    client: Any,
    args: Any,
    question: str,
    support_graphs: Sequence[Dict[str, Any]],
    llm_align_fn: Optional[Callable[[], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    deterministic = deterministic_alignment_clusters(support_graphs)
    deterministic_filtered = filter_alignment_with_constraints(
        {
            "parse_ok": deterministic.get("parse_ok", False),
            "alignments": deterministic.get("alignments", []),
            "diagnostics": deterministic.get("diagnostics", {}),
        },
        support_graphs,
    )
    diagnostics = dict(deterministic_filtered.get("diagnostics", {}))
    ambiguous_candidates = int(diagnostics.get("ambiguous_candidate_count", 0))
    llm_result: Dict[str, Any] = {"parse_ok": False, "alignments": []}
    llm_used = False
    fallback_reason = ""
    if llm_align_fn is not None and (ambiguous_candidates > 0 or not deterministic_filtered.get("alignments")):
        llm_used = True
        if ambiguous_candidates > 0:
            fallback_reason = "ambiguous_candidates"
        elif deterministic.get("alignments") and not deterministic_filtered.get("alignments"):
            fallback_reason = "deterministic_filtered_empty"
        else:
            fallback_reason = "no_deterministic_alignment"
        llm_result = llm_align_fn() or {"parse_ok": False, "alignments": []}

    merged_alignments = _merge_alignment_sets(
        deterministic_alignments=deterministic_filtered.get("alignments", []),
        llm_alignments=llm_result.get("alignments", []),
    )
    merged = {
        "parse_ok": bool(deterministic_filtered.get("parse_ok") or llm_result.get("parse_ok")),
        "alignments": merged_alignments,
        "raw": {
            "deterministic": deterministic.get("alignments", []),
            "deterministic_filtered": deterministic_filtered.get("alignments", []),
            "llm": llm_result.get("alignments", []),
        },
        "raw_output": llm_result.get("raw_output", ""),
        "mode": "hybrid",
        "diagnostics": {
            **diagnostics,
            "mode": "hybrid",
            "llm_fallback_used": llm_used,
            "llm_fallback_reason": fallback_reason,
            "llm_cluster_count": len(llm_result.get("alignments", [])),
            "llm_adjudicated_groups": len(llm_result.get("alignments", [])) if llm_used else 0,
            "merged_cluster_count": len(merged_alignments),
        },
    }
    filtered = filter_alignment_with_constraints(merged, support_graphs)
    filtered["mode"] = "hybrid"
    return filtered
