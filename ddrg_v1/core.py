import argparse
import copy
import json
import re
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterable, List, Optional, Tuple

from . import utils as ddrg
from .alignment import hybrid_align_support_graphs
from .anchor_scorer import LogisticAnchorScorer, heuristic_anchor_score as learned_heuristic_anchor_score
from .llm import LLMClient, prompt_with_problem
from .prompts import ALIGNMENT_PROMPT, ELRG_PROMPT, FRONTIER_PROBE_PROMPT, PROBE_PROMPT


def extract_options(question: str) -> Dict[str, str]:
    options: Dict[str, str] = {}
    for line in question.splitlines():
        match = re.match(r"\s*([A-E])\s*[\.\)]\s*(.+?)\s*$", line, flags=re.I)
        if match:
            options[match.group(1).upper()] = match.group(2).strip()
    return options


def normalize_answer_for_question(answer: Any, question: str) -> str:
    options = extract_options(question)
    normalized = ddrg.normalize_answer(answer)
    if not options:
        return normalized
    if normalized in options:
        return normalized
    answer_text = str(answer).strip().lower().strip(". ")
    for label, option_text in options.items():
        if normalized == ddrg.normalize_answer(option_text):
            return label
        if answer_text and answer_text == option_text.strip().lower().strip(". "):
            return label
    return normalized


def score_graph(parsed: ddrg.ParsedGraph, max_nodes: int) -> float:
    score = ddrg.score_graph(parsed, max_nodes)
    non_root_nodes = [
        node
        for node_id, node in parsed.nodes.items()
        if node_id not in {"N0", "Q", "ANS"}
    ]
    multi_parent_nodes = [node for node in parsed.nodes.values() if len(node.parents) > 1]
    root_children = [
        node
        for node_id, node in parsed.nodes.items()
        if node_id not in {"N0", "Q"} and any(parent in {"N0", "Q"} for parent in node.parents)
    ]
    if multi_parent_nodes:
        score += 0.35
    if len(root_children) >= 2:
        score += 0.20
    if len(non_root_nodes) >= 2 and not multi_parent_nodes:
        score -= 0.20
    return score


def sample_candidate(
    client: LLMClient,
    args: argparse.Namespace,
    question: str,
) -> Tuple[str, ddrg.Candidate]:
    output = client.generate(
        content=prompt_with_problem(ELRG_PROMPT, question),
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )
    parsed = ddrg.parse_elrg(output)
    parsed.final_answer = normalize_answer_for_question(
        parsed.final_answer or parsed.raw_final_answer,
        question,
    )
    return output, ddrg.Candidate(parsed=parsed, score=score_graph(parsed, args.max_graph_nodes))


def sample_graphs(
    client: LLMClient,
    args: argparse.Namespace,
    question: str,
) -> Tuple[List[str], List[ddrg.Candidate]]:
    k = max(1, int(args.k))
    workers = max(1, min(int(args.max_workers), k))
    if workers <= 1 or k <= 1:
        results = [sample_candidate(client, args, question) for _ in range(k)]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(lambda _: sample_candidate(client, args, question), range(k)))
    return [output for output, _ in results], [candidate for _, candidate in results]


def normalize_existing_candidates(
    candidates: List[ddrg.Candidate],
    question: str,
) -> List[ddrg.Candidate]:
    for candidate in candidates:
        candidate.parsed.final_answer = normalize_answer_for_question(
            candidate.parsed.final_answer or candidate.parsed.raw_final_answer,
            question,
        )
    return candidates


def majority_answer(candidates: Iterable[ddrg.Candidate], question: str) -> str:
    return ddrg.majority_vote(
        normalize_answer_for_question(candidate.parsed.final_answer, question)
        for candidate in candidates
    )


def edge_key(edge: Dict[str, Any]) -> str:
    return f"{edge.get('source', '')}->{edge.get('target', '')}"


def all_graph_edges(graph: Dict[str, Any]) -> List[Dict[str, Any]]:
    return graph.get("edges", []) + graph.get("omitted_parent_edges", [])


def compute_distance_to_ans(graph: Dict[str, Any]) -> Dict[str, int]:
    parents_by_target: Dict[str, List[str]] = {}
    for edge in all_graph_edges(graph):
        source = str(edge.get("source", ""))
        target = str(edge.get("target", ""))
        if source and target:
            parents_by_target.setdefault(target, []).append(source)

    distances: Dict[str, int] = {"ANS": 0}
    queue: deque[str] = deque(["ANS"])
    while queue:
        node_id = queue.popleft()
        for parent in parents_by_target.get(node_id, []):
            if parent in distances:
                continue
            distances[parent] = distances[node_id] + 1
            queue.append(parent)
    return distances


def enrich_support_graph(graph: Dict[str, Any]) -> Dict[str, Any]:
    enriched = copy.deepcopy(graph)
    distances = compute_distance_to_ans(enriched)
    for node in enriched.get("nodes", []):
        node["distance_to_ans"] = distances.get(str(node.get("id", "")))
    for claim in enriched.get("local_claims", []):
        claim["distance_to_ans"] = distances.get(str(claim.get("node", "")))
    return enriched


def support_graphs_from_candidates(
    candidates: List[ddrg.Candidate],
    max_support_nodes: int,
    question: str,
) -> List[Dict[str, Any]]:
    graphs = []
    for idx, candidate in enumerate(candidates, start=1):
        graph = ddrg.extract_support_graph(idx, candidate, max_support_nodes)
        graph["predicted_answer"] = normalize_answer_for_question(
            graph.get("predicted_answer", ""),
            question,
        )
        graphs.append(enrich_support_graph(graph))
    return graphs


def node_lookup(support_graphs: List[Dict[str, Any]]) -> Dict[Tuple[int, str], Dict[str, Any]]:
    lookup: Dict[Tuple[int, str], Dict[str, Any]] = {}
    for graph in support_graphs:
        graph_id = ddrg.parse_graph_index(graph.get("graph"))
        if graph_id is None:
            continue
        for node in graph.get("nodes", []):
            node_id = str(node.get("id", ""))
            if node_id:
                lookup[(graph_id, node_id)] = node
    return lookup


def compact_graph_for_prompt(graph: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "graph": graph.get("graph"),
        "answer": graph.get("predicted_answer", ""),
        "nodes": [
            {
                "id": node.get("id", ""),
                "parents": node.get("parents", []),
                "edge": node.get("edge_label", ""),
                "distance_to_ans": node.get("distance_to_ans"),
                "claim": node.get("content", ""),
            }
            for node in graph.get("nodes", [])
        ],
    }


def make_support_graph_samples(
    support_graphs: List[Dict[str, Any]],
    max_graph_chars: int,
) -> List[str]:
    samples = []
    for graph in support_graphs:
        payload = compact_graph_for_prompt(graph)
        samples.append(
            "Parsed support graph "
            + str(graph.get("graph"))
            + "\n"
            + ddrg.truncate_text(json.dumps(payload, ensure_ascii=False, indent=2), max_graph_chars)
        )
    return samples


def parse_alignment_output(
    output: str,
    support_graphs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    data = ddrg.extract_json_object(output)
    if data is None:
        return {"parse_ok": False, "raw": output, "alignments": []}

    nodes = node_lookup(support_graphs)
    alignments = []
    for idx, item in enumerate(data.get("alignments", [])):
        if not isinstance(item, dict):
            continue
        claims = []
        for claim in item.get("claims", []):
            if not isinstance(claim, dict):
                continue
            target = ddrg.normalize_repair_target(claim)
            if not target:
                continue
            graph_id = target["graph"]
            node_id = target.get("node", "")
            source_node = nodes.get((graph_id, node_id), {})
            claims.append(
                {
                    "graph": graph_id,
                    "node": node_id,
                    "answer": ddrg.normalize_answer(claim.get("answer", "")),
                    "claim": str(claim.get("claim") or source_node.get("content", "")).strip(),
                    "distance_to_ans": source_node.get("distance_to_ans"),
                }
            )
        if claims:
            alignments.append(
                {
                    "id": str(item.get("id") or f"A{idx + 1}").strip(),
                    "topic": str(item.get("topic") or "").strip(),
                    "claims": claims,
                }
            )
    return {"parse_ok": True, "alignments": alignments, "raw": data, "raw_output": output}


def align_support_graphs(
    client: LLMClient,
    args: argparse.Namespace,
    question: str,
    support_graphs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    samples = make_support_graph_samples(support_graphs, args.max_graph_chars)
    content = (
        f"{ALIGNMENT_PROMPT}\n\n"
        f"Question:\n{question}\n\n"
        f"Parsed support graphs:\n\n"
        + "\n\n---\n\n".join(samples)
    )
    output = client.generate(
        content=content,
        model=args.judge_model or args.model,
        temperature=args.judge_temperature,
        top_p=args.top_p,
        max_tokens=args.builder_max_tokens,
    )
    parsed = parse_alignment_output(output, support_graphs)
    parsed["mode"] = "llm"
    parsed["diagnostics"] = {
        "mode": "llm",
        "input_graph_count": len(support_graphs),
        "returned_cluster_count": len(parsed.get("alignments", [])),
        "parse_ok": parsed.get("parse_ok", False),
    }
    return parsed


def build_meta_graph(
    support_graphs: List[Dict[str, Any]],
    alignment: Dict[str, Any],
) -> Dict[str, Any]:
    answers = [
        ddrg.normalize_answer(graph.get("predicted_answer", ""))
        for graph in support_graphs
        if ddrg.normalize_answer(graph.get("predicted_answer", ""))
    ]
    return {
        "type": "aligned_meta_graph",
        "graph_count": len(support_graphs),
        "answer_counts": dict(Counter(answers)),
        "alignments": alignment.get("alignments", []),
    }


def normalize_conflict_side(item: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(item, dict):
        return None
    target = ddrg.normalize_repair_target(item)
    if not target:
        return None
    target_type = str(item.get("target_type") or item.get("kind") or "claim").strip().lower()
    if target_type not in {"claim", "node", "edge"}:
        target_type = "claim"
    target["target_type"] = "claim" if target_type == "node" else target_type
    target["answer"] = ddrg.normalize_answer(item.get("answer", ""))
    target["claim"] = str(item.get("claim", "")).strip()
    return target


def parse_conflict_output(output: str) -> Dict[str, Any]:
    data = ddrg.extract_json_object(output)
    if data is None:
        return {"parse_ok": False, "raw": output, "conflicts": []}

    conflicts = []
    for idx, item in enumerate(data.get("conflicts", [])):
        if not isinstance(item, dict):
            continue
        left = normalize_conflict_side(item.get("left"))
        right = normalize_conflict_side(item.get("right"))
        probe_item = item.get("probe", {})
        if not isinstance(probe_item, dict):
            probe_item = {}
        question = str(probe_item.get("question") or item.get("question") or "").strip()
        if not left or not right or not question:
            continue
        yes_verifies = ddrg.normalize_side_name(probe_item.get("yes_verifies") or "left") or "left"
        no_verifies = ddrg.normalize_side_name(probe_item.get("no_verifies") or "right") or "right"
        conflict_id = str(item.get("id") or f"C{idx + 1}").strip()
        conflicts.append(
            {
                "id": conflict_id,
                "alignment": str(item.get("alignment") or "").strip(),
                "left": left,
                "right": right,
                "issue": str(item.get("issue", "")).strip(),
                "probe": {
                    "id": str(probe_item.get("id") or f"P{idx + 1}").strip(),
                    "conflict": conflict_id,
                    "question": question,
                    "yes_verifies": yes_verifies,
                    "no_verifies": no_verifies,
                },
            }
        )
    return {"parse_ok": True, "conflicts": conflicts, "raw": data, "raw_output": output}


def localize_frontier_conflicts(
    client: LLMClient,
    args: argparse.Namespace,
    question: str,
    support_graphs: List[Dict[str, Any]],
    meta_graph: Dict[str, Any],
) -> Dict[str, Any]:
    samples = make_support_graph_samples(support_graphs, args.max_graph_chars)
    content = (
        f"{FRONTIER_PROBE_PROMPT}\n\n"
        f"Question:\n{question}\n\n"
        f"Parsed support graphs:\n\n"
        + "\n\n---\n\n".join(samples)
        + "\n\nAligned meta-graph:\n"
        + ddrg.truncate_text(
            json.dumps(meta_graph, ensure_ascii=False, indent=2),
            args.integration_max_chars,
        )
    )
    output = client.generate(
        content=content,
        model=args.judge_model or args.model,
        temperature=args.judge_temperature,
        top_p=args.top_p,
        max_tokens=args.builder_max_tokens,
    )
    return parse_conflict_output(output)


def verify_frontier_conflicts(
    client: LLMClient,
    args: argparse.Namespace,
    question: str,
    conflicts: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    probe_results = []
    for conflict in conflicts[: args.max_probes]:
        probe = conflict["probe"]
        verdict_records = []
        verdict_outputs = []
        for _ in range(args.probe_votes):
            content = (
                f"{PROBE_PROMPT}\n\n"
                f"Original question:\n{question}\n\n"
                f"Probe question:\n{probe['question']}"
            )
            output = client.generate(
                content=content,
                model=args.judge_model or args.model,
                temperature=args.probe_temperature,
                top_p=args.top_p,
                max_tokens=args.probe_max_tokens,
            )
            verdict_outputs.append(output)
            verdict_records.append(ddrg.parse_probe_verdict(output))

        verdict = ddrg.majority_verdict([record["verdict"] for record in verdict_records])
        verified_side = ""
        refuted_side = ""
        if verdict == "YES":
            verified_side = probe.get("yes_verifies", "")
        elif verdict == "NO":
            verified_side = probe.get("no_verifies", "")
        if verified_side in {"left", "right"}:
            refuted_side = "right" if verified_side == "left" else "left"
        verified_targets = [conflict[verified_side]] if verified_side else []
        refuted_targets = [conflict[refuted_side]] if refuted_side else []
        probe_results.append(
            {
                "conflict": conflict,
                "probe": probe,
                "verdict": verdict,
                "supported": verified_targets[0].get("answer", "") if verified_targets else "",
                "opposed": refuted_targets[0].get("answer", "") if refuted_targets else "",
                "verified_side": verified_side,
                "refuted_side": refuted_side,
                "verified_targets": verified_targets,
                "refuted_targets": refuted_targets,
                "verdict_records": verdict_records,
                "verdict_outputs": verdict_outputs,
            }
        )
    return probe_results


def fresh_repair_state() -> Dict[str, Any]:
    return {"verified": 0, "refuted": 0, "status": "untouched", "invalid": False, "probes": []}


def ensure_repair_state(item: Dict[str, Any]) -> Dict[str, Any]:
    if "repair" not in item or not isinstance(item["repair"], dict):
        item["repair"] = fresh_repair_state()
    return item["repair"]


def apply_repair_state(
    item: Dict[str, Any],
    action: str,
    probe_id: str,
    verdict: str,
    answer: str,
) -> None:
    state = ensure_repair_state(item)
    state[action] = int(state.get(action, 0)) + 1
    state["status"] = ddrg.repair_status(state)
    state["probes"].append({"id": probe_id, "action": action, "verdict": verdict, "answer": answer})


def init_repaired_support_graphs(support_graphs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    repaired = copy.deepcopy(support_graphs)
    for graph in repaired:
        for node in graph.get("nodes", []):
            node["repair"] = fresh_repair_state()
        for edge in all_graph_edges(graph):
            edge["repair"] = fresh_repair_state()
    return repaired


def apply_repair_target(
    graph: Dict[str, Any],
    target: Dict[str, Any],
    action: str,
    probe_id: str,
    verdict: str,
    answer: str,
) -> None:
    if target.get("target_type") == "edge" and target.get("edge"):
        edge_target = target.get("edge") or {}
        for edge in all_graph_edges(graph):
            if edge.get("source") == edge_target.get("source") and edge.get("target") == edge_target.get("target"):
                apply_repair_state(edge, action, probe_id, verdict, answer)
        return

    node_id = target.get("node")
    node_by_id = {node.get("id"): node for node in graph.get("nodes", [])}
    if node_id in node_by_id:
        apply_repair_state(node_by_id[node_id], action, probe_id, verdict, answer)


def finalize_repaired_support_graphs(repaired_support_graphs: List[Dict[str, Any]]) -> None:
    for graph in repaired_support_graphs:
        nodes = graph.get("nodes", [])
        edges = all_graph_edges(graph)
        for node in nodes:
            node["repair"]["status"] = ddrg.repair_status(node["repair"])
        for edge in edges:
            edge["repair"]["status"] = ddrg.repair_status(edge["repair"])

        incoming_by_target: Dict[str, List[Dict[str, Any]]] = {}
        for edge in edges:
            incoming_by_target.setdefault(str(edge.get("target", "")), []).append(edge)

        invalid_nodes = {
            node.get("id")
            for node in nodes
            if node.get("repair", {}).get("status") == "refuted"
        }
        invalid_nodes.discard(None)

        changed = True
        while changed:
            changed = False
            for node in nodes:
                node_id = node.get("id")
                if not node_id or node_id in {"N0", "Q"} or node_id in invalid_nodes:
                    continue
                incoming = incoming_by_target.get(str(node_id), [])
                if not incoming:
                    continue
                valid_incoming = [
                    edge
                    for edge in incoming
                    if edge.get("repair", {}).get("status") != "refuted"
                    and edge.get("source") not in invalid_nodes
                ]
                if not valid_incoming:
                    invalid_nodes.add(node_id)
                    changed = True

        for node in nodes:
            node_id = node.get("id")
            state = node["repair"]
            state["invalid"] = node_id in invalid_nodes
            if state["invalid"] and state["status"] != "refuted":
                state["status"] = "dropped"

        for edge in edges:
            state = edge["repair"]
            if state["status"] == "refuted":
                state["invalid"] = True
            elif edge.get("source") in invalid_nodes and state["status"] != "refuted":
                state["status"] = "blocked"
                state["invalid"] = True
            elif edge.get("target") in invalid_nodes:
                state["invalid"] = True
            else:
                state["invalid"] = False

        valid_node_ids = {
            node.get("id")
            for node in nodes
            if node.get("id") and node.get("id") not in invalid_nodes
        }
        graph["repaired_nodes"] = [node for node in nodes if node.get("id") in valid_node_ids]
        graph["repaired_edges"] = [
            edge
            for edge in edges
            if not edge.get("repair", {}).get("invalid", False)
            and edge.get("source") in valid_node_ids
            and edge.get("target") in valid_node_ids
        ]
        graph["repair_summary"] = {
            "answer_node_valid": "ANS" in valid_node_ids,
            "invalid_node_ids": sorted(invalid_nodes),
            "verified_node_ids": sorted(
                node.get("id")
                for node in nodes
                if node.get("id") and node.get("repair", {}).get("status") == "verified"
            ),
            "refuted_node_ids": sorted(
                node.get("id")
                for node in nodes
                if node.get("id") and node.get("repair", {}).get("status") == "refuted"
            ),
            "verified_edge_ids": sorted(
                edge_key(edge)
                for edge in edges
                if edge.get("repair", {}).get("status") == "verified"
            ),
            "refuted_edge_ids": sorted(
                edge_key(edge)
                for edge in edges
                if edge.get("repair", {}).get("status") == "refuted"
            ),
        }


def repair_support_graphs(
    support_graphs: List[Dict[str, Any]],
    probe_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    repaired = init_repaired_support_graphs(support_graphs)
    graph_by_id = {
        ddrg.parse_graph_index(graph.get("graph")): graph
        for graph in repaired
        if ddrg.parse_graph_index(graph.get("graph")) is not None
    }
    for result in probe_results:
        probe = result.get("probe", {})
        verdict = result.get("verdict", "")
        for target in ddrg.unique_targets(result.get("verified_targets", [])):
            graph = graph_by_id.get(target.get("graph"))
            if graph:
                apply_repair_target(graph, target, "verified", probe.get("id", ""), verdict, result.get("supported", ""))
        for target in ddrg.unique_targets(result.get("refuted_targets", [])):
            graph = graph_by_id.get(target.get("graph"))
            if graph:
                apply_repair_target(graph, target, "refuted", probe.get("id", ""), verdict, result.get("opposed", ""))
    finalize_repaired_support_graphs(repaired)
    return repaired


def count_nodes_with_status(graph: Dict[str, Any], status: str) -> int:
    return sum(1 for node in graph.get("nodes", []) if node.get("repair", {}).get("status") == status)


def count_edges_with_status(graph: Dict[str, Any], status: str) -> int:
    return sum(1 for edge in all_graph_edges(graph) if edge.get("repair", {}).get("status") == status)


def answer_probe_delta(answer: str, probe_results: List[Dict[str, Any]]) -> int:
    normalized = ddrg.normalize_answer(answer)
    delta = 0
    for result in probe_results:
        if normalized and normalized == ddrg.normalize_answer(result.get("supported", "")):
            delta += 1
        if normalized and normalized == ddrg.normalize_answer(result.get("opposed", "")):
            delta -= 1
    return delta


def graph_anchor_score(graph: Dict[str, Any], probe_results: List[Dict[str, Any]]) -> Tuple[float, Dict[str, Any]]:
    return learned_heuristic_anchor_score(graph, probe_results)


def select_verified_anchor(
    repaired_support_graphs: List[Dict[str, Any]],
    fallback_pred: str,
    probe_results: List[Dict[str, Any]],
    alignment: Optional[Dict[str, Any]] = None,
    anchor_scorer: Optional[LogisticAnchorScorer] = None,
) -> Tuple[str, Optional[Dict[str, Any]], Dict[str, Any]]:
    supported_answers = {
        ddrg.normalize_answer(result.get("supported", ""))
        for result in probe_results
        if ddrg.normalize_answer(result.get("supported", ""))
    }
    valid_graphs = []
    anchor_diagnostics = []
    for graph in repaired_support_graphs:
        if anchor_scorer is not None:
            score, components = anchor_scorer.score_graph(
                graph=graph,
                all_graphs=repaired_support_graphs,
                probe_results=probe_results,
                alignment=alignment,
            )
        else:
            score, components = graph_anchor_score(graph, probe_results)
        graph["anchor_score"] = score
        graph["anchor_score_components"] = components
        answer = ddrg.normalize_answer(graph.get("predicted_answer", ""))
        answer_valid = bool(graph.get("repair_summary", {}).get("answer_node_valid", False))
        if answer and answer_valid:
            valid_graphs.append(graph)
        anchor_diagnostics.append(
            {
                "graph": graph.get("graph"),
                "predicted_answer": answer,
                "answer_node_valid": answer_valid,
                "probe_supported_answer": bool(answer and answer in supported_answers),
                "anchor_score": score,
                "anchor_score_components": components,
            }
        )

    anchor_diagnostics.sort(
        key=lambda item: (
            bool(item.get("answer_node_valid", False)),
            bool(item.get("probe_supported_answer", False)),
            float(item.get("anchor_score", 0.0)),
        ),
        reverse=True,
    )
    preferred = [
        graph
        for graph in valid_graphs
        if ddrg.normalize_answer(graph.get("predicted_answer", "")) in supported_answers
    ]
    pool = preferred or valid_graphs
    if not pool:
        return fallback_pred, None, {
            "answer_source": "graph_sc_fallback",
            "fallback_reason": "no_valid_repaired_anchor",
            "score_mode": "learned_anchor_logreg" if anchor_scorer is not None else "heuristic_verified_anchor",
            "supported_answers": sorted(supported_answers),
            "anchor_diagnostics": anchor_diagnostics,
        }

    selected_graph = max(pool, key=lambda graph: float(graph.get("anchor_score", 0.0)))
    selected_answer = ddrg.normalize_answer(selected_graph.get("predicted_answer", "")) or fallback_pred
    return selected_answer, selected_graph, {
        "answer_source": "learned_anchor" if anchor_scorer is not None else "verified_anchor",
        "score_mode": "learned_anchor_logreg" if anchor_scorer is not None else "heuristic_verified_anchor",
        "selected_graph": selected_graph.get("graph"),
        "selected_graph_answer": selected_answer,
        "selected_graph_score": selected_graph.get("anchor_score"),
        "selected_graph_score_components": selected_graph.get("anchor_score_components", {}),
        "selected_graph_repair_summary": selected_graph.get("repair_summary", {}),
        "used_probe_supported_answer_filter": bool(preferred),
        "supported_answers": sorted(supported_answers),
        "anchor_diagnostics": anchor_diagnostics,
        "anchor_scorer_model": anchor_scorer.metadata if anchor_scorer is not None else None,
    }


def build_integrated_repaired_graph(
    anchor_graph: Optional[Dict[str, Any]],
    source: str = "programmatic_verified_anchor",
) -> Optional[Dict[str, Any]]:
    if not anchor_graph:
        return None
    return {
        "source": source,
        "final_answer": anchor_graph.get("predicted_answer", ""),
        "anchor_graph": anchor_graph.get("graph"),
        "anchor_score": anchor_graph.get("anchor_score"),
        "repair_summary": anchor_graph.get("repair_summary", {}),
        "nodes": [
            {
                "id": node.get("id", ""),
                "parents": node.get("parents", []),
                "edge_label": node.get("edge_label", ""),
                "claim": node.get("content", ""),
                "repair_status": node.get("repair", {}).get("status", "untouched"),
            }
            for node in anchor_graph.get("repaired_nodes", [])
        ],
        "edges": [
            {
                "source": edge.get("source", ""),
                "target": edge.get("target", ""),
                "label": edge.get("label", ""),
                "repair_status": edge.get("repair", {}).get("status", "untouched"),
            }
            for edge in anchor_graph.get("repaired_edges", [])
        ],
    }


def integrated_graph_source(answer_source: str) -> str:
    if answer_source == "learned_anchor":
        return "programmatic_learned_anchor"
    return "programmatic_verified_anchor"


def format_integrated_repaired_graph(graph: Optional[Dict[str, Any]]) -> str:
    if not graph:
        return ""
    edges_by_target: Dict[str, List[Dict[str, Any]]] = {}
    for edge in graph.get("edges", []):
        target = str(edge.get("target", ""))
        if target:
            edges_by_target.setdefault(target, []).append(edge)

    lines = [
        f"# source: {graph.get('source', '')}",
        f"# anchor_graph: {graph.get('anchor_graph', '')}",
    ]
    for node in graph.get("nodes", []):
        node_id = str(node.get("id", ""))
        claim = str(node.get("claim", "")).strip()
        if not node_id:
            continue
        incoming = edges_by_target.get(node_id, [])
        if node_id in {"N0", "Q"} or not incoming:
            lines.append(f"{node_id}: {claim}")
            continue
        parents = [str(edge.get("source", "")) for edge in incoming if edge.get("source")]
        labels = [str(edge.get("label", "")).strip() for edge in incoming if edge.get("label")]
        label = labels[0] if len(set(labels)) == 1 else "; ".join(labels)
        if label:
            lines.append(f"{node_id} <- {', '.join(parents)} [{label}]: {claim}")
        else:
            lines.append(f"{node_id} <- {', '.join(parents)}: {claim}")
    lines.append(f"Final Answer: {ddrg.normalize_answer(graph.get('final_answer', ''))}")
    return "\n".join(lines)


def run_ddrg_v1(
    client: LLMClient,
    args: argparse.Namespace,
    question: str,
    candidates: Optional[List[ddrg.Candidate]] = None,
    outputs: Optional[List[str]] = None,
) -> Tuple[str, List[str], List[ddrg.Candidate], Dict[str, Any]]:
    trace: List[Dict[str, Any]] = []
    alignment_mode = str(getattr(args, "alignment_mode", "llm") or "llm").strip().lower()
    if alignment_mode not in {"llm", "hybrid"}:
        alignment_mode = "llm"
    if candidates is None:
        outputs, candidates = sample_graphs(client, args, question)
    else:
        outputs = list(outputs or [])
        candidates = normalize_existing_candidates(list(candidates), question)

    proposal_answers = [candidate.parsed.final_answer for candidate in candidates]
    fallback_pred = majority_answer(candidates, question)
    trace.append(
        {
            "stage": "graph_proposal",
            "num_graphs": len(candidates),
            "answers": proposal_answers,
            "fallback_graph_sc": fallback_pred,
        }
    )

    support_graphs = support_graphs_from_candidates(candidates, args.max_support_nodes, question)
    trace.append(
        {
            "stage": "answer_support_subgraph_extraction",
            "num_support_graphs": len(support_graphs),
        }
    )

    alignment: Dict[str, Any] = {"parse_ok": False, "alignments": []}
    alignment_diagnostics: Dict[str, Any] = {"mode": alignment_mode, "returned_cluster_count": 0}
    meta_graph: Dict[str, Any] = {}
    conflict_builder: Dict[str, Any] = {"parse_ok": False, "conflicts": []}
    probe_results: List[Dict[str, Any]] = []

    if len(support_graphs) > 1:
        if alignment_mode == "hybrid":
            alignment = hybrid_align_support_graphs(
                client=client,
                args=args,
                question=question,
                support_graphs=support_graphs,
                llm_align_fn=lambda: align_support_graphs(client, args, question, support_graphs),
            )
        else:
            alignment = align_support_graphs(client, args, question, support_graphs)
        alignment_diagnostics = dict(alignment.get("diagnostics", {}))
        alignment_diagnostics.setdefault("mode", alignment_mode)
        alignment_diagnostics.setdefault("returned_cluster_count", len(alignment.get("alignments", [])))
        meta_graph = build_meta_graph(support_graphs, alignment)
        trace.append(
            {
                "stage": "graph_first_alignment",
                "mode": alignment_mode,
                "parse_ok": alignment.get("parse_ok", False),
                "num_alignments": len(alignment.get("alignments", [])),
                "diagnostics": alignment_diagnostics,
            }
        )
        if alignment.get("parse_ok") and alignment.get("alignments"):
            conflict_builder = localize_frontier_conflicts(
                client=client,
                args=args,
                question=question,
                support_graphs=support_graphs,
                meta_graph=meta_graph,
            )
            conflicts = conflict_builder.get("conflicts", [])[: args.max_probes]
            trace.append(
                {
                    "stage": "causal_frontier_localization",
                    "parse_ok": conflict_builder.get("parse_ok", False),
                    "num_conflicts": len(conflicts),
                }
            )
            if conflict_builder.get("parse_ok") and conflicts:
                probe_results = verify_frontier_conflicts(client, args, question, conflicts)
                trace.append(
                    {
                        "stage": "atomic_probe_verification",
                        "num_probes": len(probe_results),
                        "verdicts": [result.get("verdict", "") for result in probe_results],
                    }
                )
            else:
                trace.append(
                    {
                        "stage": "atomic_probe_verification",
                        "num_probes": 0,
                        "reason": "no_answer_sensitive_frontier",
                    }
                )
        else:
            trace.append(
                {
                    "stage": "causal_frontier_localization",
                    "num_conflicts": 0,
                    "reason": "no_valid_alignment",
                }
            )

    repaired_support_graphs = repair_support_graphs(support_graphs, probe_results)
    trace.append({"stage": "claim_edge_repair", "num_probe_results": len(probe_results)})

    selected, selected_graph, anchor_selection = select_verified_anchor(
        repaired_support_graphs=repaired_support_graphs,
        fallback_pred=fallback_pred,
        probe_results=probe_results,
        alignment=alignment,
        anchor_scorer=getattr(args, "anchor_scorer", None),
    )
    integrated_graph = build_integrated_repaired_graph(
        selected_graph,
        source=integrated_graph_source(str(anchor_selection.get("answer_source", "verified_anchor"))),
    )
    integrated_graph_text = format_integrated_repaired_graph(integrated_graph)
    trace.append(
        {
            "stage": "verified_anchor_selection",
            "selected": selected,
            "score_mode": anchor_selection.get("score_mode", "heuristic_verified_anchor"),
            "num_scored_graphs": len(anchor_selection.get("anchor_diagnostics", [])),
            "selection": anchor_selection,
        }
    )

    info = {
        "selected": selected,
        "selection_mode": anchor_selection.get("answer_source", "verified_anchor"),
        "proposal_answers": proposal_answers,
        "fallback_graph_sc": fallback_pred,
        "alignment_mode": alignment_mode,
        "alignment_diagnostics": alignment_diagnostics,
        "support_graphs": support_graphs,
        "alignment": alignment,
        "meta_graph": meta_graph,
        "conflict_builder": conflict_builder,
        "probe_results": probe_results,
        "repaired_support_graphs": repaired_support_graphs,
        "selected_graph_output": selected_graph,
        "anchor_selection": anchor_selection,
        "anchor_diagnostics": anchor_selection.get("anchor_diagnostics", []),
        "integrated_repaired_graph": integrated_graph,
        "integrated_repaired_graph_text": integrated_graph_text,
        "trace": trace,
    }
    return selected, outputs or [], candidates, info
