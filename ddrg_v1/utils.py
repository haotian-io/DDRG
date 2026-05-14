"""Standalone utilities for the DDRG v1 package.

This module intentionally contains the small subset of helpers needed by the
pipeline so ddrg_v1 does not depend on the older top-level experiment files.
"""

import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


NODE_ID_PATTERN = r"(?:ANS|Q|[A-Z]+_[A-Za-z0-9]+|[A-Z]\d+|N\d+|[A-Z])"

BENCHMARK_FILES = {
    "aiw": ("benchmarks/aiw/AIW_easy.pkl", "questions", "answers"),
    "aiw+": ("benchmarks/aiw+/AIW_hard.pkl", "questions", "answers"),
    "logiqa": ("benchmarks/logiqa/logiqa_test.csv", "Question", "Label"),
    "lsat_ar": ("benchmarks/lsat_ar/lsat-ar.csv", "Question", "Label"),
    "mathqa": ("benchmarks/mathqa/mathqa.csv", "Question", "Label"),
    "medqa": ("benchmarks/medqa/medqa.csv", "Question", "Label"),
}


@dataclass
class GraphNode:
    node_id: str
    parents: List[str]
    edge_label: str
    content: str


@dataclass
class ParsedGraph:
    nodes: Dict[str, GraphNode]
    final_answer: str
    raw_final_answer: str
    parse_ok: bool
    issues: List[str]
    output: str


@dataclass
class Candidate:
    parsed: ParsedGraph
    score: float


def project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "benchmarks").exists():
            return parent
    return here.parents[2]


def load_dataset(name: str, limit: Optional[int], seed: int) -> List[Dict[str, Any]]:
    rel_path, question_col, answer_col = BENCHMARK_FILES[name]
    path = project_root() / rel_path
    if not path.exists():
        raise FileNotFoundError(
            f"Benchmark file not found: {path}. "
            "Place the dataset under the expected benchmarks/ directory, "
            "or run a single ad-hoc question with --question."
        )
    if path.suffix == ".pkl":
        df = pd.read_pickle(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported benchmark file: {path}")

    rows = [
        {
            "id": int(i),
            "question": str(row[question_col]),
            "gold": normalize_answer(row[answer_col]),
            "raw_gold": row[answer_col],
        }
        for i, row in df.iterrows()
    ]
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:limit] if limit is not None else rows


def normalize_answer(value: Any) -> str:
    text = str(value).strip()
    if not text:
        return ""
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.strip().strip(".?:; ")

    final_match = re.search(r"(?im)^\s*Final\s*Answer\s*:\s*(.+?)\s*$", text)
    if final_match:
        text = final_match.group(1).strip()

    letter_match = re.search(r"(?i)(?:^|[^A-Z])([A-E])(?:[^A-Z]|$)", text)
    if letter_match:
        return letter_match.group(1).upper()

    number_match = re.search(r"[-+]?\d+(?:\.\d+)?", text.replace(",", ""))
    if number_match:
        num_text = number_match.group(0)
        try:
            normalized = format(Decimal(num_text).normalize(), "f")
            return normalized.rstrip("0").rstrip(".") if "." in normalized else normalized
        except InvalidOperation:
            return num_text

    return text.lower()


def answers_match(pred: str, gold: str) -> bool:
    pred_norm = normalize_answer(pred)
    gold_norm = normalize_answer(gold)
    if pred_norm == gold_norm:
        return True
    try:
        return Decimal(pred_norm) == Decimal(gold_norm)
    except InvalidOperation:
        return False


def parse_elrg(output: str) -> ParsedGraph:
    nodes: Dict[str, GraphNode] = {}
    issues: List[str] = []
    final_answer_raw = ""
    edge_re = re.compile(
        rf"^\s*({NODE_ID_PATTERN})\s*<-\s*([A-Za-z0-9_,\s]+)"
        r"(?:\s*\[([^\]]*)\])?\s*:\s*(.*?)\s*$"
    )
    root_re = re.compile(r"^\s*(N0|Q)\s*(?:\[[^\]]+\])?\s*:\s*(.*?)\s*$")
    final_re = re.compile(r"^\s*Final\s*Answer\s*:\s*(.*?)\s*$", re.I)

    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        final_match = final_re.match(line)
        if final_match:
            final_answer_raw = final_match.group(1).strip()
            continue
        edge_match = edge_re.match(line)
        if edge_match:
            node_id = edge_match.group(1)
            parents = [p.strip() for p in edge_match.group(2).split(",") if p.strip()]
            nodes[node_id] = GraphNode(
                node_id=node_id,
                parents=parents,
                edge_label=(edge_match.group(3) or "").strip(),
                content=edge_match.group(4).strip(),
            )
            continue
        root_match = root_re.match(line)
        if root_match:
            node_id = root_match.group(1)
            nodes[node_id] = GraphNode(node_id, [], "", root_match.group(2).strip())

    roots = [node_id for node_id in ("N0", "Q") if node_id in nodes]
    if not roots:
        issues.append("missing_root")
    if "ANS" not in nodes:
        issues.append("missing_ans_node")
    if not final_answer_raw and "ANS" in nodes:
        final_answer_raw = nodes["ANS"].content
    if not final_answer_raw:
        issues.append("missing_final_answer")

    defined_before: set[str] = set()
    ordered_ids = [node_id for node_id in nodes if node_id != "ANS"]
    if "ANS" in nodes:
        ordered_ids.append("ANS")
    for node_id in ordered_ids:
        node = nodes[node_id]
        if node_id not in roots and not node.parents:
            issues.append(f"{node_id}:missing_parent")
        for parent in node.parents:
            if parent not in nodes:
                issues.append(f"{node_id}:unknown_parent:{parent}")
            elif parent not in defined_before:
                issues.append(f"{node_id}:non_topological_parent:{parent}")
        defined_before.add(node_id)

    return ParsedGraph(
        nodes=nodes,
        final_answer=normalize_answer(final_answer_raw),
        raw_final_answer=final_answer_raw,
        parse_ok=not issues,
        issues=issues,
        output=output,
    )


def score_graph(parsed: ParsedGraph, max_nodes: int) -> float:
    score = 1.0
    if parsed.parse_ok:
        score += 0.75
    else:
        score -= 0.35 * len(parsed.issues)
    ans = parsed.nodes.get("ANS")
    if ans and ans.parents:
        score += 0.75
    node_count = len([node_id for node_id in parsed.nodes if node_id != "ANS"])
    if node_count == 0:
        score -= 1.0
    elif node_count > max_nodes:
        score -= 0.1 * (node_count - max_nodes)
    return score


def ancestor_node_ids(parsed: ParsedGraph, start_ids: List[str]) -> List[str]:
    seen: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in seen or node_id not in parsed.nodes:
            return
        seen.add(node_id)
        for parent in parsed.nodes[node_id].parents:
            visit(parent)

    for start_id in start_ids:
        visit(start_id)
    return [node_id for node_id in parsed.nodes if node_id in seen and node_id != "ANS"]


def compact_support_ids(support_ids: List[str], max_nodes: int) -> List[str]:
    if max_nodes <= 0 or len(support_ids) <= max_nodes:
        return support_ids
    tail_count = max(0, max_nodes - 1)
    keep = set(support_ids[:1] + (support_ids[-tail_count:] if tail_count else []))
    return [node_id for node_id in support_ids if node_id in keep]


def extract_support_graph(index: int, candidate: Candidate, max_nodes: int) -> Dict[str, Any]:
    parsed = candidate.parsed
    ans = parsed.nodes.get("ANS")
    parent_ids = ans.parents if ans else []
    support_ids = compact_support_ids(ancestor_node_ids(parsed, parent_ids), max_nodes)
    included_ids = support_ids + (["ANS"] if ans else [])
    included = set(included_ids)
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, str]] = []
    omitted_parent_edges: List[Dict[str, str]] = []

    for node_id in included_ids:
        node = parsed.nodes.get(node_id)
        if not node:
            continue
        nodes.append(
            {
                "id": node_id,
                "parents": node.parents,
                "edge_label": node.edge_label,
                "content": node.content,
                "is_answer_node": node_id == "ANS",
            }
        )
        for parent in node.parents:
            edge = {"source": parent, "target": node_id, "label": node.edge_label}
            if parent in included:
                edges.append(edge)
            else:
                omitted_parent_edges.append(edge)

    local_claims = [
        {"node": n["id"], "edge_label": n["edge_label"], "claim": n["content"]}
        for n in nodes
        if n["id"] != "N0"
    ]
    return {
        "graph": index,
        "predicted_answer": parsed.final_answer,
        "raw_final_answer": parsed.raw_final_answer,
        "parse_ok": parsed.parse_ok,
        "issues": parsed.issues,
        "graph_score": candidate.score,
        "answer_parents": parent_ids,
        "support_node_ids": support_ids,
        "nodes": nodes,
        "edges": edges,
        "omitted_parent_edges": omitted_parent_edges,
        "local_claims": local_claims,
    }


def truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 20].rstrip() + "\n...[truncated]"


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(cleaned[start : end + 1])
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def parse_graph_index(value: Any) -> Optional[int]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    match = re.search(r"\d+", str(value))
    return int(match.group(0)) if match else None


def normalize_node_id(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    match = re.search(rf"\b({NODE_ID_PATTERN})\b", text)
    return match.group(1) if match else text.strip()


def normalize_repair_target(item: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(item, dict):
        return None
    graph_id = parse_graph_index(item.get("graph") or item.get("graph_id"))
    if graph_id is None:
        return None

    edge_item = item.get("edge") if isinstance(item.get("edge"), dict) else {}
    edge_source = normalize_node_id(edge_item.get("source") or item.get("source"))
    edge_target = normalize_node_id(edge_item.get("target") or item.get("target"))
    node_id = normalize_node_id(item.get("node") or item.get("node_id") or edge_target)

    target: Dict[str, Any] = {"graph": graph_id}
    if node_id:
        target["node"] = node_id
    if edge_source and edge_target:
        target["edge"] = {"source": edge_source, "target": edge_target}
        target.setdefault("node", edge_target)
    if "node" not in target and "edge" not in target:
        return None
    return target


def target_key(target: Dict[str, Any]) -> Tuple[int, str, str, str]:
    edge = target.get("edge") or {}
    return (
        int(target.get("graph", -1)),
        str(target.get("node", "")),
        str(edge.get("source", "")),
        str(edge.get("target", "")),
    )


def unique_targets(targets: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set[Tuple[int, str, str, str]] = set()
    unique: List[Dict[str, Any]] = []
    for target in targets:
        key = target_key(target)
        if key in seen:
            continue
        seen.add(key)
        unique.append(target)
    return unique


def normalize_side_name(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text if text in {"left", "right"} else ""


def parse_probe_verdict(output: str) -> Dict[str, Any]:
    data = extract_json_object(output)
    if data is not None:
        verdict = str(data.get("verdict", "")).strip().upper()
        if verdict not in {"YES", "NO", "UNKNOWN"}:
            verdict = "UNKNOWN"
        return {
            "parse_ok": True,
            "verdict": verdict,
            "reason": str(data.get("reason", "")).strip(),
            "value": str(data.get("value", "")).strip(),
            "raw": output,
        }
    text = output.strip().upper()
    if re.search(r"\bYES\b", text):
        verdict = "YES"
    elif re.search(r"\bNO\b", text):
        verdict = "NO"
    else:
        verdict = "UNKNOWN"
    return {"parse_ok": False, "verdict": verdict, "reason": "", "value": "", "raw": output}


def majority_vote(answers: Iterable[str]) -> str:
    counts = Counter(answer for answer in answers if answer)
    return counts.most_common(1)[0][0] if counts else ""


def majority_verdict(verdicts: List[str]) -> str:
    counts = Counter(verdict for verdict in verdicts if verdict)
    return counts.most_common(1)[0][0] if counts else "UNKNOWN"


def repair_status(state: Dict[str, Any]) -> str:
    verified = int(state.get("verified", 0))
    refuted = int(state.get("refuted", 0))
    if verified and refuted and verified == refuted:
        return "contested"
    if refuted > verified:
        return "refuted"
    if verified > refuted:
        return "verified"
    return "untouched"
