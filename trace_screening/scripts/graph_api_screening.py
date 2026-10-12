#!/usr/bin/env python3
"""Run the bounded E-H search-burden screening pilot through OpenRouter.

The four directed weighted graphs cross two properties:
  E: medium burden, four optimal paths, equal edge counts.
  F: medium burden, four optimal paths, mixed edge counts.
  G: high burden, eight optimal paths, equal edge counts.
  H: high burden, eight optimal paths, mixed edge counts.

All conditions use the repaired, internally consistent List-All prompt. Each
graph also includes multiple near-optimal paths with total weight min + 1.
Provider routing is locked, fallbacks are disabled, and temperature is omitted.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import graph_api_followup as base


GraphCondition = base.GraphCondition
MODELS = base.MODELS


GRAPHS: Dict[str, GraphCondition] = {
    "E": GraphCondition(
        key="E",
        label="medium burden; four optimal paths; equal edge counts",
        edges=(
            ("S", "A", 1), ("S", "B", 1),
            ("A", "C", 2), ("A", "D", 2),
            ("B", "C", 2), ("B", "D", 2),
            ("C", "T", 3), ("D", "T", 3),
            ("A", "E", 3), ("A", "F", 3), ("A", "G", 3),
            ("B", "E", 3), ("B", "F", 3), ("B", "G", 3),
            ("E", "T", 3), ("F", "T", 3), ("G", "T", 3),
        ),
        expected_paths=(
            ("S", "A", "C", "T"), ("S", "A", "D", "T"),
            ("S", "B", "C", "T"), ("S", "B", "D", "T"),
        ),
        expected_weight=6,
        edge_count_relation="same (3 edges each)",
        optimal_path_count=4,
    ),
    "F": GraphCondition(
        key="F",
        label="medium burden; four optimal paths; mixed edge counts",
        edges=(
            ("S", "A", 1), ("A", "T", 5),
            ("S", "B", 1), ("B", "T", 5),
            ("S", "C", 1), ("C", "E", 2), ("E", "T", 3),
            ("S", "D", 1), ("D", "F", 2), ("F", "T", 3),
            ("C", "F", 3), ("D", "E", 3),
            ("C", "T", 6), ("D", "T", 6),
            ("S", "G", 2), ("G", "T", 5),
        ),
        expected_paths=(
            ("S", "A", "T"), ("S", "B", "T"),
            ("S", "C", "E", "T"), ("S", "D", "F", "T"),
        ),
        expected_weight=6,
        edge_count_relation="different (2 or 3 edges)",
        optimal_path_count=4,
    ),
    "G": GraphCondition(
        key="G",
        label="high burden; eight optimal paths; equal edge counts",
        edges=(
            ("S", "A", 1), ("S", "B", 1),
            ("A", "C", 2), ("A", "D", 2), ("A", "E", 2), ("A", "F", 2),
            ("B", "C", 2), ("B", "D", 2), ("B", "E", 2), ("B", "F", 2),
            ("C", "T", 3), ("D", "T", 3), ("E", "T", 3), ("F", "T", 3),
            ("A", "G", 3), ("A", "H", 3), ("A", "I", 3), ("A", "J", 3),
            ("B", "G", 3), ("B", "H", 3), ("B", "I", 3), ("B", "J", 3),
            ("G", "T", 3), ("H", "T", 3), ("I", "T", 3), ("J", "T", 3),
        ),
        expected_paths=tuple(
            ("S", first, second, "T")
            for first in ("A", "B")
            for second in ("C", "D", "E", "F")
        ),
        expected_weight=6,
        edge_count_relation="same (3 edges each)",
        optimal_path_count=8,
    ),
    "H": GraphCondition(
        key="H",
        label="high burden; eight optimal paths; mixed edge counts",
        edges=(
            ("S", "A", 1), ("A", "T", 5),
            ("S", "B", 1), ("B", "T", 5),
            ("S", "C", 1), ("C", "T", 5),
            ("S", "D", 1), ("D", "T", 5),
            ("S", "E", 1), ("S", "F", 1),
            ("E", "G", 2), ("E", "H", 2),
            ("F", "G", 2), ("F", "H", 2),
            ("G", "T", 3), ("H", "T", 3),
            ("E", "I", 3), ("E", "J", 3),
            ("F", "I", 3), ("F", "J", 3),
            ("I", "T", 3), ("J", "T", 3),
            ("S", "K", 2), ("K", "T", 5),
            ("S", "L", 2), ("L", "T", 5),
            ("S", "M", 2), ("M", "T", 5),
            ("S", "N", 2), ("N", "T", 5),
        ),
        expected_paths=(
            ("S", "A", "T"), ("S", "B", "T"),
            ("S", "C", "T"), ("S", "D", "T"),
            ("S", "E", "G", "T"), ("S", "E", "H", "T"),
            ("S", "F", "G", "T"), ("S", "F", "H", "T"),
        ),
        expected_weight=6,
        edge_count_relation="different (2 or 3 edges)",
        optimal_path_count=8,
    ),
}


SCHEDULE_PATTERN = ("E", "F", "G", "H", "H", "G", "F", "E", "F", "H", "E", "G")


def enumerate_simple_paths(graph: GraphCondition) -> List[Tuple[Tuple[str, ...], int]]:
    adjacency: Dict[str, List[Tuple[str, int]]] = {}
    for source, target, weight in graph.edges:
        adjacency.setdefault(source, []).append((target, weight))
    found: List[Tuple[Tuple[str, ...], int]] = []

    def walk(node: str, path: Tuple[str, ...], total: int) -> None:
        if node == "T":
            found.append((path, total))
            return
        for target, weight in adjacency.get(node, []):
            if target not in path:
                walk(target, path + (target,), total + weight)

    walk("S", ("S",), 0)
    return found


def validate_graphs() -> Dict[str, Dict[str, Any]]:
    truth: Dict[str, Dict[str, Any]] = {}
    for key, graph in GRAPHS.items():
        paths = enumerate_simple_paths(graph)
        minimum = min(weight for _, weight in paths)
        optimal = tuple(sorted(path for path, weight in paths if weight == minimum))
        if minimum != graph.expected_weight or optimal != tuple(sorted(graph.expected_paths)):
            raise ValueError(f"Graph {key} ground truth mismatch")
        near_optimal = [(path, weight) for path, weight in paths if weight == minimum + 1]
        if not near_optimal:
            raise ValueError(f"Graph {key} has no min+1 distractor")
        truth[key] = {
            "label": graph.label,
            "minimum_weight": minimum,
            "minimum_paths": [list(path) for path in optimal],
            "edge_counts": sorted(len(path) - 1 for path in optimal),
            "total_simple_path_count": len(paths),
            "near_optimal_path_count": len(near_optimal),
            "all_simple_paths": [
                {"path": list(path), "weight": weight, "edges": len(path) - 1}
                for path, weight in sorted(paths)
            ],
        }
    return truth


def make_schedule(conditions: Sequence[str], repetitions: int) -> List[str]:
    counts = Counter()
    schedule: List[str] = []
    while any(counts[key] < repetitions for key in conditions):
        for key in SCHEDULE_PATTERN:
            if key in conditions and counts[key] < repetitions:
                schedule.append(key)
                counts[key] += 1
    return schedule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=tuple(MODELS), default=list(MODELS))
    parser.add_argument("--conditions", nargs="+", choices=tuple(GRAPHS), default=list(GRAPHS))
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--output-root", type=Path, default=Path.cwd())
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.repetitions < 1:
        raise SystemExit("--repetitions must be at least 1")
    ground_truth = validate_graphs()
    schedules = {key: make_schedule(args.conditions, args.repetitions) for key in args.models}
    total_requests = sum(len(value) for value in schedules.values())

    print(f"Models: {', '.join(args.models)}")
    print(f"Conditions: {', '.join(args.conditions)}")
    for model_key, schedule in schedules.items():
        print(f"Graph order for {model_key}: {' '.join(schedule)}")
    print(f"Total requests: {total_requests}")
    for model_key in args.models:
        spec = MODELS[model_key]
        print(f"- {model_key}: {spec['model']} via {spec['provider']} (fallback disabled)")
    for key in args.conditions:
        item = ground_truth[key]
        print(
            f"- Graph {key}: {len(item['minimum_paths'])} optimal / "
            f"{item['total_simple_path_count']} total paths; min+1 distractors "
            f"{item['near_optimal_path_count']}; edge counts {item['edge_counts']}"
        )
    if args.dry_run:
        return 0

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        api_key = getpass.getpass("Paste OpenRouter key: ").strip()
    if not api_key:
        print("No API key supplied.", file=sys.stderr)
        return 2

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (args.output_root / f"graph_api_screening_{stamp}").resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    raw_path = output_dir / "raw_runs.jsonl"
    coding_path = output_dir / "coding_sheet.csv"
    manifest_path = output_dir / "manifest.json"
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "endpoint": base.ENDPOINT,
        "study_extension": {
            "purpose": "bounded screening for search-space and completeness pressure",
            "primary_factor": "candidate-path burden: medium versus high",
            "secondary_structure": "optimal paths have equal versus mixed edge counts",
            "prompt": "repaired internally consistent List-All prompt",
            "note": "Screening pilot only; not a replicated formal factorial experiment.",
        },
        "models": {key: MODELS[key] for key in args.models},
        "repetitions_per_graph_model": args.repetitions,
        "schedule_per_model": schedules,
        "temperature": "omitted; provider default",
        "provider_fallbacks": "disabled",
        "prompts": {key: base.make_prompt(GRAPHS[key]) for key in args.conditions},
        "ground_truth": {key: ground_truth[key] for key in args.conditions},
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    repetition_counts: Dict[Tuple[str, str], int] = Counter()
    coding_rows: List[Dict[str, Any]] = []
    completed = 0
    with raw_path.open("w", encoding="utf-8") as raw_handle:
        for model_key in args.models:
            spec = MODELS[model_key]
            for condition in schedules[model_key]:
                repetition_counts[(model_key, condition)] += 1
                repetition = repetition_counts[(model_key, condition)]
                run_id = f"{model_key}_{condition}{repetition:02d}"
                payload = {
                    "model": spec["model"],
                    "messages": [{"role": "user", "content": base.make_prompt(GRAPHS[condition])}],
                    "provider": {"only": [spec["provider"]], "allow_fallbacks": False},
                }
                completed += 1
                print(f"[{completed}/{total_requests}] {run_id} ... ", end="", flush=True)
                status, response = base.request_json(api_key, payload, args.timeout)
                print(f"HTTP {status}")
                timestamp_utc = datetime.now(timezone.utc).isoformat()
                raw_record = {
                    "run_id": run_id, "timestamp_utc": timestamp_utc,
                    "model_key": model_key, "condition": condition,
                    "repetition": repetition, "request": payload,
                    "http_status": status, "response": response,
                }
                raw_handle.write(json.dumps(raw_record, ensure_ascii=False) + "\n")
                raw_handle.flush()
                usage = response.get("usage", {}) if isinstance(response, dict) else {}
                coding_rows.append({
                    "run_id": run_id, "timestamp_utc": timestamp_utc,
                    "model_key": model_key, "requested_model": spec["model"],
                    "requested_provider": spec["provider"], "graph": condition,
                    "graph_label": GRAPHS[condition].label, "repetition": repetition,
                    "http_status": status,
                    "returned_model": response.get("model", "") if isinstance(response, dict) else "",
                    "returned_provider": response.get("provider", "") if isinstance(response, dict) else "",
                    "response_id": response.get("id", "") if isinstance(response, dict) else "",
                    "prompt_tokens": usage.get("prompt_tokens", ""),
                    "completion_tokens": usage.get("completion_tokens", ""),
                    "total_tokens": usage.get("total_tokens", ""),
                    "cost_usd": usage.get("cost", ""),
                    "visible_output": base.visible_output(response),
                    "exact_path_set": "", "correct_weights": "", "correct_edge_counts": "",
                    "internally_consistent": "", "criterion_confusion": "",
                    "trace_primary": "", "trace_components": "", "trace_sequence": "",
                    "completeness_support": "", "parseable": "", "notes": "",
                })
                base.write_coding_sheet(coding_path, coding_rows)
                if completed < total_requests and args.delay > 0:
                    time.sleep(args.delay)

    print(f"\nFinished. Results: {output_dir}")
    print(f"Raw archive: {raw_path.name}")
    print(f"Coding sheet: {coding_path.name}")
    print(f"Manifest: {manifest_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
