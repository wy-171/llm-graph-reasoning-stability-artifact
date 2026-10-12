#!/usr/bin/env python3
"""Follow-up API pilot for high-multiplicity minimum-weight paths.

Adds two graph conditions to the earlier A/B pilot:
  C: four optimal paths, all with the same number of edges.
  D: four optimal paths, split across two different edge counts.

The script uses OpenRouter with provider routing locked and fallbacks disabled.
Temperature is intentionally omitted so each provider uses its default.
"""

from __future__ import annotations

import argparse
import csv
import getpass
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

MODELS: Dict[str, Dict[str, str]] = {
    "claude": {
        "model": "anthropic/claude-sonnet-4.6",
        "provider": "anthropic",
    },
    "mistral": {
        "model": "mistralai/mistral-large-2512",
        "provider": "mistral",
    },
    "deepseek": {
        "model": "deepseek/deepseek-v3.2-exp",
        "provider": "siliconflow",
    },
}


@dataclass(frozen=True)
class GraphCondition:
    key: str
    label: str
    edges: Tuple[Tuple[str, str, int], ...]
    expected_paths: Tuple[Tuple[str, ...], ...]
    expected_weight: int
    edge_count_relation: str
    optimal_path_count: int


GRAPHS: Dict[str, GraphCondition] = {
    "C": GraphCondition(
        key="C",
        label="four optimal paths; equal edge counts",
        edges=(
            ("S", "A", 1),
            ("S", "B", 1),
            ("A", "C", 2),
            ("A", "D", 2),
            ("B", "C", 2),
            ("B", "D", 2),
            ("C", "T", 3),
            ("D", "T", 3),
            ("S", "T", 7),
        ),
        expected_paths=(
            ("S", "A", "C", "T"),
            ("S", "A", "D", "T"),
            ("S", "B", "C", "T"),
            ("S", "B", "D", "T"),
        ),
        expected_weight=6,
        edge_count_relation="same (3 edges each)",
        optimal_path_count=4,
    ),
    "D": GraphCondition(
        key="D",
        label="four optimal paths; different edge counts",
        edges=(
            ("S", "A", 1),
            ("A", "T", 5),
            ("S", "B", 2),
            ("B", "T", 4),
            ("B", "D", 1),
            ("S", "C", 1),
            ("C", "D", 2),
            ("D", "T", 3),
            ("S", "T", 7),
        ),
        expected_paths=(
            ("S", "A", "T"),
            ("S", "B", "T"),
            ("S", "B", "D", "T"),
            ("S", "C", "D", "T"),
        ),
        expected_weight=6,
        edge_count_relation="different (2 or 3 edges)",
        optimal_path_count=4,
    ),
}


BASE_SCHEDULE = ("C", "D", "D", "C", "D", "C", "C", "D", "C", "D")


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
    ground_truth: Dict[str, Dict[str, Any]] = {}
    for key, graph in GRAPHS.items():
        paths = enumerate_simple_paths(graph)
        if not paths:
            raise ValueError(f"Graph {key} has no S-to-T path")
        minimum = min(weight for _, weight in paths)
        optimal = tuple(sorted(path for path, weight in paths if weight == minimum))
        expected = tuple(sorted(graph.expected_paths))
        if minimum != graph.expected_weight or optimal != expected:
            raise ValueError(
                f"Graph {key} ground-truth mismatch: calculated weight={minimum}, "
                f"paths={optimal}; expected weight={graph.expected_weight}, paths={expected}"
            )
        if len(optimal) != graph.optimal_path_count:
            raise ValueError(f"Graph {key} optimal-path-count mismatch")
        ground_truth[key] = {
            "label": graph.label,
            "minimum_weight": minimum,
            "minimum_paths": [list(path) for path in optimal],
            "edge_counts": sorted(len(path) - 1 for path in optimal),
            "all_simple_paths": [
                {"path": list(path), "weight": weight, "edges": len(path) - 1}
                for path, weight in sorted(paths)
            ],
        }
    return ground_truth


def make_prompt(graph: GraphCondition) -> str:
    edge_lines = "\n".join(f"{u} → {v}: {w}" for u, v, w in graph.edges)
    return (
        "Consider the following directed, weighted graph.\n\n"
        "Only the directed edges listed below exist:\n\n"
        f"{edge_lines}\n\n"
        "Find all distinct minimum-total-weight paths from S to T.\n\n"
        "For each reported path, provide:\n"
        "- the node sequence;\n"
        "- the total weight;\n"
        "- the number of edges.\n\n"
        "Briefly explain your method and include enough intermediate\n"
        "calculations to verify that the reported paths are all the\n"
        "minimum-weight paths.\n\n"
        "Equivalent wording and formatting are acceptable as long as\n"
        "the requested information is clear."
    )


def make_schedule(conditions: Sequence[str], repetitions: int) -> List[str]:
    if repetitions < 1:
        raise ValueError("repetitions must be at least 1")
    if set(conditions) == {"C", "D"} and len(conditions) == 2:
        counts = Counter()
        schedule: List[str] = []
        while any(counts[key] < repetitions for key in conditions):
            for key in BASE_SCHEDULE:
                if key in conditions and counts[key] < repetitions:
                    schedule.append(key)
                    counts[key] += 1
        return schedule
    return [key for repetition in range(repetitions) for key in conditions]


def request_json(api_key: str, payload: Dict[str, Any], timeout: float) -> Tuple[int, Dict[str, Any]]:
    request = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            return response.status, json.loads(body)
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = {"error": {"message": body}}
        return error.code, parsed
    except urllib.error.URLError as error:
        return 0, {"error": {"message": str(error.reason)}}


def visible_output(response: Dict[str, Any]) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
        return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    except (KeyError, IndexError, TypeError):
        return ""


def write_manifest(
    path: Path,
    models: Sequence[str],
    conditions: Sequence[str],
    repetitions: int,
    schedules: Dict[str, List[str]],
    ground_truth: Dict[str, Dict[str, Any]],
) -> None:
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "endpoint": ENDPOINT,
        "study_extension": {
            "factor_1": "optimal-path multiplicity: 2 (prior A/B) versus 4 (new C/D)",
            "factor_2": "optimal paths have same versus different edge counts",
            "new_conditions": list(conditions),
            "note": "A/B and C/D form a structured 2x2 instance set, not replicated factorial cells.",
        },
        "models": {key: MODELS[key] for key in models},
        "repetitions_per_graph_model": repetitions,
        "schedule_per_model": schedules,
        "temperature": "omitted; provider default",
        "provider_fallbacks": "disabled",
        "prompts": {key: make_prompt(GRAPHS[key]) for key in conditions},
        "ground_truth": {key: ground_truth[key] for key in conditions},
    }
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def write_coding_sheet(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    fieldnames = [
        "run_id",
        "timestamp_utc",
        "model_key",
        "requested_model",
        "requested_provider",
        "graph",
        "graph_label",
        "repetition",
        "http_status",
        "returned_model",
        "returned_provider",
        "response_id",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cost_usd",
        "visible_output",
        "exact_path_set",
        "correct_weights",
        "correct_edge_counts",
        "internally_consistent",
        "criterion_confusion",
        "trace_primary",
        "trace_components",
        "trace_sequence",
        "completeness_support",
        "parseable",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=tuple(MODELS), default=list(MODELS))
    parser.add_argument("--conditions", nargs="+", choices=tuple(GRAPHS), default=list(GRAPHS))
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds between requests")
    parser.add_argument("--timeout", type=float, default=180.0, help="Per-request timeout in seconds")
    parser.add_argument("--output-root", type=Path, default=Path.cwd())
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ground_truth = validate_graphs()
    schedules = {
        model_key: make_schedule(args.conditions, args.repetitions)
        for model_key in args.models
    }
    total_requests = sum(len(schedule) for schedule in schedules.values())

    print(f"Models: {', '.join(args.models)}")
    print(f"Conditions: {', '.join(args.conditions)}")
    for model_key in args.models:
        print(f"Graph order for {model_key}: {' '.join(schedules[model_key])}")
    print(f"Total requests: {total_requests}")
    for model_key in args.models:
        spec = MODELS[model_key]
        print(
            f"- {model_key}: {spec['model']} via {spec['provider']} "
            "(fallback disabled)"
        )
    for key in args.conditions:
        truth = ground_truth[key]
        paths = [" → ".join(path) for path in truth["minimum_paths"]]
        print(
            f"- Graph {key}: {len(paths)} optimal paths, weight "
            f"{truth['minimum_weight']}, edge counts {truth['edge_counts']}"
        )

    if args.dry_run:
        return 0

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        api_key = getpass.getpass("Paste OpenRouter key: ").strip()
    if not api_key:
        print("No API key supplied.", file=sys.stderr)
        return 2

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_root / f"graph_api_followup_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    raw_path = output_dir / "raw_runs.jsonl"
    coding_path = output_dir / "coding_sheet.csv"
    manifest_path = output_dir / "manifest.json"
    write_manifest(
        manifest_path,
        args.models,
        args.conditions,
        args.repetitions,
        schedules,
        ground_truth,
    )

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
                    "messages": [{"role": "user", "content": make_prompt(GRAPHS[condition])}],
                    "provider": {
                        "only": [spec["provider"]],
                        "allow_fallbacks": False,
                    },
                }

                completed += 1
                print(f"[{completed}/{total_requests}] {run_id} ... ", end="", flush=True)
                status, response = request_json(api_key, payload, args.timeout)
                print(f"HTTP {status}")

                timestamp_utc = datetime.now(timezone.utc).isoformat()
                raw_record = {
                    "run_id": run_id,
                    "timestamp_utc": timestamp_utc,
                    "model_key": model_key,
                    "condition": condition,
                    "repetition": repetition,
                    "request": payload,
                    "http_status": status,
                    "response": response,
                }
                raw_handle.write(json.dumps(raw_record, ensure_ascii=False) + "\n")
                raw_handle.flush()

                usage = response.get("usage", {}) if isinstance(response, dict) else {}
                coding_rows.append(
                    {
                        "run_id": run_id,
                        "timestamp_utc": timestamp_utc,
                        "model_key": model_key,
                        "requested_model": spec["model"],
                        "requested_provider": spec["provider"],
                        "graph": condition,
                        "graph_label": GRAPHS[condition].label,
                        "repetition": repetition,
                        "http_status": status,
                        "returned_model": response.get("model", "") if isinstance(response, dict) else "",
                        "returned_provider": response.get("provider", "") if isinstance(response, dict) else "",
                        "response_id": response.get("id", "") if isinstance(response, dict) else "",
                        "prompt_tokens": usage.get("prompt_tokens", ""),
                        "completion_tokens": usage.get("completion_tokens", ""),
                        "total_tokens": usage.get("total_tokens", ""),
                        "cost_usd": usage.get("cost", ""),
                        "visible_output": visible_output(response),
                        "exact_path_set": "",
                        "correct_weights": "",
                        "correct_edge_counts": "",
                        "internally_consistent": "",
                        "criterion_confusion": "",
                        "trace_primary": "",
                        "trace_components": "",
                        "trace_sequence": "",
                        "completeness_support": "",
                        "parseable": "",
                        "notes": "",
                    }
                )
                write_coding_sheet(coding_path, coding_rows)
                if completed < total_requests and args.delay > 0:
                    time.sleep(args.delay)

    print(f"\nFinished. Results: {output_dir}")
    print(f"Raw archive: {raw_path.name}")
    print(f"Coding sheet: {coding_path.name}")
    print(f"Manifest: {manifest_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
