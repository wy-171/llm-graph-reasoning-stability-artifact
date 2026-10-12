#!/usr/bin/env python3
"""Run the A/B graph-reasoning pilot through OpenRouter.

Examples:
  python3 graph_api_pilot.py --dry-run
  python3 graph_api_pilot.py --models claude --repetitions 1
  python3 graph_api_pilot.py

The script deliberately omits temperature so the provider default is used,
matching the archived request condition from the earlier P4 API pilot.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import pathlib
import sys
import time
import urllib.error
import urllib.request


ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

MODELS = {
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

GRAPH_EDGES = {
    "A": """S → A: 1
A → D: 2
D → T: 3
A → T: 6
S → B: 2
B → C: 2
C → T: 2""",
    "B": """S → A: 1
A → D: 3
D → T: 3
A → T: 5
S → B: 2
B → C: 2
C → T: 2""",
}

PROMPT_TEMPLATE = """Consider the following directed, weighted graph.

Only the directed edges listed below exist:

{edges}

Find all distinct minimum-total-weight paths from S to T.

For each reported path, provide:
- the node sequence;
- the total weight;
- the number of edges.

Briefly explain your method and include enough intermediate
calculations to verify that the reported paths are all the
minimum-weight paths.

Equivalent wording and formatting are acceptable as long as
the requested information is clear."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def make_body(model_key: str, graph: str) -> dict:
    cfg = MODELS[model_key]
    return {
        "model": cfg["model"],
        "messages": [
            {
                "role": "user",
                "content": PROMPT_TEMPLATE.format(edges=GRAPH_EDGES[graph]),
            }
        ],
        "provider": {
            "only": [cfg["provider"]],
            "allow_fallbacks": False,
        },
    }


def request_openrouter(api_key: str, body: dict, timeout: int) -> tuple[int, dict]:
    request = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"raw_error": raw}
        return exc.code, payload


def content_from(payload: dict) -> str:
    try:
        value = payload["choices"][0]["message"]["content"]
        return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    except (KeyError, IndexError, TypeError):
        return ""


def build_schedule(repetitions: int) -> list[str]:
    # ABBA/BAAB-style interleaving, truncated after each graph reaches N runs.
    base = ["A", "B", "B", "A", "B", "A", "A", "B"]
    result: list[str] = []
    counts = {"A": 0, "B": 0}
    index = 0
    while counts["A"] < repetitions or counts["B"] < repetitions:
        graph = base[index % len(base)]
        index += 1
        if counts[graph] >= repetitions:
            continue
        result.append(graph)
        counts[graph] += 1
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the controlled A/B graph pilot and archive every raw response."
    )
    parser.add_argument(
        "--models",
        default="claude,mistral,deepseek",
        help="Comma-separated keys: claude,mistral,deepseek",
    )
    parser.add_argument("--repetitions", type=int, default=5, help="Runs per graph/model")
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds between requests")
    parser.add_argument("--timeout", type=int, default=180, help="HTTP timeout in seconds")
    parser.add_argument("--output", help="Output directory (default: timestamped folder)")
    parser.add_argument("--dry-run", action="store_true", help="Show schedule without API calls")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.repetitions < 1:
        raise SystemExit("--repetitions must be at least 1")

    model_keys = [item.strip() for item in args.models.split(",") if item.strip()]
    unknown = [item for item in model_keys if item not in MODELS]
    if unknown:
        raise SystemExit(f"Unknown model key(s): {', '.join(unknown)}")

    schedule = build_schedule(args.repetitions)
    total = len(model_keys) * len(schedule)
    print(f"Models: {', '.join(model_keys)}")
    print(f"Graph order per model: {' '.join(schedule)}")
    print(f"Total requests: {total}")

    if args.dry_run:
        for key in model_keys:
            cfg = MODELS[key]
            print(f"- {key}: {cfg['model']} via {cfg['provider']} (fallback disabled)")
        return 0

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Missing OPENROUTER_API_KEY environment variable.", file=sys.stderr)
        print("macOS/Linux: read -s 'OPENROUTER_API_KEY?OpenRouter key: '; export OPENROUTER_API_KEY; echo", file=sys.stderr)
        return 2

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = pathlib.Path(args.output or f"graph_api_pilot_{stamp}").resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    jsonl_path = output_dir / "raw_runs.jsonl"
    csv_path = output_dir / "coding_sheet.csv"
    manifest_path = output_dir / "manifest.json"

    manifest = {
        "created_at_utc": utc_now(),
        "endpoint": ENDPOINT,
        "models": {key: MODELS[key] for key in model_keys},
        "repetitions_per_graph_model": args.repetitions,
        "schedule_per_model": schedule,
        "temperature": "omitted; provider default",
        "prompts": {
            graph: PROMPT_TEMPLATE.format(edges=edges)
            for graph, edges in GRAPH_EDGES.items()
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_fields = [
        "run_id", "timestamp_utc", "model_key", "requested_model", "requested_provider",
        "graph", "repetition", "http_status", "returned_model", "returned_provider",
        "response_id", "prompt_tokens", "completion_tokens", "total_tokens", "visible_output",
        "exact_path_set", "correct_weights", "correct_edge_counts", "internally_consistent",
        "criterion_confusion", "trace_type", "completeness_support", "parseable", "notes",
    ]
    graph_counts = {key: {"A": 0, "B": 0} for key in model_keys}
    completed = 0

    with jsonl_path.open("w", encoding="utf-8") as jsonl_file, csv_path.open(
        "w", encoding="utf-8", newline=""
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
        writer.writeheader()

        for model_key in model_keys:
            for graph in schedule:
                graph_counts[model_key][graph] += 1
                repetition = graph_counts[model_key][graph]
                completed += 1
                run_id = f"{model_key}_{graph}{repetition:02d}"
                body = make_body(model_key, graph)
                started_at = utc_now()
                print(f"[{completed}/{total}] {run_id} ... ", end="", flush=True)

                try:
                    status, response_body = request_openrouter(api_key, body, args.timeout)
                except Exception as exc:  # Preserve unexpected transport errors as data.
                    status = 0
                    response_body = {"transport_error": f"{type(exc).__name__}: {exc}"}

                visible_output = content_from(response_body)
                record = {
                    "run_id": run_id,
                    "timestamp_utc": started_at,
                    "model_key": model_key,
                    "condition": graph,
                    "repetition": repetition,
                    "request": body,
                    "http_status": status,
                    "response": response_body,
                }
                jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                jsonl_file.flush()

                usage = response_body.get("usage") or {}
                row = {
                    "run_id": run_id,
                    "timestamp_utc": started_at,
                    "model_key": model_key,
                    "requested_model": body["model"],
                    "requested_provider": MODELS[model_key]["provider"],
                    "graph": graph,
                    "repetition": repetition,
                    "http_status": status,
                    "returned_model": response_body.get("model", ""),
                    "returned_provider": response_body.get("provider", ""),
                    "response_id": response_body.get("id", ""),
                    "prompt_tokens": usage.get("prompt_tokens", ""),
                    "completion_tokens": usage.get("completion_tokens", ""),
                    "total_tokens": usage.get("total_tokens", ""),
                    "visible_output": visible_output,
                    "exact_path_set": "",
                    "correct_weights": "",
                    "correct_edge_counts": "",
                    "internally_consistent": "",
                    "criterion_confusion": "",
                    "trace_type": "",
                    "completeness_support": "",
                    "parseable": "",
                    "notes": "" if status == 200 else json.dumps(response_body, ensure_ascii=False),
                }
                writer.writerow(row)
                csv_file.flush()
                print("OK" if status == 200 else f"HTTP {status}")
                if completed < total and args.delay > 0:
                    time.sleep(args.delay)

    print(f"\nFinished. Results: {output_dir}")
    print(f"Raw archive: {jsonl_path.name}")
    print(f"Coding sheet: {csv_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
