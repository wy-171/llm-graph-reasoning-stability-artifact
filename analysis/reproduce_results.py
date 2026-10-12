#!/usr/bin/env python3
"""Reproduce and validate the v6A result tables using only the Python standard library."""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "normalized_results.csv"
DEFAULT_OUTPUT = ROOT / "results"
FOCAL_MODEL = "mistral"
GRAPHS = ("M2", "M4", "M8")
CONDITIONS = ("ES1", "ES2", "ORACLE_BASE", "ORACLE_EDGE")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def is_success(row: dict[str, str]) -> bool:
    return not row.get("error", "").strip()


def proportion(value: int, total: int) -> float:
    return value / total if total else float("nan")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def answer_summaries(rows: list[dict[str, str]]):
    focal = [r for r in rows if r["model_key"] == FOCAL_MODEL and is_success(r)]
    definitions = (("ES1+ES2", {"ES1", "ES2"}), ("ES1", {"ES1"}), ("ES2", {"ES2"}))
    summaries: list[dict[str, object]] = []
    distributions: list[dict[str, object]] = []

    for graph in GRAPHS:
        for sample, allowed in definitions:
            subset = [r for r in focal if r["graph"] == graph and r["condition"] in allowed]
            counts = Counter(r["selected"] for r in subset)
            n = len(subset)
            correct = sum(r["selection_correct"] == "1" for r in subset)
            modal_count = max(counts.values())
            entropy = -sum((count / n) * math.log(count / n) for count in counts.values())
            summaries.append(
                {
                    "graph": graph,
                    "sample": sample,
                    "R": n,
                    "correct": correct,
                    "accuracy": fmt(proportion(correct, n)),
                    "modal_count": modal_count,
                    "answer_stability": fmt(proportion(modal_count, n)),
                    "entropy_nats": fmt(entropy),
                    "distinct_answers": len(counts),
                }
            )
            for answer, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
                answer_rows = [r for r in subset if r["selected"] == answer]
                correct_flags = {r["selection_correct"] for r in answer_rows}
                if len(correct_flags) != 1:
                    raise AssertionError(f"Inconsistent correctness labels for {graph} {answer}")
                distributions.append(
                    {
                        "graph": graph,
                        "sample": sample,
                        "canonical_answer": answer,
                        "count": count,
                        "proportion": fmt(proportion(count, n)),
                        "correct_answer": next(iter(correct_flags)),
                    }
                )
    return summaries, distributions


def slot_map(rows: list[dict[str, str]]) -> dict[str, dict[str, dict[str, str]]]:
    slots: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        if row["model_key"] == FOCAL_MODEL and is_success(row):
            slots[row["slot_id"]][row["condition"]] = row
    return slots


def exact_mcnemar_p(a_only: int, b_only: int) -> float:
    discordant = a_only + b_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(a_only, b_only) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def bootstrap_ci(per_block: dict[str, list[int]], iterations: int = 5000, seed: int = 1):
    rng = random.Random(seed)
    blocks = list(per_block)
    differences = []
    for _ in range(iterations):
        sampled = [rng.choice(blocks) for _ in blocks]
        a_correct = sum(per_block[block][0] for block in sampled)
        b_correct = sum(per_block[block][1] for block in sampled)
        total = sum(per_block[block][2] for block in sampled)
        differences.append((a_correct - b_correct) / total)
    differences.sort()
    return differences[int(0.025 * iterations)], differences[int(0.975 * iterations)]


def paired_comparison(slots, a: str, b: str, label: str) -> dict[str, object]:
    pairs = [conditions for conditions in slots.values() if a in conditions and b in conditions]
    per_block: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    a_correct = b_correct = a_only = b_only = path_agreement = 0
    for conditions in pairs:
        ca = int(conditions[a]["selection_correct"])
        cb = int(conditions[b]["selection_correct"])
        a_correct += ca
        b_correct += cb
        a_only += int(ca == 1 and cb == 0)
        b_only += int(ca == 0 and cb == 1)
        path_agreement += int(conditions[a]["selected"] == conditions[b]["selected"])
        block = conditions[a]["block_id"]
        per_block[block][0] += ca
        per_block[block][1] += cb
        per_block[block][2] += 1

    n = len(pairs)
    risk_difference = (a_correct - b_correct) / n
    ci_low, ci_high = bootstrap_ci(per_block)
    return {
        "comparison": label,
        "condition_a": a,
        "condition_b": b,
        "pairs": n,
        "a_correct": a_correct,
        "b_correct": b_correct,
        "a_accuracy": fmt(proportion(a_correct, n)),
        "b_accuracy": fmt(proportion(b_correct, n)),
        "risk_difference_a_minus_b": fmt(risk_difference),
        "ci_low": fmt(ci_low),
        "ci_high": fmt(ci_high),
        "a_only_correct": a_only,
        "b_only_correct": b_only,
        "mcnemar_exact_two_sided_p": f"{exact_mcnemar_p(a_only, b_only):.6f}",
        "exact_path_agreement": path_agreement,
    }


def accuracy_tables(rows: list[dict[str, str]]):
    graph_rows: list[dict[str, object]] = []
    for condition in CONDITIONS:
        for graph in GRAPHS:
            subset = [
                r
                for r in rows
                if r["model_key"] == FOCAL_MODEL
                and r["condition"] == condition
                and r["graph"] == graph
                and is_success(r)
            ]
            correct = sum(r["selection_correct"] == "1" for r in subset)
            graph_rows.append(
                {
                    "condition": condition,
                    "graph": graph,
                    "correct": correct,
                    "completed": len(subset),
                    "accuracy": fmt(proportion(correct, len(subset))),
                }
            )

    secondary_rows: list[dict[str, object]] = []
    for model in ("claude", "deepseek"):
        for condition in CONDITIONS:
            subset = [
                r
                for r in rows
                if r["model_key"] == model and r["condition"] == condition and is_success(r)
            ]
            correct = sum(r["selection_correct"] == "1" for r in subset)
            secondary_rows.append(
                {
                    "model_key": model,
                    "condition": condition,
                    "correct": correct,
                    "completed": len(subset),
                    "accuracy": fmt(proportion(correct, len(subset))),
                }
            )
    return graph_rows, secondary_rows


def validate(rows, stability, paired, graph_rows, secondary_rows) -> None:
    assert len(rows) == 560
    assert sum(is_success(r) for r in rows) == 558
    errors = [r for r in rows if not is_success(r)]
    assert {(r["graph"], r["condition"]) for r in errors} == {
        ("M4", "ES1"),
        ("M8", "ORACLE_BASE"),
    }

    successful_slots = slot_map(rows)
    matched_es = [d for d in successful_slots.values() if "ES1" in d and "ES2" in d]
    assert len(matched_es) == 119
    assert all(d["ES1"]["prompt_hash"] == d["ES2"]["prompt_hash"] for d in matched_es)
    assert all(d["ES1"]["request_body_hash"] == d["ES2"]["request_body_hash"] for d in matched_es)

    focal_es1 = [r for r in rows if r["model_key"] == FOCAL_MODEL and r["condition"] == "ES1" and is_success(r)]
    focal_es2 = [r for r in rows if r["model_key"] == FOCAL_MODEL and r["condition"] == "ES2" and is_success(r)]
    assert sum(r["enum_complete"] == "1" for r in focal_es1) == 119
    assert sum(r["enum_complete"] == "1" for r in focal_es2) == 119
    es2_wrong = [r for r in focal_es2 if r["selection_correct"] == "0"]
    assert len(es2_wrong) == 48
    assert sum(r["enum_complete"] == "1" for r in es2_wrong) == 47

    stability_key = {(r["graph"], r["sample"]): r for r in stability}
    expected_stability = {
        ("M2", "ES1+ES2"): (80, 50, "0.625", "0.625", "0.662"),
        ("M4", "ES1+ES2"): (79, 35, "0.443", "0.481", "0.908"),
        ("M8", "ES1+ES2"): (80, 64, "0.800", "0.800", "0.500"),
        ("M2", "ES1"): (40, 26, "0.650", "0.650", "0.647"),
        ("M2", "ES2"): (40, 24, "0.600", "0.600", "0.673"),
        ("M4", "ES1"): (39, 19, "0.487", "0.487", "0.910"),
        ("M4", "ES2"): (40, 16, "0.400", "0.525", "0.899"),
        ("M8", "ES1"): (40, 32, "0.800", "0.800", "0.500"),
        ("M8", "ES2"): (40, 32, "0.800", "0.800", "0.500"),
    }
    for key, expected in expected_stability.items():
        got = stability_key[key]
        observed = (
            got["R"],
            got["correct"],
            got["accuracy"],
            got["answer_stability"],
            got["entropy_nats"],
        )
        assert observed == expected, (key, observed, expected)

    paired_key = {r["comparison"]: r for r in paired}
    assert (
        paired_key["ES1 vs ES2"]["pairs"],
        paired_key["ES1 vs ES2"]["a_correct"],
        paired_key["ES1 vs ES2"]["b_correct"],
        paired_key["ES1 vs ES2"]["a_only_correct"],
        paired_key["ES1 vs ES2"]["b_only_correct"],
        paired_key["ES1 vs ES2"]["exact_path_agreement"],
    ) == (119, 77, 72, 29, 24, 63)
    assert (
        paired_key["ORACLE_BASE vs ES2"]["pairs"],
        paired_key["ORACLE_BASE vs ES2"]["a_correct"],
        paired_key["ORACLE_BASE vs ES2"]["b_correct"],
    ) == (119, 100, 72)
    assert (
        paired_key["ORACLE_EDGE vs ORACLE_BASE"]["pairs"],
        paired_key["ORACLE_EDGE vs ORACLE_BASE"]["a_correct"],
        paired_key["ORACLE_EDGE vs ORACLE_BASE"]["b_correct"],
    ) == (119, 89, 100)

    graph_key = {(r["condition"], r["graph"]): r for r in graph_rows}
    for key, expected in {
        ("ES2", "M2"): (24, 40),
        ("ES2", "M4"): (16, 40),
        ("ES2", "M8"): (32, 40),
        ("ORACLE_BASE", "M2"): (39, 40),
        ("ORACLE_BASE", "M4"): (34, 40),
        ("ORACLE_BASE", "M8"): (27, 39),
    }.items():
        assert (graph_key[key]["correct"], graph_key[key]["completed"]) == expected

    secondary_key = {(r["model_key"], r["condition"]): r for r in secondary_rows}
    for condition in CONDITIONS:
        assert (secondary_key[("claude", condition)]["correct"], secondary_key[("claude", condition)]["completed"]) == (10, 10)
    for condition in CONDITIONS:
        expected = (9, 10) if condition == "ES2" else (10, 10)
        assert (secondary_key[("deepseek", condition)]["correct"], secondary_key[("deepseek", condition)]["completed"]) == expected


def summary_text(rows, stability, paired) -> str:
    focal = [r for r in rows if r["model_key"] == FOCAL_MODEL and is_success(r)]
    es1 = [r for r in focal if r["condition"] == "ES1"]
    es2 = [r for r in focal if r["condition"] == "ES2"]
    es2_wrong = [r for r in es2 if r["selection_correct"] == "0"]
    complete_wrong = sum(r["enum_complete"] == "1" for r in es2_wrong)

    by_name = {r["comparison"]: r for r in paired}
    lines = [
        "v6A reproducibility summary",
        "===========================",
        "",
        f"Planned calls: {len(rows)}",
        f"Completed calls: {sum(is_success(r) for r in rows)}",
        f"Failed calls: {sum(not is_success(r) for r in rows)}",
        "",
        "Candidate discovery in successful focal ES calls:",
        f"  ES1: {sum(r['enum_complete'] == '1' for r in es1)}/{len(es1)} complete",
        f"  ES2: {sum(r['enum_complete'] == '1' for r in es2)}/{len(es2)} complete",
        f"  Incorrect ES2 selections with complete candidate set: {complete_wrong}/{len(es2_wrong)}",
        "",
        "Pooled ES1+ES2 answer stability:",
    ]
    for row in stability:
        if row["sample"] == "ES1+ES2":
            lines.append(
                f"  {row['graph']}: R={row['R']}, accuracy={row['accuracy']}, "
                f"S_ans={row['answer_stability']}, H={row['entropy_nats']} nats"
            )
    lines.extend(["", "Paired condition comparisons:"])
    for name in ("ES1 vs ES2", "ORACLE_BASE vs ES2", "ORACLE_EDGE vs ORACLE_BASE"):
        row = by_name[name]
        p = float(row["mcnemar_exact_two_sided_p"])
        p_text = "<0.001" if p < 0.001 else f"={p:.3f}"
        lines.append(
            f"  {name}: {row['a_correct']}/{row['pairs']} vs {row['b_correct']}/{row['pairs']}; "
            f"RD={float(row['risk_difference_a_minus_b']):+.3f}; "
            f"95% CI [{float(row['ci_low']):+.3f}, {float(row['ci_high']):+.3f}]; "
            f"McNemar p{p_text}"
        )
    lines.append("")
    lines.append("All manuscript-level validation checks passed.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = read_rows(args.data)
    stability, distributions = answer_summaries(rows)
    slots = slot_map(rows)
    paired = [
        paired_comparison(slots, "ES1", "ES2", "ES1 vs ES2"),
        paired_comparison(slots, "ORACLE_BASE", "ES2", "ORACLE_BASE vs ES2"),
        paired_comparison(slots, "ORACLE_EDGE", "ORACLE_BASE", "ORACLE_EDGE vs ORACLE_BASE"),
    ]
    graph_rows, secondary_rows = accuracy_tables(rows)
    validate(rows, stability, paired, graph_rows, secondary_rows)

    output = args.output
    write_csv(output / "answer_stability.csv", list(stability[0]), stability)
    write_csv(output / "answer_distributions.csv", list(distributions[0]), distributions)
    write_csv(output / "paired_comparisons.csv", list(paired[0]), paired)
    write_csv(output / "graph_level_accuracy.csv", list(graph_rows[0]), graph_rows)
    write_csv(output / "secondary_model_accuracy.csv", list(secondary_rows[0]), secondary_rows)
    text = summary_text(rows, stability, paired)
    (output / "summary.txt").write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
