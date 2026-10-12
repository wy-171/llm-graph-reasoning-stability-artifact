#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bridge experiment v2 — proxy-conflict traps, n=30, lex-min/max mirror
=====================================================================
Follows the v1 pilot finding: the ONE failure was mistral doing PROXY
SUBSTITUTION (picking the fewest-edges path instead of the lexicographic
winner) on a proxy-conflict graph with a SHARED-PREFIX trap. v1 was 89/90,
so the signal exists but is too rare to quantify. This version amplifies it.

WHAT CHANGED vs v1
  1) GRAPH GENERATOR: builds proxy-conflict traps automatically at each
     multiplicity (2/4/8 optima). In every generated graph the lexicographic
     winner has MORE edges than a competing optimum, and the fewest-edges
     decoy SHARES A PREFIX with the winner (the "tempting" structure that
     tripped mistral in v1). No hand-entered weights.
  2) n = 30 per (model, graph) + Wilson 95% CIs in the summary.
  3) MIRROR PROBE: each graph is asked twice — lexicographically SMALLEST
     and LARGEST — to test rule-responsiveness ("should change, does it?").

RUN
  export OPENROUTER_API_KEY=sk-or-...
  python3 bridge_es_lex_v2.py --dry     # self-check + prompt preview, no calls
  python3 bridge_es_lex_v2.py           # full run -> bridge_v2_<stamp>.csv
  python3 bridge_es_lex_v2.py --summary bridge_v2_XXXX.csv   # re-print stats
"""
import os, sys, csv, json, time, math, string, urllib.request, urllib.error
from datetime import datetime, timezone
from collections import defaultdict

MODELS = [
    {"key": "claude",   "model": "anthropic/claude-sonnet-4.6"},
    {"key": "mistral",  "model": "mistralai/mistral-large-2512"},
    {"key": "deepseek", "model": "deepseek/deepseek-v3.2-exp"},
]
REPEATS = 10
TEMPERATURE = None
RULES = ["max"]     # lexicographic smallest AND largest (mirror probe)

# ----------------------------------------------------------------------------
# GRAPH GENERATOR — proxy-conflict traps with shared-prefix decoys
# ----------------------------------------------------------------------------
# Construction (weights chosen so every optimum totals W=12; decoys total 13):
#   * A "winner spine" S -> A -> C1 -> C2 -> ... -> T  (many edges, starts 'A')
#     is the LONGEST optimum, and because it starts with 'A' it is the
#     lexicographic MINIMUM. It is NOT fewest-edges.
#   * A "decoy" that SHARES the prefix S -> A, then jumps to T in fewer edges:
#     S -> A -> T  (fewest edges) — the tempting proxy pick.
#   * Extra optima start with later letters (B, C, ...) to raise multiplicity.
#   * A near-optimal distractor at W+1 for realism.
# Everything is verified by ground_truth() + self-check below.

def _split(total, k):
    """k positive integer weights summing to total (as even as possible)."""
    base, extra = divmod(total, k)
    return [base + (1 if i < extra else 0) for i in range(k)]

def make_trap(multiplicity, W=12, a=2):
    """
    Proxy-conflict trap; EVERY optimum totals W:
      winner spine  S -> A -> P -> Q -> T   (4 edges, starts 'A' => lex-min)
      shared-prefix S -> A -> T             (2 edges, fewest edges => proxy decoy)
      extra optima  S -> x -> T             (2 edges) for x in B,C,... to raise m
      distractor    S -> d -> T             (weight W+1)
    The shared edge S->A has ONE weight (a); both the rest-of-spine and A->T
    equal W-a, so spine = a+(W-a) = W and decoy = a+(W-a) = W. No double-assign.
    """
    edges, seen = [], set()
    def add(u, v, w):
        if (u, v) not in seen:
            seen.add((u, v)); edges.append((u, v, w))
    add("S", "A", a)                                   # shared first edge
    ap, pq, qt = _split(W - a, 3)                       # rest of spine sums to W-a
    add("A", "P", ap); add("P", "Q", pq); add("Q", "T", qt)
    add("A", "T", W - a)                               # decoy 2nd edge => S-A-T = W
    letters = [c for c in string.ascii_uppercase if c not in "SAPQT"]
    made, li = 2, 0
    while made < multiplicity and li < len(letters):
        x = letters[li]; li += 1
        add("S", x, 1); add(x, "T", W - 1)             # S->x->T = W (2 edges)
        made += 1
    if li < len(letters):
        d = letters[li]; add("S", d, 1); add(d, "T", W)  # S->d->T = W+1 distractor
    return edges

GRAPHS = {
    "M2": {"edges": make_trap(2), "expect": {"n_opt": 2}},
    "M4": {"edges": make_trap(4), "expect": {"n_opt": 4}},
    "M8": {"edges": make_trap(8), "expect": {"n_opt": 8}},
    # paste real P4 / A-H here later if you want them alongside
}

# ----------------------------------------------------------------------------
# GROUND TRUTH
# ----------------------------------------------------------------------------
def all_st_paths(edges, src="S", dst="T"):
    adj = {}
    for u, v, w in edges: adj.setdefault(u, []).append((v, w))
    out = []
    def dfs(n, path, wsum, seen):
        if n == dst: out.append((tuple(path), wsum)); return
        for v, w in adj.get(n, []):
            if v in seen: continue
            seen.add(v); dfs(v, path + [v], wsum + w, seen); seen.remove(v)
    dfs(src, [src], 0, {src})
    return out

def winner(optima, rule):   # r(O) for lexicographic min or max
    return (min if rule == "min" else max)(optima, key=lambda p: tuple(p))

def ground_truth(edges):
    paths = all_st_paths(edges)
    if not paths: raise ValueError("no S->T path")
    wmin = min(w for _, w in paths)
    optima = [p for p, w in paths if w == wmin]
    min_e = min(len(p) - 1 for p in optima)
    few = [p for p in optima if len(p) - 1 == min_e]
    fewest = few[0] if len(few) == 1 else None
    return {"paths": paths, "wmin": wmin, "optima": optima,
            "edge_counts": sorted(len(p) - 1 for p in optima),
            "min_edges": min_e,                       # fewest-edge count among optima
            "fewest_edges_winner": fewest,            # unique shortest optimum, or None
            "lex_min": winner(optima, "min"), "lex_max": winner(optima, "max"),
            "n_total": len(paths),
            "n_min1": sum(1 for _, w in paths if w == wmin + 1)}

def fmt(p): return "-".join(p)

# ----------------------------------------------------------------------------
# PROMPT / PARSE / GRADE
# ----------------------------------------------------------------------------
def build_prompt(edges, rule):
    block = "\n".join(f"{u} -> {v} (weight {w})" for u, v, w in edges)
    which = "SMALLEST" if rule == "min" else "LARGEST"
    return f"""You are given a weighted directed graph. Nodes are single uppercase letters. S is the source and T is the target.

Edges:
{block}

Do BOTH steps.

STEP 1 — Enumerate every path from S to T whose TOTAL EDGE WEIGHT is the minimum possible. For EACH such path report the path, its total weight, and its number of edges. (Report each path's own edge count; do not assume they share one.)

STEP 2 — Among the minimum-weight paths from Step 1, choose exactly ONE using this rule:
  Lexicographic order on the node-label sequence. Write each path as its sequence of node labels from S to T and compare two paths position by position; at the first position where they differ, the one whose node label is alphabetically earlier is lexicographically smaller. Select the lexicographically {which} path.
The selection rule is lexicographic order of node labels — NOT the number of edges and NOT the total weight.

End your answer with EXACTLY this block and nothing after it:
MINIMAL PATHS:
- <node,comma,seq> | weight=<int> | edges=<int>
SELECTED: <node,comma,seq>
"""

def _seq(s):
    t = [x.strip() for x in s.replace("-", ",").split(",") if x.strip()]
    return tuple(t) if t else None

def parse_response(text):
    enum, sel = [], None
    for raw in text.splitlines():
        ln = raw.strip()
        if ln.upper().startswith("SELECTED:"):
            sel = _seq(ln.split(":", 1)[1])
        elif ln.startswith("-") and "|" in ln:
            s = _seq(ln.split("|", 1)[0].lstrip("-"))
            if s: enum.append(s)
    return enum, sel

def grade(text, gt, rule):
    enum, sel = parse_response(text)
    O = set(gt["optima"]); ystar = gt["lex_min"] if rule == "min" else gt["lex_max"]
    es = set(enum)
    parse_ok = sel is not None
    sel_ok = (sel == ystar)
    pol_suff = bool(es) and (winner(list(es), rule) == ystar)
    # proxy substitution: the winner is NOT a shortest optimum, yet the model
    # picked a shortest optimum (used an edge-count heuristic instead of the rule).
    min_e, lex_e = gt["min_edges"], len(ystar) - 1
    proxy = int(lex_e > min_e and sel is not None and sel != ystar
                and (len(sel) - 1) == min_e)
    if not parse_ok: cat = "unparseable"
    elif sel_ok and es == O: cat = "correct_full"
    elif sel_ok: cat = "correct_incomplete_but_sufficient"
    elif pol_suff: cat = "execution_error"
    elif es: cat = "decision_altering_omission"
    else: cat = "compound"
    return {"parse_ok": int(parse_ok), "n_enumerated": len(enum),
            "enum_complete": int(es == O), "selection_correct": int(sel_ok),
            "policy_sufficient": int(pol_suff), "proxy_substitution": proxy,
            "selected": fmt(sel) if sel else "", "category": cat}

# ----------------------------------------------------------------------------
# STATS (Wilson 95% CI)
# ----------------------------------------------------------------------------
def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0, 0.0)
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = (z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))) / d
    return (p, max(0, c-h), min(1, c+h))

def summarize(rows):
    print("\n" + "=" * 74)
    print("SUMMARY — selection accuracy & proxy-substitution (Wilson 95% CI)")
    print("=" * 74)
    cell = defaultdict(lambda: {"n": 0, "ok": 0, "proxy": 0, "ex": 0})
    for r in rows:
        if r.get("error", "").strip(): continue
        k = (r["graph"], r["rule"], r["model_key"]); c = cell[k]
        c["n"] += 1
        c["ok"] += int(r["selection_correct"] == "1")
        c["proxy"] += int(r["proxy_substitution"] == "1")
        c["ex"] += int(r["category"] == "execution_error")
    for (g, rule, m) in sorted(cell):
        c = cell[(g, rule, m)]; p, lo, hi = wilson(c["ok"], c["n"])
        print(f"  {g:4s} lex-{rule:3s} {m:9s} | acc {c['ok']:2d}/{c['n']:2d} "
              f"={p:.2f} [{lo:.2f},{hi:.2f}]  proxy_sub={c['proxy']:2d}  exec_err={c['ex']:2d}")

# ----------------------------------------------------------------------------
# OPENROUTER
# ----------------------------------------------------------------------------
def call(model_cfg, prompt, key):
    body = {"model": model_cfg["model"], "messages": [{"role": "user", "content": prompt}]}
    if TEMPERATURE is not None: body["temperature"] = TEMPERATURE
    if "provider" in model_cfg: body["provider"] = model_cfg["provider"]
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(body).encode(), method="POST",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        d = json.loads(r.read().decode())
    return d["choices"][0]["message"]["content"], (d.get("usage") or {}), d.get("model", model_cfg["model"])

# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def self_check():
    print("=" * 74 + "\nSELF-CHECK — generated proxy-conflict traps:\n" + "=" * 74)
    ready = []
    for name, g in GRAPHS.items():
        gt = ground_truth(g["edges"])
        conflict = (len(gt["lex_min"]) - 1) > gt["min_edges"]   # winner is not a shortest optimum
        print(f"\n[{name}] optima={len(gt['optima'])} weight={gt['wmin']} "
              f"edge_counts={gt['edge_counts']} total={gt['n_total']} min+1={gt['n_min1']}")
        print(f"  lex_min={fmt(gt['lex_min'])}  lex_max={fmt(gt['lex_max'])}  "
              f"fewest_edges={fmt(gt['fewest_edges_winner']) if gt['fewest_edges_winner'] else 'not-unique'}")
        print(f"  PROXY-CONFLICT: {'YES (lex_min != fewest_edges)' if conflict else 'no'}")
        exp = g.get("expect", {})
        if "n_opt" in exp:
            print(f"  expect n_opt={exp['n_opt']}: {'PASS' if len(gt['optima'])==exp['n_opt'] else '*** MISMATCH ***'}")
        ready.append(name)
    return ready

def main():
    if "--summary" in sys.argv:
        path = sys.argv[sys.argv.index("--summary") + 1]
        summarize(list(csv.DictReader(open(path)))); return
    dry = "--dry" in sys.argv
    ready = self_check()
    if dry:
        print("\n--- PROMPT PREVIEW (M2, lex-min) ---\n")
        print(build_prompt(GRAPHS[ready[0]]["edges"], "min"))
        print("[--dry] no calls."); return
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key: print("\nERROR: set OPENROUTER_API_KEY (or --dry)."); sys.exit(1)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = f"bridge_v2_{stamp}.csv"
    fields = ["run_id","timestamp_utc","model_key","returned_model","graph","rule","rep",
              "wmin","target","fewest_edges","parse_ok","n_enumerated","enum_complete",
              "selection_correct","policy_sufficient","proxy_substitution","selected",
              "category","prompt_tokens","completion_tokens","total_tokens","error","raw"]
    n_total = len(ready) * len(RULES) * len(MODELS) * REPEATS
    print(f"\nRunning {n_total} calls -> {out}\n")
    rows_mem, done = [], 0
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader()
        for name in ready:
            gt = ground_truth(GRAPHS[name]["edges"])
            for rule in RULES:
                prompt = build_prompt(GRAPHS[name]["edges"], rule)
                target = gt["lex_min"] if rule == "min" else gt["lex_max"]
                for m in MODELS:
                    for rep in range(1, REPEATS + 1):
                        done += 1
                        rid = f"{m['key']}_{name}_{rule}_{rep:02d}"
                        row = {f: "" for f in fields}
                        row.update({"run_id": rid,
                                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                                    "model_key": m["key"], "graph": name, "rule": rule,
                                    "rep": rep, "wmin": gt["wmin"], "target": fmt(target),
                                    "fewest_edges": fmt(gt["fewest_edges_winner"]) if gt["fewest_edges_winner"] else ""})
                        try:
                            text, usage, ret = call(m, prompt, key)
                            row["returned_model"] = ret
                            row["prompt_tokens"] = usage.get("prompt_tokens", "")
                            row["completion_tokens"] = usage.get("completion_tokens", "")
                            row["total_tokens"] = usage.get("total_tokens", "")
                            row["raw"] = text.replace("\n", "\\n")
                            row.update(grade(text, gt, rule))
                        except urllib.error.HTTPError as e:
                            row["error"] = f"HTTP {e.code}: {e.read().decode('utf-8','ignore')[:150]}"
                        except Exception as e:
                            row["error"] = f"{type(e).__name__}: {e}"
                        w.writerow(row); fh.flush(); rows_mem.append(row)
                        print(f"[{done}/{n_total}] {rid:26s} {row.get('category', row['error'][:20])}")
                        time.sleep(0.4)
    summarize(rows_mem)
    print(f"\nDone. -> {out}")

if __name__ == "__main__":
    main()
