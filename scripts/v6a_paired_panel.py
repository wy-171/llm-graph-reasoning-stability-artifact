#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v6A — Paired Protocol Panel (frozen manifest, per-block, deferred analysis)
==========================================================================
Answers ONE question causally:
  In the same graph, same slot, same time window, does giving the complete
  candidate set change Mistral's lexicographic selection — and if so, is the
  harm from "handing over the set" (ORACLE_BASE vs ES-repeat) or from showing
  the irrelevant edge-count (ORACLE_EDGE vs ORACLE_BASE)?

DESIGN (frozen by MASTER_SEED before any API call)
  20 blocks x [M2,M2,M4,M4,M8,M8] mistral slots + 1 sentinel slot/block.
  Each slot = 4 calls:
     ES1         : v2-min ES prompt, byte-identical. ALWAYS first (state marker).
     ES2         : identical ES prompt (ordinary-resampling control).
     ORACLE_BASE : full optimal set given; path + weight, NO edge count.
     ORACLE_EDGE : identical to BASE + shows edges=k.        <- only difference
  ES2/ORACLE_BASE/ORACLE_EDGE order is randomized within slot (6 orders x 20 =
  balanced serial position). ORACLE_BASE and ORACLE_EDGE share the SAME candidate
  permutation. Candidate order counterbalanced per graph.
  Sentinel: odd block Claude, even block DeepSeek, M4, all 4 conditions (哨兵,
  not strongly powered).
  Totals: mistral 480 + claude 40 + deepseek 40 = 560 calls.

PROVIDER: main experiment does NOT pin (preserve v2-min ecology); temperature/
top-p/max_tokens omitted (v2-min policy). But provider is RECORDED every call.

USAGE (blocks are meant to be REAL time blocks — run them spaced out)
  python3 v6a_paired_panel.py --init          # freeze manifest + config (once)
  python3 v6a_paired_panel.py --dry           # preview prompts + manifest balance
  export OPENROUTER_API_KEY=sk-or-...
  python3 v6a_paired_panel.py --block 1       # run ONE block (28 calls), spaced
  ... (run blocks 1..20 across >=5 dates / 3 dayparts, each launched separately)
  python3 v6a_paired_panel.py --analyze       # after all 20 blocks: one analysis
"""
import os, sys, csv, json, time, random, hashlib, urllib.request, urllib.error
from datetime import datetime, timezone
from collections import defaultdict, Counter
import bridge_es_lex_v2 as V2   # EXACT graphs + ES prompt from v2-min

EXPERIMENT_VERSION = "v6A"
MASTER_SEED = 20260815
GRAPHS = ["M2", "M4", "M8"]
N_BLOCKS = 20
SLOTS_PER_GRAPH_PER_BLOCK = 2
RULE = "min"
FOCAL = "mistral"
MODELS = {
    "mistral":  "mistralai/mistral-large-2512",
    "claude":   "anthropic/claude-sonnet-4.6",
    "deepseek": "deepseek/deepseek-v3.2-exp",
}
FOLLOWUPS = ["ES2", "ORACLE_BASE", "ORACLE_EDGE"]
TEMPERATURE = None
MAX_TRANSPORT_RETRIES = 3
MANIFEST = "manifest.csv"

def code_hash(): return hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:16]
def h(s): return hashlib.sha256(s.encode()).hexdigest()[:16]

# ---------------- ground truth / candidate sets (solver, not hand-entered) ----
def gt_of(name): return V2.ground_truth(V2.GRAPHS[name]["edges"])
def es_prompt(name): return V2.build_prompt(V2.GRAPHS[name]["edges"], RULE)   # byte-identical

def _lex_rule_text():
    # identical wording for ORACLE_BASE and ORACLE_EDGE (neutral, v2-style)
    return ("Choose exactly ONE: the lexicographically SMALLEST path. Compare the node-label "
            "sequences position by position; at the first position where they differ, the one "
            "whose label is alphabetically earlier is smaller. This is lexicographic order of "
            "node labels — NOT the number of edges and NOT the total weight.")

def oracle_prompt(name, perm, show_edges):
    gt = gt_of(name); opt = [gt["optima"][i] for i in perm]
    lines = []
    for p in opt:
        base = f"- {','.join(p)} | weight={gt['wmin']}"
        if show_edges: base += f" | edges={len(p)-1}"
        lines.append(base)
    listing = "\n".join(lines)
    return (f"The COMPLETE set of minimum-weight paths from S to T is given below "
            f"(all have equal, minimum total weight).\n\n{listing}\n\n{_lex_rule_text()}\n\n"
            f"End with EXACTLY:\nSELECTED: <node,comma,seq>\n")

def build_prompt(condition, name, perm):
    if condition in ("ES1", "ES2"): return es_prompt(name)
    if condition == "ORACLE_BASE":  return oracle_prompt(name, perm, show_edges=False)
    if condition == "ORACLE_EDGE":  return oracle_prompt(name, perm, show_edges=True)
    raise ValueError(condition)

# ---------------- parser (last block only; arrows + markdown bold) ------------
def _seq(s):
    s = s.replace("\u2192", ",").replace("->", ",").replace("-", ",")
    t = [x.strip() for x in s.split(",") if x.strip()]
    return tuple(t) if t else None

def parse(text):
    """Return (enumerated_paths_from_last_MINIMAL_block, selected)."""
    sel = None
    for ln in text.splitlines():
        u = ln.upper()
        if "SELECTED" in u:
            after = ln[u.rfind("SELECTED") + 8:]
            after = after.split(":", 1)[-1] if ":" in after else after
            sel = _seq(after.replace("*", "").replace("`", "").strip()) or sel
    idx = text.upper().rfind("MINIMAL PATHS")
    enum = []
    if idx != -1:
        for ln in text[idx:].splitlines():
            l = ln.strip()
            if l.upper().startswith("SELECTED"): break
            if l.startswith("-") and "|" in l:
                p = _seq(l.split("|", 1)[0])
                if p: enum.append(p)
    return enum, sel

def classify(sel, gt):
    if sel is None: return "unparseable"
    ystar = gt["lex_min"] if RULE == "min" else gt["lex_max"]
    if sel == ystar: return "correct"
    optset = set(gt["optima"])
    if sel in optset and (len(sel) - 1) == gt["min_edges"]: return "short_path"
    if sel in optset: return "other_optimum"
    return "off_set"

# ---------------- manifest (frozen; deterministic from MASTER_SEED) -----------
def candidate_perms(name, n):
    """n balanced permutations of the optimum indices for a graph."""
    k = len(gt_of(name)["optima"])
    base = [tuple((i + r) % k for i in range(k)) for r in range(k)] if k > 2 else [(0, 1), (1, 0)]
    reps = n // len(base)
    perms = base * reps
    assert len(perms) == n, f"{name}: {n} not divisible by {len(base)}"
    return perms

def gen_manifest():
    rng = random.Random(MASTER_SEED)
    rows = []
    # --- mistral slots: 120 = 20 blocks x (M2,M2,M4,M4,M8,M8) ---
    # follow-up orders: 6 perms x 20 = 120, shuffled
    import itertools
    all_orders = list(itertools.permutations(FOLLOWUPS))           # 6
    order_pool = all_orders * (120 // len(all_orders))             # 120
    rng.shuffle(order_pool)
    # candidate perms per graph (40 each), shuffled within graph
    perm_pool = {}
    for g in GRAPHS:
        pp = candidate_perms(g, N_BLOCKS * SLOTS_PER_GRAPH_PER_BLOCK)  # 40
        rng.shuffle(pp); perm_pool[g] = pp
    perm_idx = {g: 0 for g in GRAPHS}
    oi = 0
    for b in range(1, N_BLOCKS + 1):
        block_slots = []
        for g in GRAPHS:
            for _ in range(SLOTS_PER_GRAPH_PER_BLOCK):
                order = order_pool[oi]; oi += 1
                perm = perm_pool[g][perm_idx[g]]; perm_idx[g] += 1
                block_slots.append({"model_key": FOCAL, "graph": g,
                                    "followup_order": ",".join(order),
                                    "candidate_perm": ",".join(map(str, perm))})
        rng.shuffle(block_slots)                                   # slot order within block
        # sentinel slot (odd->claude, even->deepseek), M4, its own order+perm
        sent_model = "claude" if b % 2 == 1 else "deepseek"
        sperm = candidate_perms("M4", 4); rng.shuffle(sperm)
        block_slots.append({"model_key": sent_model, "graph": "M4",
                            "followup_order": ",".join(rng.choice(all_orders)),
                            "candidate_perm": ",".join(map(str, sperm[0]))})
        for pos, s in enumerate(block_slots, 1):
            s.update({"block_id": b, "slot_pos": pos,
                      "slot_id": f"b{b:02d}_s{pos}_{s['model_key']}_{s['graph']}"})
            rows.append(s)
    # write manifest + config + templates + code hash
    fields = ["block_id", "slot_pos", "slot_id", "model_key", "graph",
              "followup_order", "candidate_perm"]
    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in rows: w.writerow({k: r[k] for k in fields})
    cfg = {"experiment_version": EXPERIMENT_VERSION, "master_seed": MASTER_SEED,
           "n_blocks": N_BLOCKS, "graphs": GRAPHS, "rule": RULE, "conditions": ["ES1"] + FOLLOWUPS,
           "models": MODELS, "temperature": TEMPERATURE, "code_hash": code_hash(),
           "graph_hashes": {g: h(str(V2.GRAPHS[g]["edges"])) for g in GRAPHS},
           "generated_utc": datetime.now(timezone.utc).isoformat()}
    json.dump(cfg, open("experiment_config.json", "w"), indent=2)
    open("code_hash.txt", "w").write(code_hash() + "\n")
    os.makedirs("prompt_templates", exist_ok=True)
    for g in GRAPHS:
        open(f"prompt_templates/ES_{g}.txt", "w").write(es_prompt(g))
        open(f"prompt_templates/ORACLE_BASE_{g}.txt", "w").write(oracle_prompt(g, tuple(range(len(gt_of(g)["optima"]))), False))
        open(f"prompt_templates/ORACLE_EDGE_{g}.txt", "w").write(oracle_prompt(g, tuple(range(len(gt_of(g)["optima"]))), True))
    print(f"Frozen manifest: {len(rows)} slots -> {MANIFEST}")
    verify_manifest(rows)

def verify_manifest(rows=None):
    if rows is None:
        rows = list(csv.DictReader(open(MANIFEST)))
    mis = [r for r in rows if r["model_key"] == FOCAL]
    print(f"\n  mistral slots: {len(mis)} (expect 120)")
    print(f"  per graph: {Counter(r['graph'] for r in mis)}")
    print(f"  follow-up orders (expect each 20): {Counter(r['followup_order'] for r in mis)}")
    # serial position balance: for each followup condition, position across slots
    pos = defaultdict(Counter)
    for r in mis:
        for i, c in enumerate(r["followup_order"].split(","), 1):
            pos[c][i] += 1
    print("  serial-position balance (expect 40 each):")
    for c in FOLLOWUPS: print(f"    {c}: {dict(pos[c])}")
    # candidate position balance per graph
    for g in GRAPHS:
        k = len(gt_of(g)["optima"]); poscnt = defaultdict(Counter)
        for r in [x for x in mis if x["graph"] == g]:
            for slotpos, cand in enumerate(map(int, r["candidate_perm"].split(","))):
                poscnt[cand][slotpos] += 1
        exp = 40 // k
        ok = all(poscnt[c][p] == exp for c in range(k) for p in range(k))
        print(f"  {g} candidate x position balance (expect {exp} each): {'OK' if ok else 'FAIL'}")
    print(f"  sentinel slots: {Counter(r['model_key'] for r in rows if r['model_key']!=FOCAL)}")

# ---------------- OpenRouter (full provenance) -------------------------------
def call(model_str, prompt, key):
    body = {"model": model_str, "messages": [{"role": "user", "content": prompt}]}
    if TEMPERATURE is not None: body["temperature"] = TEMPERATURE
    t0 = time.time()
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(body).encode(), method="POST",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        raw = r.read().decode()
    return json.loads(raw), body, round(time.time() - t0, 2)

def run_block(n, key):
    man = [r for r in csv.DictReader(open(MANIFEST)) if int(r["block_id"]) == n]
    if not man: print(f"no slots for block {n}"); return
    ch = code_hash()
    norm = open("normalized_results.csv", "a", newline="", encoding="utf-8")
    nf = ["experiment_version","block_id","slot_id","model_key","graph","condition",
          "diagnostic_position","candidate_perm","rep","master_seed","ts_start","ts_end",
          "requested_model","returned_model","generation_id","provider","request_body_hash",
          "prompt_hash","code_hash","finish_reason","prompt_tokens","completion_tokens",
          "reasoning_tokens","latency_s","selected","selection_correct","enum_complete",
          "category","short_decoy_selected","error"]
    nw = csv.DictWriter(norm, fieldnames=nf)
    if norm.tell() == 0: nw.writeheader()
    raw_j = open("raw_responses.jsonl", "a", encoding="utf-8")
    meta_j = open("generation_metadata.jsonl", "a", encoding="utf-8")
    trans_j = open("transport_attempts.jsonl", "a", encoding="utf-8")
    print(f"===== BLOCK {n}: {len(man)} slots =====")
    for s in man:
        g = s["graph"]; gt = gt_of(g); perm = tuple(map(int, s["candidate_perm"].split(",")))
        model_str = MODELS[s["model_key"]]
        conditions = ["ES1"] + s["followup_order"].split(",")
        for dpos, cond in enumerate(conditions):   # dpos 0 = ES1 (fixed first)
            prompt = build_prompt(cond, g, perm)
            row = {k: "" for k in nf}
            row.update({"experiment_version": EXPERIMENT_VERSION, "block_id": n,
                        "slot_id": s["slot_id"], "model_key": s["model_key"], "graph": g,
                        "condition": cond, "diagnostic_position": dpos,
                        "candidate_perm": s["candidate_perm"], "rep": 1, "master_seed": MASTER_SEED,
                        "requested_model": model_str, "prompt_hash": h(prompt), "code_hash": ch})
            # transport retries (model errors are NOT retried; only transport)
            attempt = 0; d = body = None; lat = ""
            while attempt < MAX_TRANSPORT_RETRIES:
                attempt += 1; row["ts_start"] = datetime.now(timezone.utc).isoformat()
                try:
                    d, body, lat = call(model_str, prompt, key); break
                except Exception as e:
                    trans_j.write(json.dumps({"slot_id": s["slot_id"], "condition": cond,
                        "attempt": attempt, "error": f"{type(e).__name__}: {e}",
                        "ts": datetime.now(timezone.utc).isoformat()}) + "\n"); trans_j.flush()
                    if attempt >= MAX_TRANSPORT_RETRIES:
                        row["error"] = f"transport_fail_after_{attempt}"
                    time.sleep(1.5)
            row["ts_end"] = datetime.now(timezone.utc).isoformat()
            if d is not None:
                choice = (d.get("choices") or [{}])[0]
                text = (choice.get("message") or {}).get("content", "")
                usage = d.get("usage") or {}
                row.update({"returned_model": d.get("model", model_str),
                            "generation_id": d.get("id", ""),
                            "provider": d.get("provider", ""),   # if present; else look up via id
                            "request_body_hash": h(json.dumps(body, sort_keys=True)),
                            "finish_reason": choice.get("finish_reason", ""),
                            "prompt_tokens": usage.get("prompt_tokens"),
                            "completion_tokens": usage.get("completion_tokens"),
                            "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                            "latency_s": lat})
                enum, sel = parse(text); pol = classify(sel, gt)
                row["selected"] = "-".join(sel) if sel else ""
                row["category"] = pol
                row["selection_correct"] = int(pol == "correct")
                row["short_decoy_selected"] = int(pol == "short_path")
                if cond in ("ES1", "ES2"):
                    row["enum_complete"] = int(set(enum) == set(gt["optima"]))
                raw_j.write(json.dumps({"slot_id": s["slot_id"], "condition": cond,
                    "response": d}, ensure_ascii=False) + "\n"); raw_j.flush()
                meta_j.write(json.dumps({k: row[k] for k in
                    ["slot_id","condition","generation_id","provider","returned_model",
                     "finish_reason","prompt_tokens","completion_tokens","reasoning_tokens",
                     "latency_s"]}) + "\n"); meta_j.flush()
            nw.writerow(row); norm.flush()
            # RUNTIME: show only transport/parse status, NOT per-condition accuracy
            status = "ERR" if row["error"] else ("parse_fail" if row["category"]=="unparseable" else "ok")
            print(f"  {s['slot_id']:22s} {cond:12s} pos{dpos} -> {status}")
            time.sleep(0.4)
    for f in (norm, raw_j, meta_j, trans_j): f.close()
    print(f"block {n} done. (no accuracy shown by design; run --analyze after all blocks)")

# ---------------- analysis (deferred; slot-level, block-cluster bootstrap) ----
def _slotmap(rows):
    slots = defaultdict(dict)
    for r in rows:
        if r.get("error", "").strip() or r["model_key"] != FOCAL: continue
        slots[r["slot_id"]][r["condition"]] = r
    return slots

def _risk_diff_paired(slots, cond_a, cond_b):
    """P(correct|a) - P(correct|b) over slots having both; + discordant counts."""
    a=b=n=0; disc_ab=disc_ba=0
    per_block = defaultdict(lambda: [0,0,0])  # block -> [a_correct, b_correct, n]
    for sid, d in slots.items():
        if cond_a in d and cond_b in d:
            ca=int(d[cond_a]["selection_correct"]); cb=int(d[cond_b]["selection_correct"])
            a+=ca; b+=cb; n+=1
            blk=d[cond_a]["block_id"]; per_block[blk][0]+=ca; per_block[blk][1]+=cb; per_block[blk][2]+=1
            if ca and not cb: disc_ab+=1
            if cb and not ca: disc_ba+=1
    rd=(a-b)/n if n else 0
    return rd, a, b, n, disc_ab, disc_ba, per_block

def _bootstrap_ci(per_block, iters=5000, seed=1):
    rng=random.Random(seed); blocks=list(per_block); diffs=[]
    for _ in range(iters):
        samp=[rng.choice(blocks) for _ in blocks]
        na=sum(per_block[x][0] for x in samp); nb=sum(per_block[x][1] for x in samp)
        nn=sum(per_block[x][2] for x in samp)
        if nn: diffs.append((na-nb)/nn)
    diffs.sort()
    return (diffs[int(.025*len(diffs))], diffs[int(.975*len(diffs))]) if diffs else (0,0)

def analyze():
    rows=list(csv.DictReader(open("normalized_results.csv")))
    slots=_slotmap(rows)
    print("="*74+"\nv6A ANALYSIS (slot-level; block-cluster bootstrap)\n"+"="*74)
    print(f"mistral slots with data: {len(slots)}")
    for a,b,label in [("ORACLE_EDGE","ORACLE_BASE","H1: edge-count salience"),
                      ("ORACLE_BASE","ES2","H2: external-set / scaffolding")]:
        rd,ca,cb,n,dab,dba,pb=_risk_diff_paired(slots,a,b)
        lo,hi=_bootstrap_ci(pb)
        print(f"\n[{label}]  {a} vs {b}")
        print(f"  P(correct|{a})={ca}/{n}={ca/n:.3f}  P(correct|{b})={cb}/{n}={cb/n:.3f}" if n else "  no paired slots")
        print(f"  risk diff = {rd:+.3f}   block-bootstrap 95% CI [{lo:+.3f},{hi:+.3f}]")
        print(f"  discordant: {a}-only={dab}, {b}-only={dba}  (McNemar sensitivity)")
        try:
            from scipy.stats import binomtest
            if dab+dba>0: print(f"  McNemar exact two-sided p={binomtest(min(dab,dba),dab+dba,0.5).pvalue*1:.3f}")
        except: pass
    # ===== KEY: enumeration completeness — decides H2's theoretical identity =====
    # If ES enumerates the FULL optimal set yet still selects wrong, H2 is a true
    # Choosing-layer effect. If ES often omits optima, ORACLE's edge is partly
    # trivial (we just handed over a set it failed to build).
    print("\n[KEY] ES enumeration completeness (does ES list the full optimal set?)")
    for cond in ("ES1", "ES2"):
        vals = [r for r in rows if r["model_key"] == FOCAL and r["condition"] == cond
                and not r.get("error", "").strip() and r.get("enum_complete", "") != ""]
        if not vals: continue
        comp = sum(int(r["enum_complete"]) for r in vals)
        print(f"  {cond}: enum_complete {comp}/{len(vals)} = {comp/len(vals):.3f}")
    # ES2 error decomposition (set-to-choice): omission vs pure execution error
    es2_wrong = [r for r in rows if r["model_key"] == FOCAL and r["condition"] == "ES2"
                 and not r.get("error", "").strip() and r.get("enum_complete", "") != ""
                 and int(r["selection_correct"]) == 0]
    if es2_wrong:
        exec_err = sum(1 for r in es2_wrong if int(r["enum_complete"]) == 1)  # listed all, still wrong
        omission = sum(1 for r in es2_wrong if int(r["enum_complete"]) == 0)  # missed an optimum
        print(f"  ES2 errors decomposed: pure-execution (listed all, chose wrong) = {exec_err}"
              f" | omission (missed an optimum) = {omission}  [of {len(es2_wrong)} ES2 errors]")
        print(f"    -> if pure-execution dominates, H2 is a genuine Choosing-layer effect.")
    # H2 restricted to ES2 slots where enumeration WAS complete (cleanest H2)
    clean = defaultdict(dict)
    for sid, d in slots.items():
        if "ES2" in d and "ORACLE_BASE" in d and d["ES2"].get("enum_complete", "") != "" \
           and int(d["ES2"]["enum_complete"]) == 1:
            clean[sid] = d
    if clean:
        rd, ca, cb, n, dab, dba, pb = _risk_diff_paired(clean, "ORACLE_BASE", "ES2")
        lo, hi = _bootstrap_ci(pb)
        print(f"  H2 restricted to ES2-complete slots: ORACLE_BASE {ca}/{n}={ca/n:.3f} vs "
              f"ES2 {cb}/{n}={cb/n:.3f}, risk diff {rd:+.3f} CI [{lo:+.3f},{hi:+.3f}]")
        print(f"    -> ORACLE still beating ES2 HERE = the effect is not just 'we handed over the set'.")

    # secondary: ES1->ES2 transition
    print("\n[secondary] ES1->ES2 transition (state persistence):")
    tm=Counter()
    for sid,d in slots.items():
        if "ES1" in d and "ES2" in d:
            tm[(int(d["ES1"]["selection_correct"]),int(d["ES2"]["selection_correct"]))]+=1
    print(f"  correct->correct {tm[(1,1)]}  correct->wrong {tm[(1,0)]}  wrong->correct {tm[(0,1)]}  wrong->wrong {tm[(0,0)]}")
    # secondary: provider table (with denominators!)
    print("\n[secondary] provider x correct (all conditions, WITH denominators):")
    prov=defaultdict(lambda:[0,0])
    for r in rows:
        if r.get("error","").strip() or r["model_key"]!=FOCAL: continue
        prov[r.get("provider","?") or "?"][1]+=1; prov[r.get("provider","?") or "?"][0]+=int(r["selection_correct"] or 0)
    for p,(ok,tot) in sorted(prov.items()): print(f"  {p or '(blank—look up via generation_id)':40s} {ok}/{tot}")
    # secondary: condition x graph
    print("\n[secondary] correct rate by condition x graph:")
    cg=defaultdict(lambda:[0,0])
    for r in rows:
        if r.get("error","").strip() or r["model_key"]!=FOCAL or r["condition"]=="ES1": continue
        cg[(r["condition"],r["graph"])][1]+=1; cg[(r["condition"],r["graph"])][0]+=int(r["selection_correct"] or 0)
    for k in sorted(cg): print(f"  {k[0]:12s} {k[1]}: {cg[k][0]}/{cg[k][1]}")

# ---------------- main -------------------------------------------------------
def main():
    if "--init" in sys.argv: gen_manifest(); return
    if "--analyze" in sys.argv: analyze(); return
    if "--dry" in sys.argv:
        if not os.path.exists(MANIFEST): print("run --init first"); return
        verify_manifest()
        print("\n--- ES_M2 (byte-identical to v2-min) ---\n"+es_prompt("M2"))
        print("--- ORACLE_BASE_M2 ---\n"+oracle_prompt("M2",(0,1),False))
        print("--- ORACLE_EDGE_M2 ---\n"+oracle_prompt("M2",(0,1),True))
        e,s=parse("x\\n**MINIMAL PATHS:**\\n- S,A,P,Q,T | weight=12\\n**SELECTED:** S,A,T".replace("\\n","\n"))
        print("parser test -> enum",[ '-'.join(p) for p in e],"selected",'-'.join(s) if s else None)
        return
    key=os.environ.get("OPENROUTER_API_KEY")
    if not key: print("set OPENROUTER_API_KEY, or use --init/--dry/--analyze"); sys.exit(1)
    if "--block" not in sys.argv: print("specify --block N (run blocks spaced out in time)"); sys.exit(1)
    if not os.path.exists(MANIFEST): print("run --init first"); sys.exit(1)
    run_block(int(sys.argv[sys.argv.index("--block")+1]), key)

if __name__=="__main__":
    main()
