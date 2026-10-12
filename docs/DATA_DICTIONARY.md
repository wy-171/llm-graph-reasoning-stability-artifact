# Data dictionary

## File relationships

`design/manifest.csv` contains one row per task slot. Each slot schedules four conditions. `data/normalized_results.csv` therefore contains four rows per slot, including rows for calls that exhausted transport retries. The two JSONL files for responses and compact metadata contain only completed calls. Records join on the pair `slot_id` and `condition`.

## `design/manifest.csv`

| Field | Meaning |
| --- | --- |
| `block_id` | Collection block, 1 through 20 |
| `slot_pos` | Slot order within a block |
| `slot_id` | Unique slot identifier |
| `model_key` | Short model label used by the study code |
| `graph` | Graph condition: M2, M4, or M8 |
| `followup_order` | Scheduled order of the three conditions after ES1 |
| `candidate_perm` | Zero-based permutation applied to the oracle candidate listing |

## `data/normalized_results.csv`

| Field | Meaning |
| --- | --- |
| `experiment_version` | Experiment label (`v6A`) |
| `block_id` | Collection block |
| `slot_id` | Task slot identifier |
| `model_key` | Focal or secondary model label |
| `graph` | Graph condition |
| `condition` | `ES1`, `ES2`, `ORACLE_BASE`, or `ORACLE_EDGE` |
| `diagnostic_position` | Within-slot call position; ES1 is position 0 |
| `candidate_perm` | Oracle candidate-order permutation assigned to the slot |
| `rep` | Scheduled repetition indicator in the collection code |
| `master_seed` | Seed used to generate the manifest |
| `ts_start`, `ts_end` | UTC timestamps for the final request attempt and completion handling |
| `requested_model`, `returned_model` | Requested and API-returned model identifiers |
| `generation_id` | Provider generation identifier |
| `provider` | Provider reported by the API |
| `request_body_hash` | First 16 hexadecimal characters of SHA-256 over the sorted serialized request body |
| `prompt_hash` | First 16 hexadecimal characters of SHA-256 over the exact prompt text |
| `code_hash` | First 16 hexadecimal characters of SHA-256 over the collection script used for that row |
| `finish_reason` | API finish reason |
| `prompt_tokens`, `completion_tokens`, `reasoning_tokens` | Provider-reported token usage |
| `latency_s` | Request duration in seconds |
| `selected` | Parsed final path, canonicalized to a hyphen-separated ordered node sequence |
| `selection_correct` | `1` when `selected` equals the unique lexicographic winner; `0` otherwise |
| `enum_complete` | For ES1/ES2, `1` when the parsed candidate listing equals the complete optimal set; blank for oracle conditions |
| `category` | Selection category: `correct`, `short_path`, `other_optimum`, `off_set`, or `unparseable` |
| `short_decoy_selected` | `1` when the selected answer is the designated shorter tied path rather than the lexicographic winner |
| `error` | Blank on success; terminal transport status on a failed call |

The two failed calls have blank response-derived fields. They remain in the normalized table so the planned count and missingness are auditable.

## JSONL files

`data/raw_responses.jsonl` stores the complete API response object under `response`, together with `slot_id` and `condition`. `data/generation_metadata.jsonl` stores a compact subset of the same provider metadata. `data/transport_attempts.jsonl` records each caught transport exception, its attempt number, and timestamp; it includes transient errors followed by successful calls as well as the two terminal failures.

## Derived result files

`results/answer_stability.csv` reports the number of successful selections, accuracy, modal-answer stability, and Shannon entropy in natural-log units. `results/answer_distributions.csv` gives the underlying canonical-answer counts. `results/paired_comparisons.csv` uses only slots in which both listed conditions were observed. Confidence intervals resample the 20 collection blocks with replacement for 5,000 replicates using seed 1. The McNemar p-values are exact and two-sided.
