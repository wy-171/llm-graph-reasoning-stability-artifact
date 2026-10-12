# Provenance and archival notes

## Frozen design and request construction

The manifest and prompt templates were generated before the panel was collected. The master seed controls the schedule, follow-up ordering, and oracle candidate permutations; it is not an API decoding seed. The request body contained the model identifier and one user message. Temperature, top-p, maximum tokens, and a decoding seed were omitted.

For every completed call, `normalized_results.csv` records a prompt hash and request-body hash. Within each focal-model slot, ES1 and ES2 have matching prompt and request-body hashes, confirming that they used the same stored prompt and request construction while being submitted as separate fresh-context calls.

## Collection-code hashes

The call-level table contains two collection-code hashes:

| Blocks | Rows | Recorded hash |
| --- | ---: | --- |
| 1–16 | 448 | `400c031ab63a955a` |
| 17–20 | 112 | `d7207d7284c62401` |

The configuration file and `design/code_hash.txt` retain the initialization hash `400c031ab63a955a`. The archived source package contains the later collection-script snapshot, whose SHA-256 prefix is `d7207d7284c62401`. The exact earlier script snapshot corresponding to `400c031ab63a955a` was not retained in that package. This repository preserves that limitation instead of presenting the later file as the missing earlier snapshot.

The graph definitions and prompt-building module, all emitted prompt templates, frozen manifest, per-call prompt and request hashes, complete successful response objects, normalized rows, and transport logs are present. These materials permit verification of the analyzed records and matched ES1/ES2 request construction, but they do not reconstruct the unavailable earlier source file byte for byte.

## Missing calls

Two of 560 planned calls failed after three HTTP 429 responses:

- block 15, slot `b15_s3_mistral_M4`, condition ES1
- block 15, slot `b15_s1_mistral_M8`, condition ORACLE_BASE

They appear in `normalized_results.csv` and their attempts appear in `transport_attempts.jsonl`. No result was imputed. Consequently, paired comparisons use 119 observed pairs, the pooled M4 ES sample contains 79 selections, and the M8 ORACLE_BASE graph cell contains 39 completed calls.

## Release scope

Working archives also contained earlier exploratory rounds with different graph sets, model sets, and schedules. They were excluded from this frozen v6A release, along with unrelated test sets and working backups. No differently structured file has been substituted for a v6A record.
