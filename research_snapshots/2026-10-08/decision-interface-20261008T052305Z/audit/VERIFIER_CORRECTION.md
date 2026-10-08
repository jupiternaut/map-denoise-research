# Verifier schema correction

Target: liekkas, decision-interface-20261008T052305Z. This correction changes only the deterministic artifact checker. The checker author also wrote the new fixture producer; this is not an independent scientific audit. No old run, observation, prediction, calibration, evaluation, method, source lock, or earlier report is edited.

The first report, `audit/REPLAY_RECHECK.json`, is retained exactly. Its SHA256 is `b32204a64365f4702a7430b6054c6a9b43a0fcd672328103ea0e39ef34589d0d`; its recorded checker source SHA256 is `8be080f0f5e35089de5690555446d6d6849ef263d8977eca1b74e9bf7c99fa6e`. It had44 passing checks and4 failures, all named `seal_has_files`. Confirmation evaluation was still pending. The source/history hashes, current-stage hashes,324 historical decision comparisons, replay metrics, and replay gate checks had passed.

The four failures were false alarms caused by the checker's unsupported assumption that every historical SEAL uses a `files` map. Reading those seal files and their original read-only producer code established the following actual contracts:

| Historical seal | Hash fields and exact targets |
| --- | --- |
| footprint-support-20261008T025757Z/observed_stage/SEAL.json | `observations_sha256` hashes sibling `OBSERVATIONS.json`; `curves` maps120 curve paths to hashes. |
| footprint-support-20261008T025757Z/oracle_stage/SEAL.json | `observations_sha256` hashes sibling `OBSERVATIONS.json`; `curves` maps150 curve paths to hashes. |
| footprint-support-20261008T025757Z/decisions/SEAL.json | `decisions_sha256` hashes sibling `DECISIONS.json`; `observation_seals` hashes the named sibling stage SEAL files. |
| mixed-pixel-20261008T041249Z/decisions/SEAL.json | `sha256` hashes sibling `PREDICTIONS.json`; `observation_seals` hashes the named sibling stage SEAL files. |

`verify_seal` now dispatches on these actual schemas and verifies every referenced file and upstream seal. The existing `files` schema remains supported. An unrecognized schema is now recorded as incomplete semantic seal checking, not labeled corruption; byte-level `SOURCE_LOCK.history` checking remains separate and unchanged. No hash equality or metric tolerance has been weakened.

Targeted read-only validation of the four corrected seals passed all8 schema/hash checks, covering278 referenced files or upstream seals. That check wrote no report and did not run confirmation evaluation. The updated script also passed syntax parsing.

This correction does not reinterpret scientific outcomes or authorize a stage. B2 permission remains controlled by the recorded experiment gate. A later full recheck will use a new report filename after confirmation artifacts are complete; the initial report will not be overwritten.

After `confirmation/evaluation/SEAL.json` became available, the complete checker was run and wrote the new `audit/FINAL_RECHECK_CORRECTED.json`. All78 checks passed with no pending items:17 source files and1857 historical files were unchanged;18 seals verified; all2268 replay and1584 confirmation decision rows matched independently recomputed summaries and gates;324 historical S0P/N_P selections matched exactly. Calibration and confirmation each contained48 objects with the registered four-way balance and disjoint batch seeds. The corrected report SHA256 is `d3f752ad6c52f74879d2b3200b77b15e9174c607db05704cdb1875b51383361a`. This is a successful artifact/arithmetic check, not a claim that the scientific confirmation gate passed.
