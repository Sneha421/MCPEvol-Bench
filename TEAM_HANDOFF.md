# Selective Patch Benchmark Handoff

This document summarizes the adaptation work added on top of MCPEvol-Bench and points to the final, shareable artifacts. The main repository README still describes the base benchmark and server setup.

## What this work evaluates

The experiment compares three tool-harness strategies as MCP servers evolve:

1. **No update** — keep the original tool knowledge unchanged.
2. **Full rewrite** — replace the task's tool cards with definitions from the current server.
3. **Selective patch** — detect changed tool cards and update only the affected names, descriptions, schema fields, and routes.

The selective strategy is designed to reduce both failure modes:

- **Under-update:** the harness retains stale knowledge needed by the evolved server.
- **Over-update:** unrelated tool knowledge is rewritten and previously working behavior regresses.

## Main implementation

### Selective patch runner

`src/custom/real_mcp/selective_patch_runner.py` implements this pipeline:

```text
snapshot comparison
  -> cumulative tool-routing evolution
  -> minimal field-level patch
  -> controlled schema-valid argument probe
  -> optional semantic description repair
  -> validation or rollback
  -> per-task audit and cost ledger
```

Important properties:

- The controlled probe receives only the candidate tool card, not the benchmark question.
- The probe never executes an MCP tool.
- Live JSON Schema remains authoritative and cannot be changed by semantic repair.
- Unaffected cards are checked by hash to detect accidental over-updating.
- Failed repairs roll back to the original card or remove an invalid newly added card.
- Every task receives a `PATCH_AUDIT/<task_id>.json` record.

### Supporting changes

- `src/custom/real_mcp/tool_level_test.py`
  - saves and loads frozen harness snapshots;
  - applies cumulative evolution chains in chronological order;
  - supports running all selected tasks or changed-only tasks.
- `src/custom/agents/SimpleOpenAIAgent.py`
  - treats a content-only assistant response as normal completion;
  - performs a final no-tools synthesis when the round limit is reached;
  - records whether forced finalization occurred.
- `src/custom/analysis/answer/evaluation.py`
  - parses direct or fenced JSON judge responses;
  - validates aggregate scores before including them;
  - fails clearly when no judge call produces a valid score.
- `src/custom/analysis/answer/evolutionary_volatility.py`
  - aligns task scores across three stages and calculates ECS.

## Final pilot data

The pilot contains 20 tasks:

`src/custom/data/test_cases/pilot_20_clean.json`

The frozen original harness is:

`src/custom/data/harness_snapshots/original.json`

Final results are split across two result bundles because later reruns replaced selected stages:

| Strategy | Original | Iteration 3 | Iteration 5 | ECS file |
|---|---|---|---|---|
| No update | `benchmark_clean_r1/no_update_original` | `benchmark_clean_r1/no_update_iter3` | `benchmark_clean_r1/no_update_iter5` | `benchmark_clean_r1/ECS_NO_UPDATE.json` |
| Full rewrite | `benchmark_clean_r2/full_rewrite_original` | `benchmark_clean_r1/full_rewrite_iter3` | `benchmark_clean_r1/full_rewrite_iter5` | `benchmark_clean_r2/ECS_FULL_REWRITE.json` |
| Selective patch | `benchmark_clean_r2/selective_original` | `benchmark_clean_r2/selective_iter3` | `benchmark_clean_r2/selective_iter5` | `benchmark_clean_r2/ECS_SELECTIVE.json` |

All paths above are relative to `src/custom/data/` and contain both `ANSWER` and `EVAL` directories. Selective results additionally contain `PATCH_AUDIT`.

## Results

Higher ECS is better.

| Strategy | ECS |
|---|---:|
| No update | 5.643489 |
| Full rewrite | 5.305757 |
| Selective patch | **5.739725** |

Selective patch improved ECS by approximately:

- `+0.096` over no update;
- `+0.434` over full rewrite.

Using a behavioral threshold of a 1.5-point task-fulfillment decline from no update to full rewrite, Iteration 3 produced seven over-update candidates: tasks `2`, `21`, `29`, `34`, `235`, `249`, and `281`. This is a comparative diagnostic, not proof of structural corruption, because independent agent and judge runs can vary.

The selective structural audit found:

- zero unaffected-card hash failures;
- zero rolled-back or unpromoted patch records;
- three unresolved references across tasks 5 and 8. These references were absent from both source and target snapshots, so they are recorded as coverage gaps rather than evolution-induced patch failures.

## Selective adaptation cost

| Stage | Changed tasks | Model calls | Input tokens | Output tokens | Adaptation time |
|---|---:|---:|---:|---:|---:|
| Original | 0 | 0 | 0 | 0 | 0.199 s |
| Iteration 3 | 8 | 11 | 4,444 | 671 | 18.818 s |
| Iteration 5 | 11 | 19 | 7,616 | 1,433 | 49.529 s |

Across the evolved stages, selective adaptation used 30 model calls, 12,060 input tokens, 2,104 output tokens, and 68.347 seconds. Dollar cost is unknown because the original runs did not supply per-million-token pricing flags.

## Environment

Run commands from `src/` unless noted otherwise.

```bash
python3 -m venv ../.venv
source ../.venv/bin/activate
pip install -r ../requirements.txt
```

Create a local `.env` file; it is intentionally ignored by Git:

```dotenv
OPENAI_API_KEY=your_key
MODEL_URL=your_openai_compatible_base_url
```

Install and configure 1MCP as described in `README.md`. Before each evolved run, launch 1MCP with the matching original, Iteration 3, or Iteration 5 server configuration at `http://127.0.0.1:3050/mcp`.

## Reproducing selective runs

### Original stage

This run uses frozen source and target definitions, so it does not require a live server for patch construction. Tool execution still requires the original 1MCP server unless `--dry_run` is supplied.

```bash
python custom/real_mcp/selective_patch_runner.py \
  --stage early \
  --test_case_file custom/data/test_cases/pilot_20_clean.json \
  --model_name gpt-6-luna \
  --source_snapshot custom/data/harness_snapshots/original.json \
  --target_snapshot custom/data/harness_snapshots/original.json \
  --output_path custom/data/reproduction/selective_original
```

### Iteration 3

Start the Iteration 3 1MCP configuration, then run:

```bash
python custom/real_mcp/selective_patch_runner.py \
  --stage middle \
  --test_case_file custom/data/test_cases/pilot_20_clean.json \
  --model_name gpt-6-luna \
  --source_snapshot custom/data/harness_snapshots/original.json \
  --mutation_path custom/data/mutation/HYBRID \
  --mcp_server_url http://127.0.0.1:3050/mcp \
  --output_path custom/data/reproduction/selective_iter3
```

### Iteration 5

Start the Iteration 5 1MCP configuration. The nested mutation path applies both evolution levels cumulatively:

```bash
python custom/real_mcp/selective_patch_runner.py \
  --stage late \
  --test_case_file custom/data/test_cases/pilot_20_clean.json \
  --model_name gpt-6-luna \
  --source_snapshot custom/data/harness_snapshots/original.json \
  --mutation_path custom/data/mutation/HYBRID/HYBRID \
  --mcp_server_url http://127.0.0.1:3050/mcp \
  --output_path custom/data/reproduction/selective_iter5
```

Use `--dry_run` to construct and audit patches without executing benchmark tasks. Use `--overwrite` only when intentionally replacing existing outputs.

## Evaluating answers

Evaluate each generated `ANSWER` directory:

```bash
python custom/analysis/answer/evaluation.py \
  --answer_path custom/data/reproduction/selective_iter3/gpt-6-luna/ANSWER \
  --model gpt-6-luna \
  --num_runs 5
```

Then calculate ECS from the three `EVAL` directories:

```bash
python custom/analysis/answer/evolutionary_volatility.py \
  --dir_a custom/data/reproduction/selective_original/gpt-6-luna/EVAL \
  --dir_b custom/data/reproduction/selective_iter3/gpt-6-luna/EVAL \
  --dir_c custom/data/reproduction/selective_iter5/gpt-6-luna/EVAL \
  --prefer mean \
  --out custom/data/reproduction/ECS_SELECTIVE.json
```

## Inspecting patch cost

```bash
for mode in selective_original selective_iter3 selective_iter5; do
  dir="custom/data/benchmark_clean_r2/$mode/gpt-6-luna/PATCH_AUDIT"
  echo "===== $mode ====="
  jq -s '{
    cases: length,
    changed_tasks: (map(select(.changed_card_count > 0)) | length),
    changed_cards: ([.[].changed_card_count] | add // 0),
    changed_fields: ([.[].changed_field_count] | add // 0),
    model_calls: ([.[].cost_summary.model_calls] | add // 0),
    input_tokens: ([.[].cost_summary.input_tokens] | add // 0),
    output_tokens: ([.[].cost_summary.output_tokens] | add // 0),
    model_latency_ms: ([.[].cost_summary.latency_ms] | add // 0),
    adaptation_time_ms: ([.[].adaptation_time_ms] | add // 0)
  }' "$dir"/*.json
done
```

Use the per-task `PATCH_AUDIT` files as the source of truth. A resumed run can write a partial or empty `PATCH_RUN_SUMMARY.json` when all answer files already exist.

## Known caveats

- Generated result JSON contains absolute paths from the machine that produced the run. These are provenance fields; use repository-relative paths when rerunning.
- Tasks 5 and 8 reference tools missing from both frozen and live snapshots.
- Do not commit `.env`, runtime sessions, PID files, `node_modules`, or the downloaded `evol-servers` bundle.
- The server configuration files may require machine-specific path replacement before 1MCP can start.

