"""Run MCPEvol-Bench with a validated selective tool-harness patch.

Pipeline:
    structural change detection -> change classification -> minimal patch
    -> controlled argument probe -> semantic repair when needed
    -> validation/rollback -> cost ledger

The controlled probe never executes an MCP tool and never sees a benchmark task.
It asks the adaptation model for schema-valid example arguments using only the
candidate tool card. This keeps adaptation separate from scored task execution.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from dotenv import load_dotenv
from jsonschema import Draft202012Validator
from openai import OpenAI
from tqdm import tqdm

from custom.utils.mcp_utils import get_1mcp_status
from custom.utils.test_utils import clean_all_test_cases, generate_tree_string


load_dotenv()

META_KEYS = {"mutation_type", "task_idx"}
SERVER_ALIASES = {
    "server-sequential-thinking": "@modelcontextprotocol/server-sequential-thinking",
    "mcp-sequential-thinking": "@modelcontextprotocol/server-sequential-thinking",
    "mcp-server-time": "@guanxiong/mcp-server-time",
    "server-memory": "@guanxiong/server-memory",
    "pollinations/model-context-protocol": "@pollinations/model-context-protocol",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def canonical_server(server_name: str) -> str:
    return SERVER_ALIASES.get(server_name, server_name)


def card_payload(card: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": card["name"],
        "description": card.get("description", ""),
        "inputSchema": card.get("inputSchema", {}),
    }


def card_hash(card: dict[str, Any]) -> str:
    return sha256_json(card_payload(card))


def split_1mcp_name(full_name: str) -> tuple[str, str] | None:
    if "_1mcp_" not in full_name:
        return None
    return tuple(full_name.split("_1mcp_", 1))  # type: ignore[return-value]


def index_tool_status(tool_status: Any) -> dict[tuple[str, str], dict[str, Any]]:
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for tool in tool_status.tools:
        split_name = split_1mcp_name(tool.name)
        if split_name is None:
            continue
        server_name, tool_name = split_name
        indexed[(server_name, tool_name)] = {
            "server": server_name,
            "name": tool_name,
            "description": tool.description or "",
            "inputSchema": copy.deepcopy(tool.inputSchema),
        }
    return indexed


def load_snapshot(path: str) -> tuple[Any, dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        snapshot = json.load(handle)
    if snapshot.get("format_version") != 1 or not isinstance(snapshot.get("tools"), list):
        raise ValueError(f"Invalid harness snapshot: {path}")
    tools = []
    for tool in snapshot["tools"]:
        if not {"name", "description", "inputSchema"}.issubset(tool):
            raise ValueError(f"Malformed tool in harness snapshot: {path}")
        tools.append(SimpleNamespace(**tool))
    return SimpleNamespace(tools=tools), snapshot


def load_evolution_chain(mutation_path: str | None) -> list[dict[str, Any]]:
    if not mutation_path:
        return []
    current = Path(mutation_path).resolve()
    paths: list[Path] = []
    while current.name == "HYBRID":
        evolution_path = current / "evolution.json"
        if not evolution_path.exists():
            raise FileNotFoundError(evolution_path)
        paths.append(evolution_path)
        current = current.parent
    paths.reverse()
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def _first_mutated_tool(info: dict[str, Any]) -> str | None:
    for key in info:
        if key not in META_KEYS:
            return key
    return None


def evolve_task_tools(
    original_servers: dict[str, list[str]],
    evolution_chain: list[dict[str, Any]],
) -> tuple[dict[str, list[str]], list[dict[str, Any]]]:
    """Apply cumulative tool-level routing changes to one task's tool view."""
    evolved = copy.deepcopy(original_servers)
    routing_events: list[dict[str, Any]] = []
    for depth, evolution in enumerate(evolution_chain, start=1):
        for task_server, names in evolved.items():
            info = evolution.get(task_server)
            if not info or info.get("mutation_type") != "TOOL":
                continue

            before = list(names)
            if info.get("old_tool_name") and isinstance(info.get("new_tool_config"), dict):
                old_name = info["old_tool_name"]
                new_name = info["new_tool_config"].get("name")
                if old_name in names:
                    names.remove(old_name)
                    if new_name and new_name not in names:
                        names.append(new_name)
            elif isinstance(info.get("new_tool_config"), dict):
                new_name = info["new_tool_config"].get("name")
                if new_name and new_name not in names:
                    names.append(new_name)
            elif info.get("auto_optimizations") is not None:
                new_name = info.get("new_tool_name")
                if new_name and new_name not in names:
                    names.append(new_name)
                for optimization in info.get("auto_optimizations", []):
                    optimized_name = optimization.get("tool_name")
                    if optimized_name and optimized_name in before and optimized_name not in names:
                        names.append(optimized_name)
            elif info.get("deleted_tool_name"):
                deleted = info["deleted_tool_name"]
                replacement = info.get("migration_target_tool")
                if deleted in names:
                    names.remove(deleted)
                    if replacement and replacement not in names:
                        names.append(replacement)
            if names != before:
                routing_events.append(
                    {
                        "depth": depth,
                        "server": task_server,
                        "before": before,
                        "after": list(names),
                        "change_class": "TOOL",
                    }
                )
    return evolved, routing_events


def json_diff_paths(old: Any, new: Any, prefix: str = "") -> list[str]:
    if type(old) is not type(new):
        return [prefix or "$"]
    if isinstance(old, dict):
        paths: list[str] = []
        for key in sorted(set(old) | set(new)):
            child = f"{prefix}.{key}" if prefix else key
            if key not in old or key not in new:
                paths.append(child)
            else:
                paths.extend(json_diff_paths(old[key], new[key], child))
        return paths
    if isinstance(old, list):
        return [] if old == new else [prefix or "$"]
    return [] if old == new else [prefix or "$"]


def validate_schema_shape(schema: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(schema, dict):
        return ["inputSchema is not an object"]
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as exc:  # jsonschema emits several exception subtypes
        errors.append(f"invalid JSON Schema: {exc}")
    if schema.get("type") == "object" and not isinstance(schema.get("properties", {}), dict):
        errors.append("object schema properties must be an object")
    return errors


@dataclass
class CostLedger:
    input_cost_per_million: float | None = None
    output_cost_per_million: float | None = None
    events: list[dict[str, Any]] = field(default_factory=list)

    def record(
        self,
        *,
        phase: str,
        model: str,
        input_tokens: int | None,
        output_tokens: int | None,
        latency_ms: int,
        status: str,
        error: str | None = None,
    ) -> None:
        estimated_usd = None
        if (
            input_tokens is not None
            and output_tokens is not None
            and self.input_cost_per_million is not None
            and self.output_cost_per_million is not None
        ):
            estimated_usd = (
                input_tokens * self.input_cost_per_million
                + output_tokens * self.output_cost_per_million
            ) / 1_000_000
        self.events.append(
            {
                "phase": phase,
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "latency_ms": latency_ms,
                "status": status,
                "estimated_usd": estimated_usd,
                "cost_status": "priced" if estimated_usd is not None else "unknown",
                "error": error,
            }
        )

    def summary(self) -> dict[str, Any]:
        priced = [event["estimated_usd"] for event in self.events if event["estimated_usd"] is not None]
        return {
            "model_calls": len(self.events),
            "input_tokens": sum(event["input_tokens"] or 0 for event in self.events),
            "output_tokens": sum(event["output_tokens"] or 0 for event in self.events),
            "latency_ms": sum(event["latency_ms"] for event in self.events),
            "estimated_usd": sum(priced) if len(priced) == len(self.events) else None,
            "cost_status": "priced" if len(priced) == len(self.events) else "unknown",
        }


class AdaptationModel:
    def __init__(self, args: argparse.Namespace, ledger: CostLedger):
        self.model = args.adaptation_model or args.model_name
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("MODEL_URL"),
        )
        self.max_tokens = args.adaptation_max_tokens
        self.ledger = ledger

    def _json_call(self, phase: str, system: str, user: str) -> dict[str, Any]:
        started = time.perf_counter()
        response = None
        try:
            request_options: dict[str, Any] = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "max_completion_tokens": self.max_tokens,
                "response_format": {"type": "json_object"},
            }
            # GPT-6 reasoning requests reject custom sampling parameters.
            # Omitting temperature uses the model's supported default.
            if not self.model.startswith("gpt-6"):
                request_options["temperature"] = 0
            response = self.client.chat.completions.create(
                **request_options,
            )
            latency_ms = round((time.perf_counter() - started) * 1000)
            usage = getattr(response, "usage", None)
            content = response.choices[0].message.content
            if not content:
                raise ValueError("model returned empty content")
            parsed = json.loads(content)
            self.ledger.record(
                phase=phase,
                model=self.model,
                input_tokens=getattr(usage, "prompt_tokens", None),
                output_tokens=getattr(usage, "completion_tokens", None),
                latency_ms=latency_ms,
                status="ok",
            )
            return parsed
        except Exception as exc:
            latency_ms = round((time.perf_counter() - started) * 1000)
            usage = getattr(response, "usage", None) if response is not None else None
            self.ledger.record(
                phase=phase,
                model=self.model,
                input_tokens=getattr(usage, "prompt_tokens", None),
                output_tokens=getattr(usage, "completion_tokens", None),
                latency_ms=latency_ms,
                status="error",
                error=str(exc),
            )
            raise

    def probe(self, card: dict[str, Any]) -> tuple[str, str, dict[str, Any] | None]:
        system = (
            "You test MCP tool cards. Return JSON only with one key named arguments. "
            "Create the smallest plausible example arguments that satisfy the supplied JSON Schema. "
            "Do not explain your answer and do not call any tool."
        )
        user = canonical_json(
            {
                "tool_name": card["name"],
                "description": card.get("description", ""),
                "input_schema": card.get("inputSchema", {}),
            }
        )
        try:
            result = self._json_call("controlled_probe", system, user)
        except Exception as exc:
            return "inconclusive", f"probe model error: {exc}", None
        arguments = result.get("arguments")
        if not isinstance(arguments, dict):
            return "fail", "probe did not return an arguments object", None
        try:
            Draft202012Validator(card.get("inputSchema", {})).validate(arguments)
        except Exception as exc:
            return "fail", f"arguments failed schema validation: {exc}", arguments
        return "pass", "generated arguments satisfy the patched schema", arguments

    def repair(
        self,
        card: dict[str, Any],
        failure_reason: str,
        failed_arguments: dict[str, Any] | None,
    ) -> dict[str, Any]:
        system = (
            "You repair MCP tool-card semantics. Return JSON only with exactly two keys: "
            "description (string) and usage_rules (array of strings). Do not change, restate, "
            "or invent the tool name or JSON Schema. Make parameter expectations clear and concise."
        )
        user = canonical_json(
            {
                "tool_name": card["name"],
                "current_description": card.get("description", ""),
                "authoritative_input_schema": card.get("inputSchema", {}),
                "controlled_probe_failure": failure_reason,
                "failed_arguments": failed_arguments,
            }
        )
        repaired = self._json_call("semantic_repair", system, user)
        description = repaired.get("description")
        usage_rules = repaired.get("usage_rules")
        if not isinstance(description, str) or not description.strip():
            raise ValueError("semantic repair returned an empty description")
        if not isinstance(usage_rules, list) or not all(isinstance(rule, str) for rule in usage_rules):
            raise ValueError("semantic repair returned invalid usage_rules")
        result = copy.deepcopy(card)
        rules = [rule.strip() for rule in usage_rules if rule.strip()]
        result["description"] = description.strip()
        if rules:
            result["description"] += "\n\nUsage rules:\n" + "\n".join(f"- {rule}" for rule in rules)
        return result


def build_minimal_patch(
    original_servers: dict[str, list[str]],
    evolved_servers: dict[str, list[str]],
    original_index: dict[tuple[str, str], dict[str, Any]],
    live_index: dict[tuple[str, str], dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    dict[str, str],
    list[dict[str, Any]],
    list[str],
    list[str],
]:
    cards: list[dict[str, Any]] = []
    routing: dict[str, str] = {}
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    warnings: list[str] = []

    for task_server, evolved_names in evolved_servers.items():
        route_server = canonical_server(task_server)
        original_names = set(original_servers.get(task_server, []))
        for name in evolved_names:
            original = original_index.get((route_server, name))
            live = live_index.get((route_server, name))
            if live is None:
                message = f"target tool not found: {route_server}/{name}"
                if original is None:
                    warnings.append(f"unresolved in source and target: {route_server}/{name}")
                else:
                    errors.append(message)
                continue

            change_classes: list[str] = []
            field_paths: list[str] = []
            if original is None or name not in original_names:
                candidate = copy.deepcopy(live)
                change_classes.append("TOOL")
                field_paths.append("$")
            else:
                candidate = copy.deepcopy(original)
                if original.get("description", "") != live.get("description", ""):
                    candidate["description"] = live.get("description", "")
                    change_classes.append("DESC")
                    field_paths.append("description")
                if original.get("inputSchema", {}) != live.get("inputSchema", {}):
                    candidate["inputSchema"] = copy.deepcopy(live.get("inputSchema", {}))
                    change_classes.append("PARAM")
                    field_paths.extend(
                        f"inputSchema.{path}"
                        for path in json_diff_paths(
                            original.get("inputSchema", {}), live.get("inputSchema", {})
                        )
                    )

            schema_errors = validate_schema_shape(candidate.get("inputSchema", {}))
            if schema_errors:
                errors.extend(f"{route_server}/{name}: {error}" for error in schema_errors)
                continue

            cards.append(candidate)
            if name in routing and routing[name] != route_server:
                errors.append(
                    f"ambiguous tool routing for {name}: {routing[name]} and {route_server}"
                )
            routing[name] = route_server
            if change_classes:
                records.append(
                    {
                        "server": route_server,
                        "tool": name,
                        "change_classes": change_classes,
                        "field_paths": sorted(set(field_paths)),
                        "old_hash": card_hash(original) if original else None,
                        "candidate_hash": card_hash(candidate),
                        "live_hash": card_hash(live),
                        "status": "candidate",
                    }
                )

        removed_names = original_names - set(evolved_names)
        for name in sorted(removed_names):
            original = original_index.get((route_server, name))
            records.append(
                {
                    "server": route_server,
                    "tool": name,
                    "change_classes": ["TOOL"],
                    "field_paths": ["$"],
                    "old_hash": card_hash(original) if original else None,
                    "candidate_hash": None,
                    "live_hash": None,
                    "status": "removed",
                }
            )

    return cards, routing, records, errors, warnings


def validate_and_repair(
    cards: list[dict[str, Any]],
    routing: dict[str, str],
    records: list[dict[str, Any]],
    original_index: dict[tuple[str, str], dict[str, Any]],
    live_index: dict[tuple[str, str], dict[str, Any]],
    adaptation_model: AdaptationModel | None,
) -> tuple[list[dict[str, Any]], dict[str, str], list[dict[str, Any]]]:
    by_key = {(card["server"], card["name"]): card for card in cards}
    for record in records:
        if record["status"] == "removed":
            continue
        key = (record["server"], record["tool"])
        candidate = by_key[key]

        if adaptation_model is None:
            probe_status, reason, arguments = "not_run", "probe disabled", None
        else:
            probe_status, reason, arguments = adaptation_model.probe(candidate)
        record["initial_probe"] = {
            "status": probe_status,
            "reason": reason,
            "arguments": arguments,
        }

        promoted = candidate
        if probe_status == "fail" and adaptation_model is not None:
            try:
                repaired = adaptation_model.repair(candidate, reason, arguments)
                if repaired.get("inputSchema") != candidate.get("inputSchema"):
                    raise ValueError("semantic repair changed the authoritative schema")
                if repaired.get("name") != candidate.get("name"):
                    raise ValueError("semantic repair changed the tool name")
                second_status, second_reason, second_arguments = adaptation_model.probe(repaired)
                record["repair"] = {
                    "attempted": True,
                    "description_before_hash": sha256_json(candidate.get("description", "")),
                    "description_after_hash": sha256_json(repaired.get("description", "")),
                }
                record["validation_probe"] = {
                    "status": second_status,
                    "reason": second_reason,
                    "arguments": second_arguments,
                }
                if second_status == "pass":
                    promoted = repaired
                    record["status"] = "promoted_after_semantic_repair"
                else:
                    record["status"] = "rolled_back"
            except Exception as exc:
                record["repair"] = {"attempted": True, "error": str(exc)}
                record["status"] = "rolled_back"
        elif probe_status == "pass" or probe_status == "not_run":
            record["status"] = "promoted"
        elif probe_status == "inconclusive":
            raise RuntimeError(
                f"Probe was inconclusive for {key[0]}/{key[1]}: {reason}. "
                "Refusing to replace a live schema with the stale source schema."
            )
        else:
            record["status"] = "rolled_back"

        if record["status"] == "rolled_back":
            original = original_index.get(key)
            if original is None:
                by_key.pop(key, None)
                routing.pop(record["tool"], None)
                record["rollback_target"] = "tool_removed_from_harness"
            else:
                by_key[key] = copy.deepcopy(original)
                record["rollback_target"] = "original_card"
        else:
            live = live_index[key]
            if promoted.get("inputSchema") != live.get("inputSchema"):
                raise ValueError(f"promoted schema diverges from live schema: {key}")
            by_key[key] = promoted
            record["final_hash"] = card_hash(promoted)

    final_cards = list(by_key.values())
    return final_cards, routing, records


def unchanged_hash_check(
    cards: list[dict[str, Any]],
    records: list[dict[str, Any]],
    original_index: dict[tuple[str, str], dict[str, Any]],
) -> tuple[bool, list[str]]:
    changed = {(record["server"], record["tool"]) for record in records}
    errors: list[str] = []
    for card in cards:
        key = (card["server"], card["name"])
        if key in changed or key not in original_index:
            continue
        if card_hash(card) != card_hash(original_index[key]):
            errors.append(f"unaffected card changed: {key[0]}/{key[1]}")
    return not errors, errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run validated selective harness patches")
    parser.add_argument("--test_case_file", default="custom/data/test_cases/pilot_20_clean.json")
    parser.add_argument("--model_name", default="gpt-6-luna")
    parser.add_argument("--adaptation_model", default=None)
    parser.add_argument("--stage", required=True, choices=["early", "middle", "late"])
    parser.add_argument("--source_snapshot", required=True)
    parser.add_argument(
        "--target_snapshot",
        default=None,
        help="Optional frozen target definitions; otherwise query --mcp_server_url.",
    )
    parser.add_argument("--mutation_path", default=None)
    parser.add_argument("--mcp_server_url", default="http://127.0.0.1:3050/mcp")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--adaptation_max_tokens", type=int, default=1200)
    parser.add_argument("--input_cost_per_million", type=float, default=None)
    parser.add_argument("--output_cost_per_million", type=float, default=None)
    parser.add_argument("--disable_probe", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_cases", type=int, default=None)
    parser.add_argument("--fschat_flag", action="store_true")
    args = parser.parse_args()
    if args.stage != "early" and not args.mutation_path:
        parser.error("--mutation_path is required for middle and late")
    return args


def main() -> None:
    args = parse_args()
    original_status, snapshot_metadata = load_snapshot(args.source_snapshot)
    if args.target_snapshot:
        live_status, target_snapshot_metadata = load_snapshot(args.target_snapshot)
    else:
        live_status, _ = asyncio.run(get_1mcp_status(args.mcp_server_url))
        target_snapshot_metadata = None
    original_index = index_tool_status(original_status)
    live_index = index_tool_status(live_status)
    evolution_chain = load_evolution_chain(args.mutation_path)

    with open(args.test_case_file, "r", encoding="utf-8") as handle:
        cases = clean_all_test_cases(json.load(handle))
    if args.max_cases is not None:
        cases = cases[: args.max_cases]

    result_root = Path(args.output_path) / args.model_name
    answer_dir = result_root / "ANSWER"
    audit_dir = result_root / "PATCH_AUDIT"
    answer_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    run_summary = {
        "strategy": "selective_patch",
        "stage": args.stage,
        "source_snapshot": str(Path(args.source_snapshot).resolve()),
        "source_snapshot_captured_at_utc": snapshot_metadata.get("captured_at_utc"),
        "target_snapshot": (
            str(Path(args.target_snapshot).resolve()) if args.target_snapshot else None
        ),
        "target_snapshot_captured_at_utc": (
            target_snapshot_metadata.get("captured_at_utc")
            if target_snapshot_metadata is not None
            else None
        ),
        "mutation_depth": len(evolution_chain),
        "test_case_file": str(Path(args.test_case_file).resolve()),
        "test_case_sha256": hashlib.sha256(Path(args.test_case_file).read_bytes()).hexdigest(),
        "cases": [],
    }

    for test_case in tqdm(cases, desc=f"selective patch {args.stage}", unit="case"):
        task_id = test_case["idx"]
        answer_path = answer_dir / f"{task_id}.json"
        audit_path = audit_dir / f"{task_id}.json"
        if answer_path.exists() and not args.overwrite and not args.dry_run:
            continue

        started = time.perf_counter()
        evolved_servers, routing_events = evolve_task_tools(test_case["servers"], evolution_chain)
        cards, routing, records, errors, warnings = build_minimal_patch(
            test_case["servers"], evolved_servers, original_index, live_index
        )
        if errors:
            raise RuntimeError(f"Task {task_id} patch construction failed: {'; '.join(errors)}")

        ledger = CostLedger(args.input_cost_per_million, args.output_cost_per_million)
        adaptation_model = None if args.disable_probe else AdaptationModel(args, ledger)
        cards, routing, records = validate_and_repair(
            cards, routing, records, original_index, live_index, adaptation_model
        )
        hashes_ok, hash_errors = unchanged_hash_check(cards, records, original_index)
        if not hashes_ok:
            raise RuntimeError(f"Task {task_id} minimality check failed: {'; '.join(hash_errors)}")

        adaptation_ms = round((time.perf_counter() - started) * 1000)
        audit = {
            "task_id": task_id,
            "strategy": "selective_patch",
            "stage": args.stage,
            "mutation_depth": len(evolution_chain),
            "routing_events": routing_events,
            "warnings": warnings,
            "patch_records": records,
            "changed_card_count": len(records),
            "changed_field_count": sum(len(record["field_paths"]) for record in records),
            "unchanged_hash_check": hashes_ok,
            "adaptation_time_ms": adaptation_ms,
            "cost_ledger": ledger.events,
            "cost_summary": ledger.summary(),
            "final_tool_hash": sha256_json([card_payload(card) for card in cards]),
        }
        audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
        run_summary["cases"].append(
            {
                "task_id": task_id,
                "changed_card_count": audit["changed_card_count"],
                "changed_field_count": audit["changed_field_count"],
                "adaptation_time_ms": adaptation_ms,
                **ledger.summary(),
            }
        )

        if args.dry_run:
            continue
        if not cards:
            raise RuntimeError(f"Task {task_id} has no tools after patching")

        # Keep FastChat optional for patch construction and dry-run auditing.
        repo_src = Path(__file__).resolve().parents[2]
        fastchat_root = repo_src / "FastChat"
        if fastchat_root.exists() and str(fastchat_root) not in sys.path:
            sys.path.insert(0, str(fastchat_root))
        from custom.agents.SimpleOpenAIAgent import ChatModel

        openai_agent = ChatModel(
            model_name=args.model_name,
            api_key=os.getenv("OPENAI_API_KEY"),
            model_url=os.getenv("MODEL_URL"),
            tools=[card_payload(card) for card in cards],
            max_round=max(10, 2 * len(test_case.get("tool_calls", []))),
            tool_server_map=routing,
            file_exists=generate_tree_string("./anotation_path"),
            mcp_server_url=args.mcp_server_url,
        )
        answer, trajectory, stop_flag, exceeded_flag = openai_agent.run_with_server(
            test_case["question"], args.fschat_flag
        )
        saved = copy.deepcopy(test_case)
        saved.update(
            {
                "answer": answer,
                "trajectory_messages": trajectory,
                "stop": stop_flag,
                "exceeded": exceeded_flag,
                "forced_finalization": openai_agent.forced_finalization,
                "involved_tools": [card_payload(card) for card in cards],
                "harness_strategy": "selective_patch",
                "stage": args.stage,
                "patch_audit": str(audit_path.resolve()),
                "adaptation_cost": ledger.summary(),
                "unchanged_hash_check": hashes_ok,
            }
        )
        answer_path.write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_path = result_root / "PATCH_RUN_SUMMARY.json"
    summary_path.write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote selective-patch audit to {audit_dir}")
    if not args.dry_run:
        print(f"Wrote answers to {answer_dir}")


if __name__ == "__main__":
    main()
