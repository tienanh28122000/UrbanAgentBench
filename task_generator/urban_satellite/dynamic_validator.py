"""
dynamic_validator.py
====================
Component 2: Dynamic Oracle Verification filter for urban_map_web scenarios.

Processes samples that already passed the Static Check
(generated_all_static_validator.json) and further filters them using
end-to-end simulation results produced by tau2-bench.

Per-sample scoring
------------------
  action_score  : mean of per-action ``action_reward`` (0.0/1.0) stored in
                  the simulation's ``reward_info.action_checks``.
                  Uses the same pre-computed rewards as the official benchmark.py
                  so results are directly comparable.

  nl_score      : fraction of ``reward_info.nl_assertions`` whose ``met``
                  field is True.

Filter policy
-------------
  A sample PASSES iff:
    action_score >= --action-threshold   (default 1.00 = 100 %)
    nl_score     >= --nl-threshold       (default 0.80 =  80 %)

  Samples that have no matching simulation entry are handled according to
  --no-sim-policy:
    "pass"  – keep the sample without scoring (conservative inclusion)
    "fail"  – reject the sample as unverifiable (default, conservative)

Outputs
-------
  generated_all_dynamic_validator.json        – samples that passed
  generated_all_dynamic_validator_error.json  – rejected samples with scores
    [optional] file from --save-result          – same format as --sim,
                                                                                                but keeps only passed task_id

Usage
-----
    python dynamic_validator.py \\
        --input    generated/generated_all_static_validator.json \\
        --sim      ../../data/simulations/<timestamp>_urban_map_web_*.json \\
        --tools    tools.json \\
        [--output  generated/generated_all_dynamic_validator.json] \\
        [--error-output generated/generated_all_dynamic_validator_error.json] \\
        [--save-result generated/generated_all_dynamic_validator_simulations.json] \\
        [--action-threshold 1.0] \\
        [--nl-threshold 0.8] \\
        [--no-sim-policy fail|pass]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# Reuse the official benchmark scoring logic
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SCRIPTS_DIR))
from benchmark import load_tools, is_name_only_tool, extract_tool_calls, compute_action_scores


# ---------------------------------------------------------------------------
# Scoring helpers (uses benchmark.py's official compute_action_scores)
# ---------------------------------------------------------------------------

_TASK_SUFFIX_RE = re.compile(r"_\d+$")


def _get_base_task_id(task_id: str) -> str:
    return _TASK_SUFFIX_RE.sub("", task_id)


def _build_passed_subtask_summary(all_samples: list[dict], valid_samples: list[dict]) -> list[dict[str, Any]]:
    passed_counts: Counter[str] = Counter()

    # Initialize all base task IDs found in input with 0 to ensure they appear even if none passed
    for sample in all_samples:
        if isinstance(sample, dict):
            sample_id = sample.get("id")
            if isinstance(sample_id, str) and sample_id:
                # Accessing it ensures it exists in the Counter keys
                passed_counts[_get_base_task_id(sample_id)] += 0

    # Count passed subtasks
    for sample in valid_samples:
        sample_id = sample.get("id")
        if isinstance(sample_id, str) and sample_id:
            passed_counts[_get_base_task_id(sample_id)] += 1

    return [
        {
            "task_id": task_id,
            "passed_subtasks": count,
        }
        for task_id, count in sorted(passed_counts.items())
    ]

def _compute_sample_scores(sim_entry: dict, tools: dict) -> tuple[float, float, int, int]:
    """
    Recompute per-sample action and NL scores using the official benchmark.py
    logic (Group 1 name-only / Group 2 strict matching with sequential consume).

    Returns
    -------
    action_score : float   mean of per-action scores
    nl_score     : float   fraction of nl_assertions with met == True
    n_actions    : int     number of action checks evaluated
    n_nl         : int     number of NL assertions evaluated
    """
    reward_info: dict = sim_entry.get("reward_info") or {}
    messages: list[dict] = sim_entry.get("messages") or []

    # ---- action score (recomputed, not pre-computed) ----
    action_checks: list[dict] = reward_info.get("action_checks") or []
    if action_checks:
        actual_calls = extract_tool_calls(messages)
        scores = compute_action_scores(action_checks, actual_calls, tools)
        
        # Ghi đè vào kết quả để khi lưu file save-result sẽ có các trường này
        for check, score in zip(action_checks, scores):
            check["action_match"] = bool(score)
            check["action_reward"] = float(score)

        action_score = sum(scores) / len(scores)
        n_actions = len(scores)
    else:
        action_score = 0.0
        n_actions = 0

    # ---- NL assertion score (pre-computed met flag is reliable) ----
    nl_assertions: list[dict] = reward_info.get("nl_assertions") or []
    if nl_assertions:
        met_count = sum(1 for nl in nl_assertions if isinstance(nl, dict) and nl.get("met") is True)
        nl_score = met_count / len(nl_assertions)
        n_nl = len(nl_assertions)
    else:
        nl_score = 0.0
        n_nl = 0

    return action_score, nl_score, n_actions, n_nl


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dynamic oracle verification filter for urban_map_web benchmark scenarios."
    )
    script_dir = Path(__file__).resolve().parent
    gen_dir    = script_dir / "generated"

    parser.add_argument(
        "--input",
        default=str(gen_dir / "generated_all_static_validator.json"),
        help="Path to the static-validated samples JSON (input).",
    )
    parser.add_argument(
        "--sim",
        required=True,
        help="Path to the tau2-bench simulation result JSON file.",
    )
    parser.add_argument(
        "--tools",
        default=str(script_dir / "tools.json"),
        help="Path to the domain tools.json for Group 1/2 action matching.",
    )
    parser.add_argument(
        "--output",
        default=str(gen_dir / "generated_all_dynamic_validator.json"),
        help="Path to write samples that passed both thresholds.",
    )
    parser.add_argument(
        "--error-output",
        default=str(gen_dir / "generated_all_dynamic_validator_error.json"),
        help="Path to write rejected samples with scores and reasons.",
    )
    parser.add_argument(
        "--summary-output",
        default=str(gen_dir / "generated_all_dynamic_validator_task_pass_summary.csv"),
        help="Path to write the passed sub-task count summary CSV grouped by base task_id.",
    )
    parser.add_argument(
        "--save-result",
        dest="save_result",
        default=None,
        help=(
            "Optional path to write a filtered simulation JSON with the same format "
            "as --sim, keeping only entries whose task_id passed dynamic validation."
        ),
    )
    parser.add_argument(
        "--action-threshold",
        type=float,
        default=1.0,
        help="Minimum required action accuracy (0.0–1.0). Default: 1.0 (100%%).",
    )
    parser.add_argument(
        "--nl-threshold",
        type=float,
        default=0.8,
        help="Minimum required NL-assertion accuracy (0.0–1.0). Default: 0.8 (80%%).",
    )
    parser.add_argument(
        "--no-sim-policy",
        choices=["pass", "fail"],
        default="fail",
        help=(
            "How to handle samples with no matching simulation entry. "
            "'fail' (default): reject as unverifiable. "
            "'pass': keep without scoring."
        ),
    )
    args = parser.parse_args()

    input_path  = Path(args.input)
    sim_path    = Path(args.sim)
    output_path = Path(args.output)
    error_path  = Path(args.error_output)
    summary_path = Path(args.summary_output)
    save_result_path = Path(args.save_result) if args.save_result else None

    # ---- Print config ----
    print(f"Input (static)     : {input_path}")
    print(f"Simulation file    : {sim_path}")
    print(f"Output (valid)     : {output_path}")
    print(f"Output (errors)    : {error_path}")
    print(f"Output (summary)   : {summary_path}")
    print(f"Output (sim pass)  : {save_result_path if save_result_path else 'disabled'}")
    print(f"Action threshold   : {args.action_threshold:.0%}")
    print(f"NL threshold       : {args.nl_threshold:.0%}")
    print(f"No-sim policy      : {args.no_sim_policy}")
    print()

    # ---- Load files ----
    try:
        with open(input_path, encoding="utf-8") as f:
            static_samples: list[dict] = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"ERROR: Cannot load input file: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(sim_path, encoding="utf-8") as f:
            sim_data: dict = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"ERROR: Cannot load simulation file: {exc}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(static_samples, list):
        print("ERROR: Input must be a JSON list of samples.", file=sys.stderr)
        sys.exit(1)

    # ---- Load tools for action scoring ----
    tools_path = Path(args.tools)
    try:
        tools = load_tools(str(tools_path))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"ERROR: Cannot load tools file: {exc}", file=sys.stderr)
        sys.exit(1)

    # ---- Build task_id → simulation entry map ----
    # If multiple trials exist for the same task_id, keep a list (aggregated later)
    sim_map: dict[str, list[dict]] = {}
    for entry in sim_data.get("simulations", []):
        tid = entry.get("task_id")
        if tid:
            sim_map.setdefault(tid, []).append(entry)

    total = len(static_samples)
    valid_samples:   list[dict] = []
    invalid_samples: list[dict] = []
    passed_sim_task_ids: set[str] = set()

    print(f"Evaluating {total} samples against simulation results...\n")
    print("=" * 72)
    print(f"{'ID':<45} {'Action':>8} {'NL':>7}  {'Result'}")
    print("-" * 72)

    for sample in static_samples:
        if not isinstance(sample, dict):
            continue

        sample_id = sample.get("id", "<unknown>")
        entries   = sim_map.get(sample_id)

        # ---- No simulation found ----
        if not entries:
            if args.no_sim_policy == "pass":
                valid_samples.append(dict(sample))
                print(f"{sample_id:<45} {'N/A':>8} {'N/A':>7}  PASS (no-sim→pass)")
            else:
                reason = "No simulation result found for this task_id"
                error_record = dict(sample)
                error_record["_dynamic_scores"] = {
                    "action_score": None,
                    "nl_score": None,
                    "n_actions": 0,
                    "n_nl": 0,
                }
                error_record["_dynamic_errors"] = [reason]
                invalid_samples.append(error_record)
                print(f"{sample_id:<45} {'N/A':>8} {'N/A':>7}  FAIL ({reason})")
            continue

        # ---- Aggregate across trials (mean) ----
        all_action_scores: list[float] = []
        all_nl_scores:     list[float] = []
        total_n_actions = 0
        total_n_nl      = 0
        per_trial_details: list[dict] = []

        for entry in entries:
            a_score, n_score, n_a, n_n = _compute_sample_scores(entry, tools)
            all_action_scores.append(a_score)
            all_nl_scores.append(n_score)
            total_n_actions += n_a
            total_n_nl      += n_n
            per_trial_details.append({
                "trial":        entry.get("trial", 0),
                "action_score": round(a_score, 4),
                "nl_score":     round(n_score, 4),
                "n_actions":    n_a,
                "n_nl":         n_n,
                "termination_reason": entry.get("termination_reason"),
            })

        action_score = sum(all_action_scores) / len(all_action_scores)
        nl_score     = sum(all_nl_scores)     / len(all_nl_scores)

        # ---- Apply thresholds ----
        fail_reasons: list[str] = []
        if action_score < args.action_threshold:
            fail_reasons.append(
                f"action_score {action_score:.0%} < threshold {args.action_threshold:.0%}"
            )
        if nl_score < args.nl_threshold:
            fail_reasons.append(
                f"nl_score {nl_score:.0%} < threshold {args.nl_threshold:.0%}"
            )

        score_summary = {
            "action_score":   round(action_score, 4),
            "nl_score":       round(nl_score, 4),
            "n_actions":      total_n_actions,
            "n_nl":           total_n_nl,
            "n_trials":       len(entries),
            "per_trial":      per_trial_details,
            "action_threshold": args.action_threshold,
            "nl_threshold":   args.nl_threshold,
        }

        if fail_reasons:
            error_record = dict(sample)
            error_record["_dynamic_scores"] = score_summary
            error_record["_dynamic_errors"] = fail_reasons
            invalid_samples.append(error_record)
            tag = "FAIL"
        else:
            valid_samples.append(dict(sample))
            if isinstance(sample_id, str):
                passed_sim_task_ids.add(sample_id)
            tag = "PASS"

        print(f"{sample_id:<45} {action_score:>7.0%} {nl_score:>6.0%}  {tag}")

    print("=" * 72)

    n_no_sim = sum(
        1 for s in invalid_samples
        if s.get("_dynamic_scores", {}).get("action_score") is None
    )
    n_simulated = len(sim_map)
    n_sim_passed = len(valid_samples) - (n_no_sim if args.no_sim_policy == "pass" else 0)
    n_sim_failed = n_simulated - n_sim_passed

    print(f"\nSimulated samples : {n_simulated}")
    print(f"  Passed          : {n_sim_passed}  ({n_sim_passed/max(n_simulated,1)*100:.1f}%)")
    print(f"  Failed          : {n_sim_failed}  ({n_sim_failed/max(n_simulated,1)*100:.1f}%)")
    if n_no_sim:
        print(f"\nNo simulation     : {n_no_sim}  (policy: {args.no_sim_policy})")
    print(f"\nTotal static input: {total}")
    print(f"Final output      : {len(valid_samples)} valid / {len(invalid_samples)} rejected\n")

    # ---- Write outputs ----
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(valid_samples, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(valid_samples):>3} valid   samples → {output_path}")

    error_path.parent.mkdir(parents=True, exist_ok=True)
    with open(error_path, "w", encoding="utf-8") as f:
        json.dump(invalid_samples, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(invalid_samples):>3} invalid samples → {error_path}")

    passed_summary = _build_passed_subtask_summary(static_samples, valid_samples)

    print("\nPassed sub-task summary by base task_id")
    print("-" * 72)
    print(f"{'Task ID':<56} {'Passed'}")
    print("-" * 72)
    for row in passed_summary:
        print(f"{row['task_id']:<56} {row['passed_subtasks']:>6}")
    print("-" * 72)

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task_id", "passed_subtasks"])
        writer.writeheader()
        writer.writerows(passed_summary)
    print(f"Saved {len(passed_summary):>3} task summaries → {summary_path}")

    if save_result_path is not None:
        filtered_sim_data: dict[str, Any] = dict(sim_data)
        simulations = sim_data.get("simulations")
        if isinstance(simulations, list):
            filtered_sim_data["simulations"] = [
                entry
                for entry in simulations
                if isinstance(entry, dict) and entry.get("task_id") in passed_sim_task_ids
            ]
        else:
            filtered_sim_data["simulations"] = []

        save_result_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_result_path, "w", encoding="utf-8") as f:
            json.dump(filtered_sim_data, f, ensure_ascii=False, indent=2)
        kept = len(filtered_sim_data.get("simulations", []))
        print(f"Saved {kept:>3} passed simulation entries → {save_result_path}")


if __name__ == "__main__":
    main()
