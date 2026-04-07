"""Score raw generations and export the leaderboard.

Generation is handled by generate.py. This script reads the raw JSONL files
produced by generate.py, scores them, and builds the leaderboard.

Usage:
    python run.py score                          # score all models with raw files
    python run.py score --models claude-haiku-4.5 gpt-5-mini
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections import defaultdict
from datetime import datetime, timezone

from benchmarks.legal_eval import (
    list_raw_result_models,
    load_existing_results,
    load_raw_results,
    load_tasks,
    save_results,
)
from config import MODEL_REGISTRY
from export import export
from scoring import compute_accuracy, extract_answer


# -- helpers ------------------------------------------------------------------


def _score_task(samples: list[dict], responses: list[str], all_samples: list[dict] | None = None) -> dict:
    """Score responses for a single task. Returns accuracy dict.

    If all_samples is provided, derive valid labels from the full task data
    (important when only a subset is sampled).
    """
    eval_method = samples[0]["eval_method"]
    label_source = all_samples if all_samples else samples
    labels = set(s["answer"] for s in label_source)
    valid_labels = sorted(labels) if len(labels) <= 20 else None
    predictions = [extract_answer(r, valid_labels) for r in responses]
    expected = [s["answer"] for s in samples]
    acc = compute_accuracy(predictions, expected, eval_method)
    acc["predictions"] = predictions
    acc["expected"] = expected
    acc["responses"] = responses
    return acc


# -- score --------------------------------------------------------------------


async def cmd_score(args):
    """Score raw generations and export the leaderboard."""
    if args.models:
        model_keys = args.models
    else:
        model_keys = list_raw_result_models()
        if not model_keys:
            print("No raw generation files found. Run 'python generate.py --models <model>' first.")
            return

    print("Loading tasks...")
    all_tasks = load_tasks()
    print(f"Loaded {len(all_tasks)} tasks\n")

    scored_any = False

    for model_key in model_keys:
        raw_rows = load_raw_results(model_key)
        if raw_rows is None:
            print(f"  {model_key}: no raw generations found, skipping")
            continue

        model_info = MODEL_REGISTRY.get(model_key, {})

        # Group raw rows by task
        task_responses: dict[str, list[tuple[int, str]]] = defaultdict(list)
        for row in raw_rows:
            task_responses[row["task_name"]].append(
                (row["sample_index"], row["response"])
            )

        # Score each task
        results = load_existing_results(model_key) or {
            "model_key": model_key,
            "display_name": model_info.get("display_name", model_key),
            "provider": model_info.get("provider", ""),
            "developer": model_info.get("developer", model_info.get("provider", "")),
            "open_weight": model_info.get("open_weight", False),
            "tasks": {},
        }

        task_names = sorted(task_responses.keys())
        warnings = []
        from tqdm import tqdm as tqdm_sync
        for task_name in tqdm_sync(task_names, desc=f"  {model_key}", unit="task", leave=False):
            if task_name not in all_tasks:
                warnings.append(f"{task_name}: task not found in benchmark")
                continue

            # Match responses to samples by sample_index
            resp_by_idx = {idx: text for idx, text in task_responses[task_name]}
            samples = all_tasks[task_name]

            matched_samples = []
            matched_responses = []
            for s in samples:
                si = s["sample_index"]
                if si in resp_by_idx:
                    matched_samples.append(s)
                    matched_responses.append(resp_by_idx[si])

            if len(matched_samples) < len(samples):
                warnings.append(f"{task_name}: {len(samples) - len(matched_samples)} missing responses")

            if not matched_samples:
                continue

            acc = _score_task(matched_samples, matched_responses, all_samples=samples)

            # Get benchmark name from the samples
            benchmark = matched_samples[0].get("benchmark", "legalbench")

            results["tasks"][task_name] = {
                "benchmark": benchmark,
                "accuracy": acc["accuracy"],
                "correct": acc["correct"],
                "total": acc["total"],
                "parse_failures": acc.get("parse_failures", 0),
                "predictions": [
                    {
                        "sample_index": matched_samples[i]["sample_index"],
                        "expected": acc["expected"][i],
                        "predicted": acc["predictions"][i],
                        "raw_response": acc["responses"][i][:500],
                    }
                    for i in range(len(matched_samples))
                ],
            }

        # Compute per-benchmark accuracy
        bench_task_accs: dict[str, list[float]] = defaultdict(list)
        for t_name, t_data in results["tasks"].items():
            bench = t_data.get("benchmark", "legalbench")
            bench_task_accs[bench].append(t_data["accuracy"])

        results["benchmarks"] = {}
        for bench, accs in sorted(bench_task_accs.items()):
            results["benchmarks"][bench] = {
                "overall_accuracy": sum(accs) / len(accs) if accs else 0.0,
                "num_tasks": len(accs),
            }

        # Overall accuracy = mean of benchmark accuracies
        bench_accs = [b["overall_accuracy"] for b in results["benchmarks"].values()]
        results["overall_accuracy"] = sum(bench_accs) / len(bench_accs) if bench_accs else 0.0

        total_pf = sum(t.get("parse_failures", 0) for t in results["tasks"].values())
        results["evaluated_at"] = datetime.now(timezone.utc).isoformat()

        # Count missing
        scored_tasks = set(task_responses.keys()) & set(all_tasks.keys())
        missing_tasks = set(all_tasks.keys()) - scored_tasks
        total_missing_samples = sum(
            len(all_tasks[t]) - len([1 for idx, _ in task_responses.get(t, [])
                                     if idx in {s["sample_index"] for s in all_tasks[t]}])
            for t in scored_tasks
        )

        save_results(model_key, results)
        scored_any = True

        # Print summary
        summary_parts = []
        if total_pf:
            summary_parts.append(f"{total_pf} parse failures")
        if missing_tasks:
            summary_parts.append(f"{len(missing_tasks)} tasks missing")
        if total_missing_samples:
            summary_parts.append(f"{total_missing_samples} samples missing")
        suffix = f"  ({', '.join(summary_parts)})" if summary_parts else ""
        print(f"  {model_key}: {results['overall_accuracy']:.1%} "
              f"({len(results['tasks'])}/{len(all_tasks)} tasks){suffix}")
        for w in warnings:
            print(f"    WARNING: {w}")

    if scored_any:
        export()


# -- CLI ----------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Score generated predictions and export the leaderboard")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # score
    score_parser = subparsers.add_parser("score", help="Score raw generations and export leaderboard")
    score_parser.add_argument(
        "--models", nargs="+", default=None,
        help="Model keys to score (default: all available raw files)",
    )

    args = parser.parse_args()

    if args.command == "score":
        asyncio.run(cmd_score(args))


if __name__ == "__main__":
    main()
