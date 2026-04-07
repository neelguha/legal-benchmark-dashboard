"""Export evaluation results to leaderboard.json for the dashboard.

Reads per-model result JSONs from results/legal-eval/ and builds a
leaderboard with per-benchmark breakdowns (legalbench, barexam, etc.).

Usage:
    python export.py
"""

import glob
import json
import os
import shutil
from datetime import datetime, timezone

from config import DASHBOARD_DATA_DIR, RESULTS_DIR
from benchmarks.legal_eval import BENCHMARK_NAME

# Pretty display names for benchmarks (override .title() which gives "Legalbench")
BENCHMARK_DISPLAY_NAMES = {
    "legalbench": "LegalBench",
    "barexam": "BarExam",
    "lexam": "LEXam",
    "housingqa": "HousingQA",
    "legal_hallucinations": "Hallucinations",
}


def build_leaderboard() -> dict:
    """Build the full leaderboard from model result files."""
    results_dir = os.path.join(RESULTS_DIR, BENCHMARK_NAME)
    pattern = os.path.join(results_dir, "*.json")

    all_benchmarks = {}  # benchmark_name -> {display_name, num_tasks}
    all_models = []

    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            data = json.load(f)

        model_key = data["model_key"]

        # Build per-benchmark task summaries from the tasks dict
        benchmarks_for_model = {}
        for task_name, task_data in data.get("tasks", {}).items():
            bench = task_data.get("benchmark", "legalbench")

            if bench not in benchmarks_for_model:
                benchmarks_for_model[bench] = {"per_task": {}, "accuracies": []}

            benchmarks_for_model[bench]["per_task"][task_name] = {
                "accuracy": task_data["accuracy"],
                "correct": task_data["correct"],
                "total": task_data["total"],
            }
            benchmarks_for_model[bench]["accuracies"].append(task_data["accuracy"])

        # Finalize per-benchmark stats
        model_benchmarks = {}
        for bench, bdata in benchmarks_for_model.items():
            accs = bdata["accuracies"]
            bench_acc = sum(accs) / len(accs) if accs else 0.0
            num_tasks = len(bdata["per_task"])

            model_benchmarks[bench] = {
                "overall_accuracy": bench_acc,
                "num_tasks": num_tasks,
                "per_task": bdata["per_task"],
            }

            # Track global benchmark metadata (use max num_tasks across models)
            if bench not in all_benchmarks:
                all_benchmarks[bench] = {
                    "display_name": BENCHMARK_DISPLAY_NAMES.get(bench, bench.replace("_", " ").title()),
                    "num_tasks": num_tasks,
                }
            else:
                all_benchmarks[bench]["num_tasks"] = max(
                    all_benchmarks[bench]["num_tasks"], num_tasks
                )

        # Overall = mean of benchmark accuracies
        bench_accs = [b["overall_accuracy"] for b in model_benchmarks.values()]
        overall = sum(bench_accs) / len(bench_accs) if bench_accs else 0.0

        all_models.append({
            "model_key": model_key,
            "display_name": data.get("display_name", model_key),
            "provider": data.get("provider", ""),
            "developer": data.get("developer", data.get("provider", "")),
            "open_weight": data.get("open_weight", False),
            "evaluated_at": data.get("evaluated_at", ""),
            "overall_accuracy": overall,
            "benchmarks": model_benchmarks,
        })

    # Sort by overall accuracy descending
    all_models.sort(key=lambda m: m["overall_accuracy"], reverse=True)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmarks": all_benchmarks,
        "models": all_models,
    }


def export():
    """Build leaderboard and write to results/ and dashboard/data/."""
    leaderboard = build_leaderboard()

    # Write to results/
    results_path = os.path.join(RESULTS_DIR, "leaderboard.json")
    with open(results_path, "w") as f:
        json.dump(leaderboard, f, indent=2)
    print(f"Wrote {results_path}")

    # Copy to dashboard/data/
    os.makedirs(DASHBOARD_DATA_DIR, exist_ok=True)
    dashboard_path = os.path.join(DASHBOARD_DATA_DIR, "leaderboard.json")
    shutil.copy2(results_path, dashboard_path)
    print(f"Copied to {dashboard_path}")

    # Summary
    n_models = len(leaderboard["models"])
    benchmarks = leaderboard["benchmarks"]
    print(f"\nLeaderboard: {n_models} models, {len(benchmarks)} benchmark(s)")
    for bench_name, bench_info in sorted(benchmarks.items()):
        print(f"  {bench_name}: {bench_info['num_tasks']} tasks")

    print()
    for m in leaderboard["models"]:
        parts = [f"{m['display_name']:30s} overall={m['overall_accuracy']:.3f}"]
        for bench_name in sorted(benchmarks.keys()):
            b = m["benchmarks"].get(bench_name)
            if b:
                parts.append(f"{bench_name}={b['overall_accuracy']:.3f}")
        print("  " + "  ".join(parts))


if __name__ == "__main__":
    export()
