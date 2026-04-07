"""Legal benchmark dataset loading and scoring.

Loads tasks from the nguha/legal-eval HuggingFace dataset.
Supports multiple benchmarks (legalbench, barexam, etc.) via the
'benchmark' column.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

from datasets import load_dataset

from config import RESULTS_DIR

BENCHMARK_NAME = "legal-eval"
DATASET_ID = "nguha/legal-eval"

# Cache the dataset in memory so multiple calls don't re-download
_dataset_cache = None


def _load_all():
    global _dataset_cache
    if _dataset_cache is None:
        _dataset_cache = load_dataset(DATASET_ID, split="train")
    return _dataset_cache


def load_tasks(
    task_filter: list[str] | None = None,
    benchmark_filter: str | None = None,
) -> dict[str, list[dict]]:
    """Load tasks from the legal-eval dataset.

    Args:
        task_filter: If set, only include these task names.
        benchmark_filter: If set, only include tasks from this benchmark
            (e.g., "legalbench", "barexam"). If None, includes all benchmarks.

    Returns dict mapping task_name -> list of sample dicts with keys:
        input, answer, eval_method, sample_index
    """
    ds = _load_all()

    tasks = defaultdict(list)
    for row in ds:
        benchmark = row.get("benchmark", "legalbench")
        if benchmark_filter and benchmark != benchmark_filter:
            continue
        task_name = row["task_name"]
        if task_filter and task_name not in task_filter:
            continue
        idx = len(tasks[task_name])
        tasks[task_name].append({
            "benchmark": benchmark,
            "input": row["input"],
            "answer": row["answer"],
            "eval_method": row["eval_method"],
            "sample_index": idx,
        })

    return dict(tasks)


def get_results_path(model_key: str) -> str:
    results_dir = os.path.join(RESULTS_DIR, BENCHMARK_NAME)
    os.makedirs(results_dir, exist_ok=True)
    return os.path.join(results_dir, f"{model_key}.json")


def load_existing_results(model_key: str) -> dict | None:
    path = get_results_path(model_key)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def save_results(model_key: str, results: dict):
    path = get_results_path(model_key)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


# -- raw generation helpers ---------------------------------------------------


def _raw_dir(test: bool = False) -> str:
    """Return the raw results directory path."""
    if test:
        return os.path.join(RESULTS_DIR, BENCHMARK_NAME, "test")
    return os.path.join(RESULTS_DIR, BENCHMARK_NAME, "raw")


def get_raw_results_path(model_key: str, test: bool = False) -> str:
    raw_dir = _raw_dir(test)
    os.makedirs(raw_dir, exist_ok=True)
    return os.path.join(raw_dir, f"{model_key}.jsonl")


def save_raw_results(model_key: str, rows: list[dict], test: bool = False):
    """Write raw generations as JSONL (one JSON object per line)."""
    path = get_raw_results_path(model_key, test)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def load_raw_results(model_key: str, test: bool = False) -> list[dict] | None:
    """Read raw generations from JSONL. Returns None if file doesn't exist."""
    path = get_raw_results_path(model_key, test)
    if not os.path.exists(path):
        return None
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def list_raw_result_models(test: bool = False) -> list[str]:
    """List model keys that have raw generation files."""
    raw_dir = _raw_dir(test)
    if not os.path.isdir(raw_dir):
        return []
    return [
        f[:-6]  # strip .jsonl
        for f in sorted(os.listdir(raw_dir))
        if f.endswith(".jsonl")
    ]
