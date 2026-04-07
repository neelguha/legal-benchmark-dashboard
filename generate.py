#!/usr/bin/env python3
"""Generate predictions for any model via concurrent direct API calls.

Reads provider from models.yaml and routes to the correct API.
Saves results incrementally so partial progress survives crashes.
Re-running skips already-completed requests.

Usage:
    python generate.py --models claude-haiku-4.5
    python generate.py --models gpt-5-mini grok-4.2-non-reasoning mimo-v2-pro
    python generate.py --models claude-haiku-4.5 --tasks abercrombie hearsay mbe
    python generate.py --models claude-haiku-4.5 --concurrency 20

Then score:
    python run.py score --models claude-haiku-4.5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

from tqdm.asyncio import tqdm as atqdm

from benchmarks.legal_eval import RESULTS_DIR, BENCHMARK_NAME, load_tasks
from config import API_KEYS, DEFAULT_SYSTEM_PROMPT, MODEL_REGISTRY

RAW_DIR = os.path.join(RESULTS_DIR, BENCHMARK_NAME, "raw")


def _get_system_prompt(model_key: str) -> str:
    model_info = MODEL_REGISTRY.get(model_key, {})
    return model_info.get("system_prompt", DEFAULT_SYSTEM_PROMPT)


def _load_existing_raw(model_key: str) -> set[tuple[str, int]]:
    path = os.path.join(RAW_DIR, f"{model_key}.jsonl")
    completed = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    completed.add((row["task_name"], row["sample_index"]))
    return completed


def _remove_raw_for_tasks(model_key: str, task_names: set[str]):
    """Remove existing raw results for specific tasks (rewrite JSONL without them)."""
    path = os.path.join(RAW_DIR, f"{model_key}.jsonl")
    if not os.path.exists(path):
        return
    kept = []
    with open(path) as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                if row["task_name"] not in task_names:
                    kept.append(line)
    with open(path, "w") as f:
        f.writelines(kept)


def _append_raw(model_key: str, rows: list[dict]):
    os.makedirs(RAW_DIR, exist_ok=True)
    path = os.path.join(RAW_DIR, f"{model_key}.jsonl")
    with open(path, "a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _make_caller(provider: str, model_info: dict, api_key: str, system_prompt: str):
    """Return an async function that calls the appropriate API for a single request."""
    model_id = model_info["model_id"]
    max_tokens = model_info.get("max_tokens", 4096)
    effort = model_info.get("effort")
    thinking_level = model_info.get("thinking_level")
    thinking_budget = model_info.get("thinking_budget")
    thinking_type = model_info.get("thinking_type")
    reasoning_enabled = model_info.get("reasoning", False)

    if provider == "anthropic":
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=api_key)

        async def call(input_text: str) -> tuple[str, str]:
            kwargs = {
                "model": model_id,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": input_text}],
            }
            if system_prompt:
                kwargs["system"] = system_prompt
            if thinking_budget is not None or thinking_type:
                t_type = thinking_type or "enabled"
                thinking_params = {"type": t_type}
                if t_type != "adaptive" and thinking_budget is not None:
                    thinking_params["budget_tokens"] = thinking_budget
                kwargs["thinking"] = thinking_params
            if effort:
                kwargs["output_config"] = {"effort": effort}

            resp = await client.messages.create(**kwargs)
            text = "".join(b.text for b in resp.content if hasattr(b, "text"))
            return text, ""

        return call

    elif provider == "openai":
        import openai
        client = openai.AsyncOpenAI(api_key=api_key)

        async def call(input_text: str) -> tuple[str, str]:
            messages = [{"role": "user", "content": input_text}]
            if system_prompt:
                messages.insert(0, {"role": "system", "content": system_prompt})

            api_kwargs = {"model": model_id, "input": messages}
            if effort:
                api_kwargs["reasoning"] = {"effort": effort}

            resp = await client.responses.create(**api_kwargs)
            text = resp.output_text or ""
            reasoning = ""
            for block in (resp.output or []):
                if getattr(block, "type", None) == "reasoning":
                    for summary in getattr(block, "summary", []):
                        reasoning += getattr(summary, "text", "")
            return text, reasoning

        return call

    elif provider == "xai":
        import openai
        client = openai.AsyncOpenAI(api_key=api_key, base_url="https://api.x.ai/v1")

        async def call(input_text: str) -> tuple[str, str]:
            messages = [{"role": "user", "content": input_text}]
            if system_prompt:
                messages.insert(0, {"role": "system", "content": system_prompt})

            resp = await client.chat.completions.create(
                model=model_id,
                messages=messages,
            )
            text = resp.choices[0].message.content or ""
            return text, ""

        return call

    elif provider == "google":
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=api_key)

        async def call(input_text: str) -> tuple[str, str]:
            config_kwargs = {}
            if system_prompt:
                config_kwargs["system_instruction"] = system_prompt
            if thinking_level:
                config_kwargs["thinking_config"] = types.ThinkingConfig(
                    thinking_level=thinking_level
                )
            config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

            resp = await asyncio.to_thread(
                client.models.generate_content,
                model=model_id,
                contents=input_text,
                config=config,
            )
            text = resp.text or ""
            return text, ""

        return call

    elif provider == "openrouter":
        import openai
        client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )

        async def call(input_text: str) -> tuple[str, str]:
            messages = [{"role": "user", "content": input_text}]
            if system_prompt:
                messages.insert(0, {"role": "system", "content": system_prompt})

            kwargs = {"model": model_id, "messages": messages}
            extra_body = {}
            if reasoning_enabled:
                extra_body["reasoning"] = {"effort": "high"}
            if extra_body:
                kwargs["extra_body"] = extra_body

            resp = await client.chat.completions.create(**kwargs)
            text = resp.choices[0].message.content or ""
            reasoning = ""
            msg = resp.choices[0].message
            rc = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
            if rc and isinstance(rc, str):
                reasoning = rc
            return text, reasoning

        return call

    else:
        raise ValueError(f"Unknown provider: {provider}")


async def generate(model_key: str, task_filter: list[str] | None,
                   benchmark_filter: str | None, fresh: bool, concurrency: int):
    model_info = MODEL_REGISTRY[model_key]
    provider = model_info["provider"]
    model_id = model_info["model_id"]

    api_key = API_KEYS.get(provider, "")
    if not api_key:
        print(f"Error: no API key for {provider} (set in .env)")
        sys.exit(1)

    print("Loading tasks...")
    all_tasks = load_tasks(task_filter=task_filter, benchmark_filter=benchmark_filter)
    print(f"Loaded {len(all_tasks)} tasks")

    system_prompt = _get_system_prompt(model_key)
    call = _make_caller(provider, model_info, api_key, system_prompt)

    work_items = []
    for task_name, samples in sorted(all_tasks.items()):
        for s in samples:
            work_items.append({
                "task_name": task_name,
                "sample_index": s["sample_index"],
                "input": s["input"],
            })

    if fresh:
        # Remove existing results for the filtered tasks so they get regenerated
        filtered_tasks = set(all_tasks.keys())
        _remove_raw_for_tasks(model_key, filtered_tasks)
        completed = set()
        print(f"Fresh run: cleared existing results for {len(filtered_tasks)} tasks")
    else:
        completed = _load_existing_raw(model_key)
        if completed:
            before = len(work_items)
            work_items = [
                w for w in work_items
                if (w["task_name"], w["sample_index"]) not in completed
            ]
            print(f"Resuming: {before - len(work_items)} already done, {len(work_items)} remaining")

    if not work_items:
        print("All requests already completed.")
        return

    total = len(work_items)
    print(f"\nGenerating {total} predictions for {model_key} ({model_id}) via {provider}")
    print(f"Concurrency: {concurrency}\n")

    semaphore = asyncio.Semaphore(concurrency)
    pending_results = []
    pending_lock = asyncio.Lock()
    failed = 0
    save_every = 100
    pbar = atqdm(total=total, desc=model_key, unit="req")

    max_retries = 5
    base_delay = 2

    async def _call(i: int, item: dict):
        nonlocal failed
        async with semaphore:
            text = ""
            reasoning = ""
            for attempt in range(max_retries):
                try:
                    text, reasoning = await call(item["input"])
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        pbar.write(f"  RETRY [{item['task_name']}#{item['sample_index']}] attempt {attempt+1}: {e}")
                        await asyncio.sleep(delay)
                    else:
                        failed += 1
                        pbar.write(f"  FAIL [{item['task_name']}#{item['sample_index']}]: {e}")

            row = {
                "task_name": item["task_name"],
                "sample_index": item["sample_index"],
                "response": text.strip(),
            }
            if reasoning:
                row["reasoning"] = reasoning.strip()

            async with pending_lock:
                pending_results.append(row)
                if len(pending_results) >= save_every:
                    _append_raw(model_key, list(pending_results))
                    pending_results.clear()

            pbar.update(1)
            if failed:
                pbar.set_postfix(failed=failed)

    await asyncio.gather(*[_call(i, item) for i, item in enumerate(work_items)])
    pbar.close()

    # Flush remaining
    if pending_results:
        _append_raw(model_key, pending_results)

    total_saved = len(completed) + total
    print(f"\nSaved {total_saved} total responses to {RAW_DIR}/{model_key}.jsonl")
    print(f"Run: python run.py score --models {model_key}")


def main():
    parser = argparse.ArgumentParser(description="Generate predictions for any model")
    parser.add_argument("--models", nargs="+", required=True,
                        help="Model keys from models.yaml (or 'all')")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Optional: specific task names")
    parser.add_argument("--benchmarks", nargs="+", default=None,
                        help="Optional: only tasks from these benchmarks (e.g., barexam legalbench)")
    parser.add_argument("--fresh", action="store_true",
                        help="Regenerate even if results exist (for filtered tasks only)")
    parser.add_argument("--concurrency", type=int, default=10,
                        help="Max concurrent requests (default: 10)")
    args = parser.parse_args()

    if args.models == ["all"]:
        model_keys = list(MODEL_REGISTRY.keys())
    else:
        model_keys = args.models
        for mk in model_keys:
            if mk not in MODEL_REGISTRY:
                print(f"Error: Unknown model '{mk}'. Available: {list(MODEL_REGISTRY.keys())}")
                sys.exit(1)

    # If multiple benchmarks, run each separately
    benchmark_filters = args.benchmarks or [None]
    for mk in model_keys:
        for bf in benchmark_filters:
            asyncio.run(generate(mk, args.tasks, bf, args.fresh, args.concurrency))


if __name__ == "__main__":
    main()
