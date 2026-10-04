#!/usr/bin/env python3
"""Run a repeatable OpenAI-compatible SGLang throughput benchmark."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import statistics
import time
import urllib.error
import urllib.request


def percentile(values: list[float], percent: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    index = (len(ordered) - 1) * percent
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def make_prompt(target_words: int, request_id: int) -> str:
    seed = (
        "Profile this inference request carefully and continue with concise "
        "technical prose about efficient language model serving. "
    )
    words = (seed * ((target_words // len(seed.split())) + 2)).split()
    return f"Request {request_id}. " + " ".join(words[:target_words])


def send_request(url: str, api_key: str, prompt: str, output_tokens: int) -> dict:
    body = json.dumps(
        {
            "model": "qwen3-0.6b",
            "prompt": prompt,
            "max_tokens": output_tokens,
            "temperature": 0,
            "ignore_eos": True,
            "stream": False,
        }
    ).encode()
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {detail[:500]}") from error
    latency = time.perf_counter() - started
    usage = payload.get("usage", {})
    return {
        "latency": latency,
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "output_tokens": int(usage.get("completion_tokens", 0)),
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:30000/v1/completions")
    parser.add_argument("--requests", type=int, default=12)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--input-words", type=int, default=192)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--label", default="profile")
    args = parser.parse_args()

    api_key = os.environ.get("SGLANG_API_KEY", "")
    if not api_key:
        raise SystemExit("SGLANG_API_KEY is not set")

    for index in range(args.warmup):
        await asyncio.to_thread(
            send_request,
            args.url,
            api_key,
            make_prompt(args.input_words, -(index + 1)),
            min(args.output_tokens, 32),
        )

    semaphore = asyncio.Semaphore(args.concurrency)

    async def bounded_request(index: int) -> dict:
        async with semaphore:
            return await asyncio.to_thread(
                send_request,
                args.url,
                api_key,
                make_prompt(args.input_words, index),
                args.output_tokens,
            )

    started = time.perf_counter()
    results = await asyncio.gather(
        *(bounded_request(index) for index in range(args.requests))
    )
    wall_time = time.perf_counter() - started

    latencies = [result["latency"] for result in results]
    prompt_tokens = sum(result["prompt_tokens"] for result in results)
    output_tokens = sum(result["output_tokens"] for result in results)
    summary = {
        "label": args.label,
        "requests": args.requests,
        "concurrency": args.concurrency,
        "wall_seconds": round(wall_time, 3),
        "request_throughput_rps": round(args.requests / wall_time, 3),
        "output_throughput_tokens_per_second": round(output_tokens / wall_time, 3),
        "total_throughput_tokens_per_second": round(
            (prompt_tokens + output_tokens) / wall_time, 3
        ),
        "latency_mean_seconds": round(statistics.mean(latencies), 3),
        "latency_p50_seconds": round(percentile(latencies, 0.50), 3),
        "latency_p95_seconds": round(percentile(latencies, 0.95), 3),
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(main())
