#!/usr/bin/env python3
"""
Local benchmark for 512-d sentence embeddings (native SDK or OpenAI-shaped HTTP).

Examples:

  # Direct NLEmbedding (no server)
  uv run python scripts/benchmark_embeddings.py

  # With server: apple-fm-cli server --port 8000
  uv run python scripts/benchmark_embeddings.py --mode http --base-url http://127.0.0.1:8000

  uv run python scripts/benchmark_embeddings.py -n 100 -w 5 --batch
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections.abc import Callable

import httpx

SAMPLE_TEXTS: tuple[str, ...] = (
    "Short string.",
    "The quick brown fox jumps over the lazy dog. " * 2,
    "Machine learning on Apple Silicon uses the Neural Engine and on-device models "
    "so prompts never leave the device by default. ",
)


def _percentile_ms(samples_ms: list[float], p: float) -> float:
    if not samples_ms:
        return float("nan")
    s = sorted(samples_ms)
    n = len(s)
    if n == 1:
        return s[0]
    k = (n - 1) * (p / 100.0)
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _stats(samples_ms: list[float]) -> dict[str, float]:
    if not samples_ms:
        return {k: float("nan") for k in ("mean", "stdev", "p50", "p95", "p99", "min", "max")}
    return {
        "mean": statistics.mean(samples_ms),
        "stdev": statistics.stdev(samples_ms) if len(samples_ms) > 1 else 0.0,
        "p50": _percentile_ms(samples_ms, 50.0),
        "p95": _percentile_ms(samples_ms, 95.0),
        "p99": _percentile_ms(samples_ms, 99.0),
        "min": min(samples_ms),
        "max": max(samples_ms),
    }


def _print_table(
    label: str,
    iterations: int,
    total_s: float,
    dim: int,
    samples_ms: list[float],
) -> None:
    st = _stats(samples_ms)
    tput = iterations / total_s if total_s > 0 else 0.0
    print()
    print(label)
    print(f"  runs:         {iterations}")
    print(f"  total time:  {total_s * 1000.0:.1f} ms")
    print(f"  throughput:  {tput:,.1f} calls/s")
    print(f"  dim:         {dim}")
    print("  latency (ms):")
    for key in ("min", "p50", "mean", "p95", "p99", "max", "stdev"):
        print(f"    {key:5s}  {st[key]:8.3f}")


def _bench_single(
    label: str,
    iterations: int,
    warmup: int,
    text_cycle: list[str],
    one: Callable[[str], int],
) -> None:
    for _ in range(warmup):
        _ = one(text_cycle[0])
    times: list[float] = []
    dim = 512
    t0 = time.perf_counter()
    for i in range(iterations):
        t_text = text_cycle[i % len(text_cycle)]
        t1 = time.perf_counter()
        dim = one(t_text)
        times.append((time.perf_counter() - t1) * 1000.0)
    total_s = time.perf_counter() - t0
    _print_table(f"{label} (one string per call)", iterations, total_s, dim, times)


def _bench_batch_http(
    url: str, iterations: int, warmup: int, batch_texts: list[str]
) -> None:
    with httpx.Client(timeout=120.0) as client:
        for _ in range(warmup):
            r = client.post(
                url,
                json={"input": batch_texts, "model": "text-embedding-3-small"},
            )
            r.raise_for_status()
        t0 = time.perf_counter()
        last_dims: list[int] = []
        for _ in range(iterations):
            r = client.post(
                url,
                json={"input": batch_texts, "model": "text-embedding-3-small"},
            )
            r.raise_for_status()
            data = r.json()["data"]
            last_dims = []
            for x in data:
                emb = x["embedding"]
                last_dims.append(len(emb) if isinstance(emb, list) else 0)
        total_s = time.perf_counter() - t0
    n_emb = max(1, iterations * len(batch_texts))
    per_ms = (total_s * 1000.0) / n_emb
    print()
    print(f"Mode: http (batched, {len(batch_texts)} strings per request)")
    print(f"  calls:         {iterations}  (≈{n_emb} embeddings total)")
    print(f"  total time:  {total_s * 1000.0:.1f} ms")
    print(
        f"  throughput:  {n_emb / total_s:,.1f} emb/s  (~{per_ms:.3f} ms per embedding)"
    )
    d0 = last_dims[0] if last_dims else 0
    print(f"  dim:         {d0}")


def _bench_batch_native(
    get_emb: Callable[[str], list[float]], iterations: int, warmup: int, batch_texts: list[str]
) -> None:
    def once() -> list[int]:
        return [len(get_emb(t)) for t in batch_texts]

    for _ in range(warmup):
        _ = once()
    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = once()
    total_s = time.perf_counter() - t0
    n_emb = max(1, iterations * len(batch_texts))
    per_ms = (total_s * 1000.0) / n_emb
    d0 = len(get_emb(batch_texts[0]))
    print()
    print(f"Mode: native (batched, {len(batch_texts)} sequential embeds per round)")
    print(f"  rounds:        {iterations}  (≈{n_emb} embeddings total)")
    print(f"  total time:  {total_s * 1000.0:.1f} ms")
    print(
        f"  throughput:  {n_emb / total_s:,.1f} emb/s  (~{per_ms:.3f} ms per embedding)"
    )
    print(f"  dim:         {d0}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark sentence embeddings (native SDK or HTTP /v1/embeddings).",
    )
    parser.add_argument(
        "--mode",
        choices=["native", "http"],
        default="native",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Server root for --mode http.",
    )
    it_help = "Timed iterations after warmup (default: 50)."
    parser.add_argument("-n", "--iterations", type=int, default=50, help=it_help)
    w_help = "Warmup runs not included in the timed stats (default: 3)."
    parser.add_argument("-w", "--warmup", type=int, default=3, help=w_help)
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Batch mode: several strings per HTTP request; native runs them in a tight loop.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit one JSON object with summary stats (single-call mode only).",
    )
    args = parser.parse_args()

    text_cycle = list(SAMPLE_TEXTS)
    batch_texts: list[str] = list(SAMPLE_TEXTS)[:3]

    if args.mode == "native":
        import apple_fm_sdk as fm  # noqa: PLC0415 — optional heavy import

        if args.json:
            if args.batch:
                print(json.dumps({"error": "use --json without --batch"}))
                return
            get_emb = fm.get_sentence_embedding

            for _ in range(args.warmup):
                _ = get_emb(text_cycle[0])
            times: list[float] = []
            t0 = time.perf_counter()
            dim = 0
            for i in range(args.iterations):
                t1 = time.perf_counter()
                v = get_emb(text_cycle[i % len(text_cycle)])
                times.append((time.perf_counter() - t1) * 1000.0)
                dim = len(v)
            total_s = time.perf_counter() - t0
            st = _stats(times)
            payload = {
                "mode": "native",
                "iterations": args.iterations,
                "total_ms": round(total_s * 1000.0, 3),
                "embeddings_per_s": round(args.iterations / total_s, 2) if total_s else None,
                "dim": dim,
                "latency_ms": {k: round(st[k], 4) for k in st},
            }
            print(json.dumps(payload, indent=2))
            return

        if args.batch:
            _bench_batch_native(
                fm.get_sentence_embedding, args.iterations, args.warmup, batch_texts
            )
        else:
            _bench_single(
                "Mode: native",
                args.iterations,
                args.warmup,
                text_cycle,
                lambda t: len(fm.get_sentence_embedding(t)),
            )
        return

    # HTTP
    base = args.base_url.rstrip("/")
    url = f"{base}/v1/embeddings"
    with httpx.Client(timeout=120.0) as client:
        try:
            r = client.post(url, json={"input": "ping", "model": "text-embedding-3-small"})
            r.raise_for_status()
        except Exception as e:
            raise SystemExit(
                f"HTTP mode: could not POST {url} (start: apple-fm-cli server). {e}"
            ) from e

    if args.json and args.batch:
        print(json.dumps({"error": "use --json without --batch"}))
        return
    if args.json:
        with httpx.Client(timeout=120.0) as cj:
            for _ in range(args.warmup):
                w = cj.post(
                    url, json={"input": text_cycle[0], "model": "text-embedding-3-small"}
                )
                w.raise_for_status()
            times_json: list[float] = []
            t0j = time.perf_counter()
            dim_j = 0
            for i in range(args.iterations):
                t1j = time.perf_counter()
                rj = cj.post(
                    url,
                    json={
                        "input": text_cycle[i % len(text_cycle)],
                        "model": "text-embedding-3-small",
                    },
                )
                rj.raise_for_status()
                times_json.append((time.perf_counter() - t1j) * 1000.0)
                emb0 = rj.json()["data"][0]["embedding"]
                dim_j = len(emb0) if isinstance(emb0, list) else 0
        total_j = time.perf_counter() - t0j
        stj = _stats(times_json)
        out = {
            "mode": "http",
            "url": url,
            "iterations": args.iterations,
            "total_ms": round(total_j * 1000.0, 3),
            "embeddings_per_s": (
                round(args.iterations / total_j, 2) if total_j else None
            ),
            "dim": dim_j,
            "latency_ms": {k: round(stj[k], 4) for k in stj},
        }
        print(json.dumps(out, indent=2))
        return

    if args.batch:
        _bench_batch_http(url, args.iterations, args.warmup, batch_texts)
    else:
        with httpx.Client(timeout=120.0) as client:
            def one_line(t: str) -> int:
                rr = client.post(
                    url, json={"input": t, "model": "text-embedding-3-small"}
                )
                rr.raise_for_status()
                emb = rr.json()["data"][0]["embedding"]
                if isinstance(emb, list):
                    return len(emb)
                return 0

            _bench_single("Mode: http", args.iterations, args.warmup, text_cycle, one_line)


if __name__ == "__main__":
    main()
