#!/usr/bin/env python3
"""Benchmark sync ``VirtualTIFF.__call__`` vs async ``VirtualTIFF.aopen``.

Times N concurrent parses of the same TIFF under two patterns:

* ``sync via to_thread``  — ``asyncio.gather(*[asyncio.to_thread(parser, url, reg) ...])``
  This is what a downstream async caller has to do today with the sync API.
  Each thread ends up calling ``zarr.core.sync.sync`` which dispatches every
  coroutine onto the single shared ``zarr_io`` event loop, serialising them.
* ``async aopen``         — ``asyncio.gather(*[parser.aopen(url, reg) ...])``
  Awaits ``async_tiff.TIFF.open`` on the caller's loop directly. Real
  concurrency.

The default target is the AEF tile from ``tests/test_multiband_s3.py`` —
a 64-band PlanarConfiguration=2 TIFF on the public source.coop bucket.
Anonymous read; no AWS credentials required. Local-file mode is also
supported but the zarr_io bottleneck only surfaces on workloads where
each ``_open_tiff`` actually awaits real I/O (i.e. network).

Usage:
    pixi run -e test python scripts/bench_async_parse.py            # S3, default
    pixi run -e test python scripts/bench_async_parse.py --n 64
    pixi run -e test python scripts/bench_async_parse.py --local tests/data/github/test_reference.tif
"""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import LocalStore, S3Store

from virtual_tiff import VirtualTIFF

REPO_ROOT = Path(__file__).resolve().parent.parent

AEF_BUCKET = "us-west-2.opendata.source.coop"
AEF_PREFIX = f"s3://{AEF_BUCKET}/"
AEF_TILE = (
    "s3://us-west-2.opendata.source.coop/"
    "tge-labs/aef/v1/annual/2023/10N/"
    "xjtqldak16clgy5os-0000000000-0000008192.tiff"
)


def aef_registry() -> tuple[str, ObjectStoreRegistry]:
    store = S3Store(bucket=AEF_BUCKET, skip_signature=True, region="us-west-2")
    return AEF_TILE, ObjectStoreRegistry({AEF_PREFIX: store})


def local_registry(tiff: Path) -> tuple[str, ObjectStoreRegistry]:
    return f"file://{tiff.resolve()}", ObjectStoreRegistry({"file://": LocalStore()})


async def bench_sync_via_to_thread(url: str, registry: ObjectStoreRegistry, n: int) -> float:
    parser = VirtualTIFF()
    start = time.perf_counter()
    await asyncio.gather(
        *(asyncio.to_thread(parser, url, registry) for _ in range(n))
    )
    return time.perf_counter() - start


async def bench_sync_naive(url: str, registry: ObjectStoreRegistry, n: int) -> float:
    """Naive pattern: call sync ``__call__`` from inside a coroutine without
    ``to_thread``. Each call blocks the running loop, so ``gather`` is serial.
    This is the worst-case pattern downstream callers fall into when they
    forget to thread the sync API.
    """
    parser = VirtualTIFF()

    async def one() -> None:
        parser(url, registry)

    start = time.perf_counter()
    await asyncio.gather(*(one() for _ in range(n)))
    return time.perf_counter() - start


async def bench_aopen(url: str, registry: ObjectStoreRegistry, n: int) -> float:
    parser = VirtualTIFF()
    start = time.perf_counter()
    await asyncio.gather(*(parser.aopen(url, registry) for _ in range(n)))
    return time.perf_counter() - start


async def time_one_aopen(url: str, registry: ObjectStoreRegistry) -> float:
    parser = VirtualTIFF()
    start = time.perf_counter()
    await parser.aopen(url, registry)
    return time.perf_counter() - start


async def main_async(
    url: str,
    registry: ObjectStoreRegistry,
    label: str,
    n: int,
    repeats: int,
) -> None:
    # Warm-up: prime any one-time imports / caches so they don't bias results.
    await VirtualTIFF().aopen(url, registry)

    single = await time_one_aopen(url, registry)
    print(f"single aopen: {single*1000:.1f} ms")
    print(f"N={n}, repeats={repeats}, target={label}\n")

    print(f"{'pattern':<24} {'wall':>10} {'per-call':>12} {'speedup':>10}")
    print("-" * 60)

    naive_walls: list[float] = []
    for _ in range(repeats):
        naive_walls.append(await bench_sync_naive(url, registry, n))
    naive_best = min(naive_walls)
    print(
        f"{'sync naive (gather)':<24} {naive_best:>9.3f}s "
        f"{(naive_best/n)*1000:>10.1f} ms {'1.00x':>10}"
    )

    thread_walls: list[float] = []
    for _ in range(repeats):
        thread_walls.append(await bench_sync_via_to_thread(url, registry, n))
    thread_best = min(thread_walls)
    print(
        f"{'sync via to_thread':<24} {thread_best:>9.3f}s "
        f"{(thread_best/n)*1000:>10.1f} ms "
        f"{(naive_best/thread_best):>9.2f}x"
    )

    aopen_walls: list[float] = []
    for _ in range(repeats):
        aopen_walls.append(await bench_aopen(url, registry, n))
    aopen_best = min(aopen_walls)
    print(
        f"{'async aopen':<24} {aopen_best:>9.3f}s "
        f"{(aopen_best/n)*1000:>10.1f} ms "
        f"{(naive_best/aopen_best):>9.2f}x"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--local",
        type=Path,
        default=None,
        help="Path to a local TIFF. If omitted, hits the AEF tile on S3.",
    )
    p.add_argument("--n", type=int, default=32, help="Concurrent parses per run.")
    p.add_argument("--repeats", type=int, default=3, help="Repeats; best wall reported.")
    args = p.parse_args()

    if args.local is not None:
        if not args.local.exists():
            raise SystemExit(
                f"TIFF not found: {args.local}. Run "
                f"`pixi run -e test download-test-images` first."
            )
        url, registry = local_registry(args.local)
        label = f"local file {args.local.name} ({args.local.stat().st_size/1e6:.1f} MB)"
    else:
        url, registry = aef_registry()
        label = f"S3 {AEF_TILE}"

    asyncio.run(main_async(url, registry, label, args.n, args.repeats))


if __name__ == "__main__":
    main()
