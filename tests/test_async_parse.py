import asyncio
import time

import numpy as np
import pytest
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import LocalStore

from virtual_tiff import VirtualTIFF


@pytest.fixture
def registry() -> ObjectStoreRegistry:
    return ObjectStoreRegistry({"file://": LocalStore()})


def _manifest_arrays(ms):
    return ms._group.arrays


def test_aopen_matches_call(geotiff_file, registry):
    url = f"file://{geotiff_file}"

    sync_ms = VirtualTIFF()(url, registry)
    async_ms = asyncio.run(VirtualTIFF().aopen(url, registry))

    sync_arrays = _manifest_arrays(sync_ms)
    async_arrays = _manifest_arrays(async_ms)
    assert sync_arrays.keys() == async_arrays.keys()

    for key, sync_arr in sync_arrays.items():
        async_arr = async_arrays[key]
        sync_manifest = sync_arr.manifest
        async_manifest = async_arr.manifest
        np.testing.assert_array_equal(sync_manifest._paths, async_manifest._paths)
        np.testing.assert_array_equal(
            sync_manifest._offsets, async_manifest._offsets
        )
        np.testing.assert_array_equal(
            sync_manifest._lengths, async_manifest._lengths
        )
        assert sync_arr.metadata.to_dict() == async_arr.metadata.to_dict()


def test_aopen_concurrent_scaling(geotiff_file, registry):
    url = f"file://{geotiff_file}"
    n = 32

    # Warm any one-time setup so it doesn't bias the single-open baseline.
    asyncio.run(VirtualTIFF().aopen(url, registry))

    async def time_single():
        start = time.perf_counter()
        await VirtualTIFF().aopen(url, registry)
        return time.perf_counter() - start

    single_wall = asyncio.run(time_single())

    async def gather_n():
        start = time.perf_counter()
        await asyncio.gather(
            *(VirtualTIFF().aopen(url, registry) for _ in range(n))
        )
        return time.perf_counter() - start

    gather_wall = asyncio.run(gather_n())

    # If aopen serialised on a shared loop, gather_wall would be ~ n * single.
    # Real concurrency should bring it well under that. 4x ceiling is generous
    # for a local fixture where per-call wall is dominated by fixed CPU work.
    assert gather_wall < max(4 * single_wall, 0.5), (
        f"gather_wall={gather_wall:.3f}s vs single_wall={single_wall:.3f}s "
        f"with n={n}"
    )
