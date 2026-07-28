"""Real-file coverage for LZW tiles written without an End-Of-Information code.

GLAD/UMD's annual class maps are 256x256-tiled LZW BigTIFFs in which a handful
of tile streams omit the EOI code that TIFF 6.0 requires. Reading those tiles
used to fail deterministically -- either ``ImcdError IMCD_LZW_CORRUPT`` or
``cannot reshape array of size 65540 into shape (256, 256)`` -- because
imagecodecs was left to infer the decoded size by walking the code stream.

These tests read the affected windows through the full stack and compare them
against GDAL, which reads the same file without complaint.
"""

import math
from urllib.parse import urlparse

import imagecodecs
import numpy as np
import obstore as obs
import pytest
import xarray as xr
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import HTTPStore
from zarr.core.sync import sync

from virtual_tiff import VirtualTIFF
from virtual_tiff.parser import _open_tiff

from .conftest import requires_network

URL = "https://glad.umd.edu/projects/AnnualClassMapsV1/SouthAmerica_Soybean_2020.tif"

# Tiles whose LZW streams carry no EOI code. 322502 makes the size pre-scan
# over-emit by 4 bytes; 123609 makes it raise IMCD_LZW_CORRUPT. Both decode to
# the correct 65536 bytes once the expected size is supplied.
TILES_MISSING_EOI = (322502, 123609)

# The geometry those tile indices were recorded against, asserted against the
# file itself by the `geometry` fixture.
EXPECTED_TILE = 256
EXPECTED_TILES_ACROSS = 751  # ceil(192004 / 256)


def _window(tile_index: int, geometry: tuple[int, int]) -> tuple[int, int]:
    """Top-left pixel of a tile, for the row-major tile order TIFF mandates."""
    tile, tiles_across = geometry
    row, col = divmod(tile_index, tiles_across)
    return row * tile, col * tile


@pytest.fixture(scope="module")
def registry() -> ObjectStoreRegistry:
    parsed = urlparse(URL)
    base = f"{parsed.scheme}://{parsed.netloc}"
    return ObjectStoreRegistry({base: HTTPStore.from_url(base)})


@pytest.fixture(scope="module")
def source(registry: ObjectStoreRegistry):
    """The file's first IFD, plus what is needed to range-read from it."""
    store, path = registry.resolve(URL)
    tiff = sync(_open_tiff(store=store, path=path))
    return store, path, tiff.ifds[0]


@pytest.fixture(scope="module")
def geometry(source) -> tuple[int, int]:
    """Tile size and tile-grid width, read from the file rather than assumed.

    A tile index only identifies a window under the geometry it was recorded
    against, so if the source file is ever republished at a different width or
    tile size these tests must fail rather than quietly compare an unaffected
    window.
    """
    _, _, ifd = source
    assert ifd.tile_width == ifd.tile_height == EXPECTED_TILE
    assert ifd.samples_per_pixel == 1 and tuple(ifd.bits_per_sample) == (8,)
    tiles_across = math.ceil(ifd.image_width / ifd.tile_width)
    assert tiles_across == EXPECTED_TILES_ACROSS
    return ifd.tile_width, tiles_across


@pytest.fixture(scope="module")
def virtual_array(registry: ObjectStoreRegistry) -> xr.DataArray:
    """The whole image as a lazy DataArray; slicing it fetches only the tiles
    that the requested window overlaps."""
    store = VirtualTIFF(ifd=0)(URL, registry=registry)
    ds = xr.open_zarr(
        store,
        zarr_format=3,
        consolidated=False,
        chunks=None,
        mask_and_scale=False,
    )
    return ds["0"]


@pytest.fixture(scope="module")
def gdal_source():
    """The same file opened by GDAL, as an independent LZW implementation."""
    rasterio = pytest.importorskip("rasterio")
    with rasterio.open(URL) as src:
        yield src


@pytest.fixture(scope="module")
def raw_tiles(source) -> dict[int, bytes]:
    """The compressed bytes of each affected tile, fetched by range request."""
    store, path, ifd = source
    offsets, byte_counts = ifd.tile_offsets, ifd.tile_byte_counts
    return {
        tile: bytes(
            obs.get_range(
                store,
                path,
                start=offsets[tile],
                length=byte_counts[tile],
            )
        )
        for tile in TILES_MISSING_EOI
    }


@requires_network
@pytest.mark.parametrize("tile_index", TILES_MISSING_EOI)
def test_source_file_still_exhibits_missing_eoi(raw_tiles, geometry, tile_index):
    """Guard the test below: assert these tiles still lack an EOI code, so that
    a republished (fixed) source file cannot silently turn the check into a
    no-op. Without EOI, an unsized decode either over-emits or raises; with the
    expected size it always returns exactly one tile of pixels."""
    raw = raw_tiles[tile_index]
    tile, _ = geometry
    expected_nbytes = tile * tile

    try:
        unsized = len(imagecodecs.lzw_decode(raw))
    except imagecodecs.LzwError:
        pass  # the size pre-scan walked into an undecodable code
    else:
        assert unsized > expected_nbytes, (
            f"tile {tile_index} no longer over-emits ({unsized} bytes); the "
            "source file may have been rewritten with EOI codes"
        )

    sized = imagecodecs.lzw_decode(raw, out=np.zeros(expected_nbytes, dtype=np.uint8))
    assert len(memoryview(sized).cast("B")) == expected_nbytes


@requires_network
@pytest.mark.parametrize("tile_index", TILES_MISSING_EOI)
def test_lzw_tile_without_eoi_matches_gdal(
    virtual_array, gdal_source, geometry, tile_index
):
    """Reading an EOI-less tile through the full parser -> zarr -> xarray stack
    returns the same pixels GDAL reads from the same window."""
    from rasterio.windows import Window

    tile, _ = geometry
    y0, x0 = _window(tile_index, geometry)
    actual = np.asarray(virtual_array[y0 : y0 + tile, x0 : x0 + tile])
    expected = gdal_source.read(1, window=Window(x0, y0, tile, tile))

    np.testing.assert_array_equal(actual, expected)
