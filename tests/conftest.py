import struct
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pytest
import rioxarray
import xarray as xr
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import LocalStore

from virtual_tiff import VirtualTIFF

requires_network = pytest.mark.network

LZW_CLEAR_CODE = 256
LZW_EOI_CODE = 257


def lzw_encode_literals(
    data: bytes, *, with_eoi: bool = True, trailing: bytes = b""
) -> bytes:
    """Encode ``data`` as a TIFF-LZW stream built only from 9-bit literal codes.

    Each byte is emitted as its own literal code, with a ClearCode every 200
    codes so the decoder's dictionary never reaches 511 entries and the code
    width therefore stays at 9 bits for the whole stream. That keeps this helper
    clear of the TIFF "early change" code-width subtleties while still producing
    a stream that any conformant LZW decoder accepts.

    Parameters
    ----------
    with_eoi
        Whether to terminate the stream with the mandatory End-Of-Information
        code. TIFF 6.0 requires it, but writers in the wild sometimes omit it.
    trailing
        Extra bytes appended after the code stream, as emitted by writers that
        pad tile data. Together with a missing EOI these leave enough bits for a
        decoder's size pre-scan to read a phantom code: zero bytes make it
        over-estimate the decoded size, ``0xff`` bytes make it fail outright.
    """
    out = bytearray()
    acc = nbits = 0

    def write(code: int) -> None:
        nonlocal acc, nbits
        acc = (acc << 9) | code
        nbits += 9
        while nbits >= 8:
            nbits -= 8
            out.append((acc >> nbits) & 0xFF)
        acc &= (1 << nbits) - 1

    write(LZW_CLEAR_CODE)
    since_clear = 0
    for byte in data:
        if since_clear >= 200:
            write(LZW_CLEAR_CODE)
            since_clear = 0
        write(byte)
        since_clear += 1
    if with_eoi:
        write(LZW_EOI_CODE)
    if nbits:  # pad the final byte with zero bits
        out.append((acc << (8 - nbits)) & 0xFF)
    return bytes(out) + trailing


_TIFF_SHORT, _TIFF_LONG = 3, 4


def write_lzw_tiff(
    path: Path | str,
    pixels: np.ndarray,
    *,
    with_eoi: bool = True,
    trailing: bytes = b"",
) -> str:
    """Write ``pixels`` as an uncompressed-metadata, LZW-compressed, one-tile TIFF.

    ``pixels`` is a ``(height, width)`` or ``(height, width, samples)`` uint8
    array whose height and width are multiples of 16, so the whole image is a
    single chunky tile. ``with_eoi`` and ``trailing`` are passed through to
    :func:`lzw_encode_literals`, which is how a file with a non-conformant tile
    stream gets built.
    """
    pixels = np.ascontiguousarray(pixels, dtype=np.uint8)
    if pixels.ndim == 2:
        pixels = pixels[:, :, None]
    height, width, samples = pixels.shape
    tile_data = lzw_encode_literals(
        pixels.tobytes(), with_eoi=with_eoi, trailing=trailing
    )

    # Tags must appear in ascending order.
    entries = [
        (256, _TIFF_LONG, [width]),  # ImageWidth
        (257, _TIFF_LONG, [height]),  # ImageLength
        (258, _TIFF_SHORT, [8] * samples),  # BitsPerSample
        (259, _TIFF_SHORT, [5]),  # Compression: LZW
        (262, _TIFF_SHORT, [2 if samples >= 3 else 1]),  # RGB / BlackIsZero
        (277, _TIFF_SHORT, [samples]),  # SamplesPerPixel
        (284, _TIFF_SHORT, [1]),  # PlanarConfiguration: chunky
        (317, _TIFF_SHORT, [1]),  # Predictor: none
        (322, _TIFF_LONG, [width]),  # TileWidth
        (323, _TIFF_LONG, [height]),  # TileLength
        (324, _TIFF_LONG, [0]),  # TileOffsets, filled in below
        (325, _TIFF_LONG, [len(tile_data)]),  # TileByteCounts
        (339, _TIFF_SHORT, [1] * samples),  # SampleFormat: unsigned integer
    ]

    ifd_offset = 8
    data_offset = ifd_offset + 2 + 12 * len(entries) + 4

    # Values wider than the entry's 4-byte value field live after the IFD.
    overflow = bytearray()
    value_fields = {}
    for tag, typ, values in entries:
        fmt = "H" if typ == _TIFF_SHORT else "I"
        packed = struct.pack(f"<{len(values)}{fmt}", *values)
        if len(packed) <= 4:
            value_fields[tag] = packed.ljust(4, b"\x00")
        else:
            value_fields[tag] = struct.pack("<I", data_offset + len(overflow))
            overflow += packed
    tile_offset = data_offset + len(overflow)
    value_fields[324] = struct.pack("<I", tile_offset)

    out = bytearray(struct.pack("<2sHI", b"II", 42, ifd_offset))
    out += struct.pack("<H", len(entries))
    for tag, typ, values in entries:
        out += struct.pack("<HHI", tag, typ, len(values)) + value_fields[tag]
    out += struct.pack("<I", 0)  # no further IFDs
    out += overflow
    out += tile_data

    path = Path(path)
    path.write_bytes(bytes(out))
    return str(path)


# Pytest configuration
def pytest_addoption(parser):
    """Add command-line flags for pytest."""
    parser.addoption(
        "--run-network-tests",
        action="store_true",
        help="runs tests requiring a network connection",
    )


def pytest_runtest_setup(item):
    """Skip network tests unless explicitly enabled."""
    if "network" in item.keywords and not item.config.getoption("--run-network-tests"):
        pytest.skip(
            "set --run-network-tests to run tests requiring an internet connection"
        )


@pytest.fixture
def geotiff_file(tmp_path: Path) -> str:
    """Create a NetCDF4 file with air temperature data."""
    filepath = tmp_path / "air.tif"
    with xr.tutorial.open_dataset("air_temperature") as ds:
        ds.isel(time=0).rio.to_raster(filepath, driver="COG", COMPRESS="DEFLATE")
    return str(filepath)


def resolve_folder(folder: str):
    current_file_path = Path(__file__).resolve()
    repo_root = current_file_path.parent.parent
    return repo_root / folder


def list_tiffs(folder):
    tif_files = list(folder.glob("*.tif"))
    return [file.name for file in tif_files]


def github_examples():
    data_dir = resolve_folder("tests/data/github")
    return list_tiffs(data_dir)


def gdal_examples():
    """Recursively find all .tif files under tests/data/gdal/, returning paths relative to gdal/."""
    data_dir = resolve_folder("tests/data/gdal")
    tif_files = sorted(data_dir.rglob("*.tif"))
    return [str(f.relative_to(data_dir)) for f in tif_files]


def geotiff_test_data_examples():
    """Recursively find all .tif files under tests/data/geotiff-test-data/, returning paths relative to geotiff-test-data/."""
    data_dir = resolve_folder("tests/data/geotiff-test-data")
    tif_files = sorted(data_dir.rglob("*.tif"))
    return [str(f.relative_to(data_dir)) for f in tif_files]


def loadable_dataset(filepath, registry, mask_and_scale=True):
    parser = VirtualTIFF(ifd=0)
    ms = parser(filepath, registry=registry)
    return xr.open_dataset(
        ms,
        engine="zarr",
        consolidated=False,
        zarr_format=3,
        mask_and_scale=mask_and_scale,
    ).load()


def rioxarray_comparison(
    filepath, registry: ObjectStoreRegistry = None, mask_and_scale=True
):
    if not registry:
        registry = ObjectStoreRegistry({filepath: LocalStore()})
    ds = loadable_dataset(filepath, registry, mask_and_scale=mask_and_scale)
    assert isinstance(ds, xr.Dataset)
    expected = rioxarray.open_rasterio(filepath, masked=mask_and_scale)
    filepath = urlparse(filepath).path
    if isinstance(expected, xr.DataArray):
        np.testing.assert_allclose(ds["0"].data.squeeze(), expected.data.squeeze())
    elif isinstance(expected, xr.Dataset):
        expected = expected[filepath.replace("/", "_").lstrip("_")]
        np.testing.assert_allclose(ds["0"].data.squeeze(), expected.data.squeeze())
    elif isinstance(expected, list):
        expected = expected[0][filepath.replace("/", "_").lstrip("_")]
        np.testing.assert_allclose(ds["0"].data.squeeze(), expected.data.squeeze())
    else:
        raise ValueError(
            f"Unexpected type returned from rioxarray.open_rasterio{filepath}"
        )
