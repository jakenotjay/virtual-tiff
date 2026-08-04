import struct
from pathlib import Path
from urllib.parse import urlparse

import imagecodecs
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
    """Encode ``data`` as a TIFF-LZW stream of 9-bit literal codes."""
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


def lzw_unsized_decode_fails(stream: bytes, nbytes: int) -> bool:
    """Whether imagecodecs mis-decodes ``stream`` when not told the output size."""
    try:
        return len(imagecodecs.lzw_decode(stream)) != nbytes
    except imagecodecs.LzwError:
        return True


_TIFF_SHORT, _TIFF_LONG = 3, 4


def write_lzw_tiff(
    path: Path | str,
    pixels: np.ndarray,
    *,
    tile: tuple[int, int] | None = None,
    rows_per_strip: int | None = None,
    planar_configuration: int = 1,
    with_eoi: bool = True,
    trailing: bytes = b"",
) -> str:
    """Write uint8 ``pixels``, ``(h, w)`` or ``(h, w, samples)``, as an
    LZW-compressed TIFF and return its path.
    """
    # BitsPerSample and SampleFormat below are hardcoded 8-bit unsigned, so refuse
    # anything else rather than wrapping the values mod 256 on the way in.
    if np.asarray(pixels).dtype != np.uint8:
        raise ValueError(f"pixels must be uint8, got {np.asarray(pixels).dtype}")
    pixels = np.ascontiguousarray(pixels)
    if pixels.ndim == 2:
        pixels = pixels[:, :, None]
    height, width, samples = pixels.shape
    if tile is not None and rows_per_strip is not None:
        raise ValueError("pass either tile or rows_per_strip, not both")
    if planar_configuration not in (1, 2):
        raise ValueError(
            f"PlanarConfiguration must be 1 or 2, got {planar_configuration}"
        )

    if rows_per_strip is not None:
        block_height, block_width = min(rows_per_strip, height), width
    else:
        block_height, block_width = tile or (height, width)
        if block_height % 16 or block_width % 16:
            raise ValueError(
                f"TIFF tile dimensions must be multiples of 16, got "
                f"{(block_height, block_width)}"
            )
    if height % block_height or width % block_width:
        raise ValueError(
            f"image {(height, width)} is not an exact number of "
            f"{(block_height, block_width)} blocks"
        )

    # Chunky files hold one block per grid position, each interleaving all samples;
    # planar files hold one block per sample per grid position, ordered by sample.
    planes = (
        [pixels] if planar_configuration == 1 else np.split(pixels, samples, axis=2)
    )
    blocks = [
        lzw_encode_literals(
            np.ascontiguousarray(
                plane[y : y + block_height, x : x + block_width]
            ).tobytes(),
            with_eoi=with_eoi,
            trailing=trailing,
        )
        for plane in planes
        for y in range(0, height, block_height)
        for x in range(0, width, block_width)
    ]
    byte_counts = [len(data) for data in blocks]

    if rows_per_strip is not None:
        offsets_tag = 273  # StripOffsets
        layout = [
            (278, _TIFF_LONG, [rows_per_strip]),  # RowsPerStrip
            (279, _TIFF_LONG, byte_counts),  # StripByteCounts
        ]
    else:
        offsets_tag = 324  # TileOffsets
        layout = [
            (322, _TIFF_LONG, [block_width]),  # TileWidth
            (323, _TIFF_LONG, [block_height]),  # TileLength
            (325, _TIFF_LONG, byte_counts),  # TileByteCounts
        ]

    entries = sorted(  # a TIFF IFD's entries must be in ascending tag order
        [
            (256, _TIFF_LONG, [width]),  # ImageWidth
            (257, _TIFF_LONG, [height]),  # ImageLength
            (258, _TIFF_SHORT, [8] * samples),  # BitsPerSample
            (259, _TIFF_SHORT, [5]),  # Compression: LZW
            (262, _TIFF_SHORT, [2 if samples >= 3 else 1]),  # RGB / BlackIsZero
            (277, _TIFF_SHORT, [samples]),  # SamplesPerPixel
            (284, _TIFF_SHORT, [planar_configuration]),  # PlanarConfiguration
            (317, _TIFF_SHORT, [1]),  # Predictor: none
            (offsets_tag, _TIFF_LONG, [0] * len(blocks)),  # resolved below
            (339, _TIFF_SHORT, [1] * samples),  # SampleFormat: unsigned integer
            *layout,
        ]
    )

    def pack(typ: int, values: list[int]) -> bytes:
        return struct.pack(
            f"<{len(values)}{'H' if typ == _TIFF_SHORT else 'I'}", *values
        )

    ifd_offset = 8
    data_offset = ifd_offset + 2 + 12 * len(entries) + 4

    # Values wider than an entry's 4-byte value field live after the IFD, and have
    # to be laid out before the pixel data because the block offsets depend on
    # where that starts. Both value types written here are even width, so the value
    # blocks stay word aligned without padding. Their sizes depend only on each
    # entry's type and count, so the layout is identical in both passes.
    overflow_positions: dict[int, int] = {}
    overflow_size = 0
    for tag, typ, values in entries:
        packed = pack(typ, values)
        if len(packed) > 4:
            overflow_positions[tag] = overflow_size
            overflow_size += len(packed)

    block_offsets = []
    next_offset = data_offset + overflow_size
    for count in byte_counts:
        block_offsets.append(next_offset)
        next_offset += count
    entries = [
        (tag, typ, block_offsets if tag == offsets_tag else values)
        for tag, typ, values in entries
    ]

    overflow = bytearray(overflow_size)
    value_fields = {}
    for tag, typ, values in entries:
        packed = pack(typ, values)
        if len(packed) > 4:
            position = overflow_positions[tag]
            overflow[position : position + len(packed)] = packed
            value_fields[tag] = struct.pack("<I", data_offset + position)
        else:
            value_fields[tag] = packed.ljust(4, b"\x00")

    out = bytearray(struct.pack("<2sHI", b"II", 42, ifd_offset))
    out += struct.pack("<H", len(entries))
    for tag, typ, values in entries:
        out += struct.pack("<HHI", tag, typ, len(values)) + value_fields[tag]
    out += struct.pack("<I", 0)  # no further IFDs
    out += overflow
    for data in blocks:
        out += data

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
