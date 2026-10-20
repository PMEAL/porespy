import math
import operator
import re
import stat
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import imageio.v3 as iio
import numpy as np

from porespy.tools import get_tqdm, settings

tqdm = get_tqdm()


__all__ = [
    "zip_to_stack",
    "folder_to_stack",
]

_IMAGE_SUFFIXES = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".pbm",
    ".pgm",
    ".png",
    ".pnm",
    ".tif",
    ".tiff",
    ".webp",
}


def _is_image(path):
    return path.suffix.lower() in _IMAGE_SUFFIXES and not any(
        part.startswith(".") or part == "__MACOSX" for part in path.parts
    )


def _natural_key(name):
    # Tag the parts so all keys remain comparable, including names starting in digits.
    return tuple(
        (1, int(part)) if part.isdigit() else (0, part.casefold())
        for part in re.split(r"(\d+)", Path(name).name)
    )


def _read_stack(names, open_file, max_bytes, description):
    if max_bytes is not None:
        try:
            if isinstance(max_bytes, (bool, np.bool_)):
                raise TypeError
            max_bytes = operator.index(max_bytes)
        except TypeError as exc:
            raise ValueError("max_bytes must be a non-negative integer or None") from exc
        if max_bytes < 0:
            raise ValueError("max_bytes must be a non-negative integer or None")
    if not names:
        raise ValueError("No supported image slices found")
    names = sorted(names, key=_natural_key)
    keys = [_natural_key(name) for name in names]
    if len(set(keys)) != len(keys):
        raise ValueError("Slice filenames have ambiguous natural ordering")

    shape, dtype = None, None
    # Inspect every header before allocating the stack or decoding image pixels.
    for name in names:
        try:
            with open_file(name) as source:
                suffix = Path(name).suffix.lower()
                # Inspect all TIFF pages/series and all frames of animated formats.
                options = {"page": Ellipsis} if suffix in (".tif", ".tiff") else {}
                props = iio.improps(source, extension=suffix, index=Ellipsis, **options)
        except Exception as exc:
            raise ValueError(f"Could not read slice {name!r}: {exc}") from exc
        current_shape = tuple(props.shape)
        if props.is_batch:
            if props.n_images != 1:
                raise ValueError(f"Slice {name!r} must contain exactly one 2D grayscale image")
            current_shape = current_shape[1:]
        current_dtype = np.dtype(props.dtype)
        if len(current_shape) != 2 or any(size <= 0 for size in current_shape):
            raise ValueError(f"Slice {name!r} must be a non-empty 2D grayscale image")
        if shape is None:
            shape, dtype = current_shape, current_dtype
            required = math.prod(shape) * len(names) * dtype.itemsize
            if max_bytes is not None and required > max_bytes:
                raise ValueError(
                    f"Image stack requires {required} bytes, exceeding max_bytes={max_bytes}"
                )
        elif current_shape != shape or current_dtype != dtype:
            raise ValueError(f"Slice {name!r} must have shape {shape} and dtype {dtype}")

    im = np.empty((*shape, len(names)), dtype=dtype)
    for index, name in enumerate(tqdm(names, desc=description, **settings.tqdm)):
        try:
            with open_file(name) as source:
                layer = iio.imread(source, extension=Path(name).suffix.lower(), index=0)
        except Exception as exc:
            raise ValueError(f"Could not read slice {name!r}: {exc}") from exc
        # Check decoded data too: never broadcast shapes or silently cast different dtypes.
        if layer.shape != shape or layer.dtype != dtype:
            raise ValueError(f"Slice {name!r} must have shape {shape} and dtype {dtype}")
        im[..., index] = layer
    return im


def folder_to_stack(target_dir, *, max_bytes=None):
    r"""
    Read a folder of 2D grayscale images into one 3D array.

    Parameters
    ----------
    target_dir : str or path object
        Folder containing the image slices. Subfolders are not searched.
    max_bytes : int, optional
        Maximum size in bytes of the returned array. The image headers are
        checked before allocating the array or decoding pixels. This does not
        include the memory needed to decode an individual slice. By default,
        no limit is imposed.

    Returns
    -------
    im : ndarray
        Array with shape ``(rows, columns, number_of_slices)`` and the slices'
        original dtype and values.

    Raises
    ------
    ValueError
        If no supported images are found, a slice cannot be read, the slices
        are not 2D or have differing shapes or dtypes, filenames have ambiguous
        ordering, or the array exceeds ``max_bytes``.

    Notes
    -----
    Images are sorted by filename in case-insensitive natural order: ``slice2``
    precedes ``slice10``. Zero-padded names such as ``001.tif`` also work.
    Names equivalent under this ordering, such as ``1.tif`` and ``01.tif``,
    are rejected. Missing sequence numbers are not filled or inferred.

    TIFF, PNG, JPEG, BMP, GIF, PBM, PGM, PNM and WebP extensions are recognized;
    the corresponding imageio decoder must be installed. Color images and
    multi-page images are rejected. Hidden files, subfolders and files with
    other extensions are ignored. Symbolic links to images are rejected.

    See Also
    --------
    zip_to_stack
    """
    folder = Path(target_dir)
    names = []
    for path in folder.iterdir():
        if not _is_image(Path(path.name)):
            continue
        if path.is_symlink():
            raise ValueError(f"Symbolic image links are not supported: {path.name!r}")
        if path.is_file():
            names.append(path.name)
    return _read_stack(
        names,
        lambda name: (folder / name).open("rb"),
        max_bytes,
        "folder_to_stack",
    )


def zip_to_stack(f, *, max_bytes=None):
    r"""
    Read a ZIP of 2D grayscale image slices into one 3D array.

    Parameters
    ----------
    f : str, path object or file-like object
        ZIP archive containing the slices, either at its root or in one folder.
        A file-like object must be seekable and opened in binary mode.
    max_bytes : int, optional
        Maximum size in bytes of the returned array, as in ``folder_to_stack``.
        By default, no limit is imposed.

    Returns
    -------
    im : ndarray
        Array with shape ``(rows, columns, number_of_slices)`` and the slices'
        original dtype and values.

    Raises
    ------
    ValueError
        For the slice and size validation failures listed in
        ``folder_to_stack``, or if image members have unsafe paths, are symbolic
        links, have duplicate names, or occur in more than one folder.

    Notes
    -----
    Slice formats, natural filename ordering and validation are the same as in
    ``folder_to_stack``. Hidden files, ``__MACOSX`` metadata and files with
    unrecognized extensions are ignored. Images must all share one parent
    folder; separate series are never combined automatically.

    Members are read directly from the ZIP without extraction. No directory
    is created, overwritten or deleted on disk, including on failure.

    See Also
    --------
    folder_to_stack
    """
    with ZipFile(f) as archive:
        names = []
        for member in archive.infolist():
            path = PurePosixPath(member.filename)
            if member.is_dir() or path.suffix.lower() not in _IMAGE_SUFFIXES:
                continue
            if (
                path.is_absolute()
                or ".." in path.parts
                or "\\" in member.filename
                or ":" in member.filename
            ):
                raise ValueError(f"Unsafe image member path: {member.filename!r}")
            if not _is_image(path):
                continue
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError(f"Symbolic image links are not supported: {member.filename!r}")
            names.append(member.filename)
        if len(names) != len(set(names)):
            raise ValueError("ZIP contains duplicate image member names")
        if len({PurePosixPath(name).parent for name in names}) > 1:
            raise ValueError("ZIP image slices must all be in one folder")
        return _read_stack(names, archive.open, max_bytes, "zip_to_stack")
