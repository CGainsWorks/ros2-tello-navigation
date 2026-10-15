"""Generating the QR bitmaps the simulator prints on its walls.

The symbols are produced by OpenCV's own ``QRCodeEncoder`` and read back by
OpenCV's ``QRCodeDetector``. That matters more than it might look: the simulated
detection path is not a shortcut that hands the estimator a marker identity and
four exact corners. A real QR symbol is drawn into the world at its surveyed
position, rendered through the camera model with the perspective, blur and noise
of a real frame and then found by the same detector that would run on hardware.
When detection fails in simulation, it fails for the same reasons it would fail
in the room: too far, too oblique, too blurred, too dark.

The one thing simulation cannot reproduce is print quality. A symbol that comes
out of a laser printer with slightly bled edges, or is taped to a wall with a
crease through the timing pattern, will decode worse than anything here. See
``docs/HARDWARE.md``.
"""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np


class QrEncodeError(RuntimeError):
    """Raised when a payload cannot be encoded as a QR symbol."""


@lru_cache(maxsize=256)
def _encode(payload: str) -> np.ndarray:
    """The bare symbol, with OpenCV's own quiet zone stripped off.

    ``cv2.QRCodeEncoder`` returns the symbol wrapped in a two-module white
    border, so a short payload comes back as 25x25 rather than the 21x21 of a
    version-1 symbol. That border has to go before anything else happens, because
    the marker map defines a marker's size as the side of the *symbol*: leaving
    the border in place would make every printed marker 25/21 of the size the map
    says it is and every position estimate would come out 19 per cent long.

    That was not a hypothetical. It is what the first end-to-end run did and it
    showed up as a reprojection residual of about twenty pixels on a solve that
    was otherwise perfectly well conditioned.
    """
    encoder = cv2.QRCodeEncoder_create()
    symbol = encoder.encode(payload)
    if symbol is None or symbol.size == 0:
        raise QrEncodeError(f"OpenCV could not encode payload {payload!r}")
    symbol = np.ascontiguousarray(symbol.astype(np.uint8))

    border = 0
    while (
        symbol.shape[0] - 2 * border > 8
        and symbol[border, :].min() == 255
        and symbol[-1 - border, :].min() == 255
        and symbol[:, border].min() == 255
        and symbol[:, -1 - border].min() == 255
    ):
        border += 1
    if border:
        symbol = np.ascontiguousarray(symbol[border:-border, border:-border])
    return symbol


def symbol_modules(payload: str) -> int:
    """Side length of the encoded symbol in modules.

    Used to work out how many image pixels per module a given viewing distance
    gives, which is the quantity that actually decides whether the detector will
    succeed. OpenCV's decoder needs roughly three pixels per module on a clean
    synthetic image and closer to five on a real one.
    """
    return int(_encode(payload).shape[0])


def render_sheet(payload: str, module_px: int = 6, quiet_zone_modules: int = 4) -> tuple:
    """The printed sheet: symbol plus quiet zone, at an exact module size.

    Returns ``(bitmap, symbol_px, total_px)``. The caller needs all three because
    the marker map defines a marker's size as the side of the *symbol*, not of
    the sheet, so placing the sheet in the world means scaling by
    ``total_px / symbol_px``. Doing that arithmetic here rather than at each call
    site is the difference between a marker that is the size the map says and one
    that is 38 per cent larger, which would put a constant scale error into every
    position estimate.
    """
    if module_px < 1:
        raise ValueError("module_px must be at least 1")
    symbol = _encode(payload)
    modules = symbol.shape[0]
    symbol_px = modules * module_px
    total_px = symbol_px + 2 * quiet_zone_modules * module_px
    scaled = cv2.resize(symbol, (symbol_px, symbol_px), interpolation=cv2.INTER_NEAREST)
    canvas = np.full((total_px, total_px), 255, dtype=np.uint8)
    offset = quiet_zone_modules * module_px
    canvas[offset : offset + symbol_px, offset : offset + symbol_px] = scaled
    return canvas, symbol_px, total_px


def decodable_range(
    payload: str,
    marker_size_m: float,
    focal_px: float,
    pixels_per_module: float = 2.6,
) -> float:
    """Rough maximum range at which a marker should still decode, metres.

    A first-order sanity check, not a measurement. The measured curve is in
    ``results/detection_range.csv`` and is produced by
    ``scripts/measure_detection.py`` against the renderer; this function exists
    so that a map can be checked for obvious impossibility before anything is
    flown at it.

    ``pixels_per_module`` of 2.6 is where the measured curve puts the point at
    which detection starts to fall away, for a symbol viewed square on with the
    renderer's default noise and blur: 54 pixels across a 21-module symbol at
    3.8 m, still decoding 85 per cent of the time, against 15 per cent one step
    further out.
    """
    modules = symbol_modules(payload)
    required_px = pixels_per_module * modules
    return float(focal_px * marker_size_m / required_px)
