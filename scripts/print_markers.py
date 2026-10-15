#!/usr/bin/env python3
"""Generate printable marker sheets for the markers in the map.

One PNG per marker at a stated physical size, plus a contact sheet showing where
each one goes. Printing them is the first physical step of building the test area
and it is the step where the most expensive mistake is available: a marker printed
at the wrong size scales every position estimate by exactly that factor, with a
perfectly healthy-looking reprojection residual and nothing anywhere to suggest a
problem.

So each sheet carries, printed on it:

* the marker identifier and the payload it encodes;
* the intended symbol size in millimetres, measured across the outer edge of the
  symbol and **not** including the white quiet zone;
* crop marks at exactly that size, so the size can be checked with a ruler
  against the paper rather than trusted to the print dialogue;
* the surveyed position and the wall it belongs on.

Print at 100 per cent scale with any "fit to page" option off, then measure the
crop marks. If they are not the stated size, the print scaled and the sheet is
useless.

The payload is deliberately short and ASCII. A long payload needs a higher QR
version, which means more modules in the same physical square, which means fewer
pixels per module at the same range, which means less detection range for no
benefit at all.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "tello_core"))

import cv2
import numpy as np

from tello_core.localisation.markers import MarkerMap
from tello_core.sim.qr import render_sheet, symbol_modules

MM_PER_INCH = 25.4


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--map", dest="marker_map", default="maps/marker_map.yaml")
    parser.add_argument("--out", default="build/markers")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--quiet-zone-modules",
        type=int,
        default=4,
        help="white border in modules. Four is what the QR standard asks for and "
             "what the detector's finder-pattern search needs.",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    marker_map = MarkerMap.from_yaml(REPO_ROOT / args.marker_map)
    out = REPO_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    sheets = []
    for marker in marker_map:
        path = out / f"{marker.marker_id}.png"
        image = _sheet(marker, args.dpi, args.quiet_zone_modules)
        cv2.imwrite(str(path), image)
        sheets.append((marker, path, image))
        print(
            f"  {marker.marker_id}  {marker.payload:<8} "
            f"{marker.size * 1000:.0f} mm symbol, {symbol_modules(marker.payload)} modules, "
            f"{image.shape[1]}x{image.shape[0]} px at {args.dpi} dpi  ->  "
            f"{path.relative_to(REPO_ROOT)}"
        )

    contact = out / "contact_sheet.png"
    cv2.imwrite(str(contact), _contact_sheet(sheets))
    print(f"\ncontact sheet  {contact.relative_to(REPO_ROOT)}")
    _print_instructions(marker_map, args)
    return 0


def _sheet(marker, dpi: int, quiet_zone_modules: int) -> np.ndarray:
    """One marker on a white page with its label and crop marks."""
    modules = symbol_modules(marker.payload)
    pixels_per_mm = dpi / MM_PER_INCH
    symbol_px = int(round(marker.size * 1000 * pixels_per_mm))
    module_px = max(1, symbol_px // modules)

    bitmap, actual_symbol_px, total_px = render_sheet(
        marker.payload, module_px=module_px, quiet_zone_modules=quiet_zone_modules
    )

    margin = int(round(18 * pixels_per_mm))
    label_height = int(round(22 * pixels_per_mm))
    height = total_px + 2 * margin + label_height
    width = total_px + 2 * margin
    page = np.full((height, width), 255, dtype=np.uint8)
    page[margin : margin + total_px, margin : margin + total_px] = bitmap

    # Crop marks at the symbol's own corners, which is what a ruler should be
    # held against. Drawn outside the quiet zone so they cannot interfere with
    # the detector if the sheet is used as printed.
    quiet_px = quiet_zone_modules * module_px
    left = margin + quiet_px
    top = margin + quiet_px
    right = left + actual_symbol_px
    bottom = top + actual_symbol_px
    tick = int(round(6 * pixels_per_mm))
    gap = int(round(2 * pixels_per_mm))
    for x, y, dx, dy in (
        (left, top, -1, -1), (right, top, 1, -1),
        (left, bottom, -1, 1), (right, bottom, 1, 1),
    ):
        cv2.line(page, (x + dx * gap, y), (x + dx * (gap + tick), y), 0, 2)
        cv2.line(page, (x, y + dy * gap), (x, y + dy * (gap + tick)), 0, 2)

    scale = pixels_per_mm / 4.0
    baseline = margin + total_px + int(round(9 * pixels_per_mm))
    lines = [
        f"{marker.marker_id}   payload {marker.payload}",
        f"symbol {marker.size * 1000:.0f} mm across the crop marks "
        f"(quiet zone NOT included)   {modules} modules",
        f"wall position ({marker.position[0]:.3f}, {marker.position[1]:.3f}, "
        f"{marker.position[2]:.3f}) m   face bearing "
        f"{np.rad2deg(marker.normal_yaw):.0f} deg",
        "print at 100 per cent, no fit-to-page, then measure the crop marks",
    ]
    for index, text in enumerate(lines):
        cv2.putText(
            page, text, (margin, baseline + int(index * 5.0 * pixels_per_mm)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42 * scale, 0, max(1, int(scale)), cv2.LINE_AA,
        )
    return page


def _contact_sheet(sheets) -> np.ndarray:
    """All the markers at thumbnail size, so a sheet cannot be taped up wrong."""
    if not sheets:
        return np.full((100, 100), 255, dtype=np.uint8)
    thumb = 240
    columns = min(6, len(sheets))
    rows = int(np.ceil(len(sheets) / columns))
    label = 42
    page = np.full((rows * (thumb + label), columns * thumb), 255, dtype=np.uint8)
    for index, (marker, _, image) in enumerate(sheets):
        row, column = divmod(index, columns)
        small = cv2.resize(image, (thumb - 16, thumb - 16), interpolation=cv2.INTER_AREA)
        y = row * (thumb + label) + 8
        x = column * thumb + 8
        page[y : y + small.shape[0], x : x + small.shape[1]] = small
        cv2.putText(
            page, marker.marker_id, (x, y + small.shape[0] + 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1, cv2.LINE_AA,
        )
        cv2.putText(
            page,
            f"({marker.position[0]:.2f},{marker.position[1]:.2f},{marker.position[2]:.2f})",
            (x, y + small.shape[0] + 34),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, 0, 1, cv2.LINE_AA,
        )
    return page


def _print_instructions(marker_map: MarkerMap, args) -> None:
    size_mm = max(m.size for m in marker_map) * 1000
    quiet_mm = size_mm * args.quiet_zone_modules / symbol_modules(
        marker_map.markers[0].payload
    )
    sheet_mm = size_mm + 2 * quiet_mm
    print()
    print("putting them up")
    print(f"  symbol      {size_mm:.0f} mm across, quiet zone {quiet_mm:.0f} mm each side")
    print(f"  sheet       {sheet_mm:.0f} mm square, so A2 portrait (420 mm) fits it")
    print("  1  print at 100 per cent scale, fit-to-page off")
    print("  2  measure between the crop marks. Wrong size means the print scaled,")
    print("     and a marker 5 per cent too small makes every position estimate")
    print("     5 per cent too far away with nothing to indicate a problem")
    print("  3  tape it flat. A crease through the timing pattern costs detection")
    print("     range; a reflection off a ceiling light costs more")
    print("  4  measure the centre of the symbol, not of the paper and put that")
    print("     into maps/marker_map.yaml")
    print("  5  set survey.method in the map to how you measured it and")
    print("     survey.stated_accuracy_m to how well")


if __name__ == "__main__":
    raise SystemExit(main())
