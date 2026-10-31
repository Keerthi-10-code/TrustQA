"""Make an image-only ("scanned") PDF from a digital PDF, for OCR testing.

Each page is rendered to a grayscale image, slightly rotated, given light noise and
JPEG compression, then written back as a PDF page with no text layer. The original
digital text is saved alongside as OCR ground truth (one .txt per page) for CER/WER.

Usage:
  python scripts/make_scanned_pdf.py SRC.pdf OUT.pdf [--pages 1-9] [--dpi 150]
         [--skew 0.6] [--noise 8] [--truth-dir DIR] [--seed 0]
"""

from __future__ import annotations

import argparse
import io
import random
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image


def parse_pages(spec: str | None, n: int) -> list[int]:
    """'1-3,5' -> [0, 1, 2, 4] (input is 1-indexed, output 0-indexed)."""
    if not spec:
        return list(range(n))
    out: list[int] = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out.extend(range(int(a) - 1, int(b or a)))
    return [p for p in out if 0 <= p < n]


def scan_page(page: pymupdf.Page, dpi: int, skew: float, noise: float, rng: random.Random) -> bytes:
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
    img = Image.frombytes("L", (pix.width, pix.height), pix.samples)
    angle = rng.uniform(-skew, skew)
    img = img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=255)
    if noise > 0:
        arr = np.asarray(img, dtype=np.float32)
        arr += np.random.default_rng(rng.randrange(2**32)).normal(0, noise, arr.shape)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)
    return buf.getvalue()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--pages", help="1-indexed pages, e.g. 1-9 or 1,3,5 (default: all)")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--skew", type=float, default=0.6, help="max rotation in degrees")
    ap.add_argument("--noise", type=float, default=8.0, help="gaussian noise std (0-255)")
    ap.add_argument("--truth-dir", type=Path, help="write original page text here")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    rng = random.Random(args.seed)
    src = pymupdf.open(args.src)
    out = pymupdf.open()
    pages = parse_pages(args.pages, src.page_count)
    if args.truth_dir:
        args.truth_dir.mkdir(parents=True, exist_ok=True)

    for out_no, idx in enumerate(pages, start=1):
        page = src[idx]
        jpg = scan_page(page, args.dpi, args.skew, args.noise, rng)
        new = out.new_page(width=page.rect.width, height=page.rect.height)
        new.insert_image(new.rect, stream=jpg)
        if args.truth_dir:
            (args.truth_dir / f"page_{out_no:03d}.txt").write_text(page.get_text(), encoding="utf-8")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.save(args.out, garbage=4, deflate=True)
    print(f"wrote {args.out} ({len(pages)} pages, image-only)")


if __name__ == "__main__":
    main()
