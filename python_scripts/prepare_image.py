#!/usr/bin/env python3
"""Image utilities for the stream blur demo.

Subcommands:
    prepare  -- Convert any image to a square grayscale PGM
    topng    -- Convert a PGM to PNG (for slides / reports)

Examples:
    python prepare_image.py prepare ../input/test.jpg --size 4096 -o ../input/test.pgm
    python prepare_image.py topng   ../output/blurred.pgm -o ../output/blurred.png
"""

import argparse
import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("Pillow is required: pip install Pillow", file=sys.stderr)
    sys.exit(1)


def write_pgm(path: str, width: int, height: int, data: bytes):
    with open(path, "wb") as f:
        header = f"P5\n{width} {height}\n255\n".encode("ascii")
        f.write(header)
        f.write(data)


def cmd_prepare(args):
    size = args.size
    if args.input:
        img = Image.open(args.input).convert("L")
        img = img.resize((size, size), Image.LANCZOS)
        out_path = args.output or str(Path(args.input).with_suffix(".pgm"))
    else:
        img = Image.new("L", (size, size))
        pixels = img.load()
        for y in range(size):
            for x in range(size):
                pixels[x, y] = (x + y) % 256
        out_path = args.output or "gradient.pgm"

    write_pgm(out_path, size, size, img.tobytes())
    print(f"Wrote {size}x{size} grayscale PGM to {out_path}")
    file_mb = (size * size) / (1024 * 1024)
    print(f"  {file_mb:.1f} MB of pixel data")


def cmd_topng(args):
    img = Image.open(args.input)
    out_path = args.output or str(Path(args.input).with_suffix(".png"))
    img.save(out_path)
    print(f"Converted {args.input} -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command")

    p_prep = sub.add_parser("prepare", help="Convert image to square grayscale PGM")
    p_prep.add_argument("input", nargs="?", help="Input image (JPG, PNG, etc.)")
    p_prep.add_argument("--size", type=int, default=4096)
    p_prep.add_argument("-o", "--output", default=None)

    p_png = sub.add_parser("topng", help="Convert PGM to PNG")
    p_png.add_argument("input", help="Input PGM file")
    p_png.add_argument("-o", "--output", default=None)

    args = parser.parse_args()
    if args.command == "prepare":
        cmd_prepare(args)
    elif args.command == "topng":
        cmd_topng(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
