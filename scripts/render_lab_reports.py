"""Render the synthetic OCR-text samples into scanned-looking PNGs for the VLM / OCR path.

    uv run python scripts/render_lab_reports.py [--font PATH] [--out datasets/lab_reports/images]

Adds slight rotation, blur and noise so the images look like phone photos of paper.
Needs a font with Hangul glyphs (NanumGothic, Noto Sans CJK, WenQuanYi Zen Hei, ...).
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "C:/Windows/Fonts/malgun.ttf",
]


def find_font(explicit: str | None) -> str:
    for candidate in [explicit, *FONT_CANDIDATES]:
        if candidate and Path(candidate).exists():
            return candidate
    raise SystemExit("no Hangul font found; pass --font /path/to/font.ttf")


def render(text: str, font_path: str, seed: int) -> Image.Image:
    rng = random.Random(seed)
    font = ImageFont.truetype(font_path, 26)
    lines = text.splitlines()
    width = max(int(font.getlength(line)) for line in lines) + 120
    height = 60 + 40 * len(lines) + 60
    img = Image.new("L", (width, height), 250)
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        draw.text((60, 60 + 40 * i), line, font=font, fill=rng.randint(10, 40))
    for _ in range(width * height // 400):  # speckle noise
        draw.point((rng.randrange(width), rng.randrange(height)), fill=rng.randint(120, 200))
    img = img.rotate(rng.uniform(-1.5, 1.5), expand=True, fillcolor=245, resample=Image.BICUBIC)
    return img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.8)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/lab_reports")
    parser.add_argument("--out", default="datasets/lab_reports/images")
    parser.add_argument("--font")
    args = parser.parse_args()

    font = find_font(args.font)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for i, path in enumerate(sorted(Path(args.dataset).glob("*.ocr.txt"))):
        target = out / path.name.replace(".ocr.txt", ".png")
        render(path.read_text("utf-8"), font, seed=i).save(target)
        print(target)


if __name__ == "__main__":
    main()
