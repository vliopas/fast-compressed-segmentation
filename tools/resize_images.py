from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageOps


SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}


def resize_image(src_path: Path, dst_dir: Path, target_width: int, suffix: str) -> Path:
    img = Image.open(src_path)
    img = ImageOps.exif_transpose(img)  # respect orientation from EXIF

    if img.width <= target_width:
        # Skip upscaling; just copy with new name
        resized = img
    else:
        scale = target_width / float(img.width)
        new_size = (target_width, int(img.height * scale))
        resized = img.resize(new_size, Image.Resampling.LANCZOS)

    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_path = dst_dir / f"{src_path.stem}{suffix}{src_path.suffix}"

    save_kwargs: dict[str, object] = {"optimize": True}
    if src_path.suffix.lower() in {".jpg", ".jpeg"}:
        save_kwargs["quality"] = 90

    resized.save(dst_path, **save_kwargs)
    return dst_path


def collect_images(folder: Path) -> list[Path]:
    return [p for p in folder.iterdir() if p.suffix.lower() in SUPPORTED_EXTS and p.is_file()]


def main() -> None:
    default_img_dir = Path(__file__).resolve().parent.parent / "img"

    parser = argparse.ArgumentParser(description="Resize images to a target width while keeping aspect ratio.")
    parser.add_argument(
        "--dir",
        type=Path,
        default=default_img_dir,
        help="Input directory containing images (default: repo_root/img)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to the input directory.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=700,
        help="Target width in pixels (default: 700).",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_700w",
        help="Suffix to append to resized files (default: _700w).",
    )
    args = parser.parse_args()

    src_dir = args.dir
    out_dir = args.out or src_dir

    if not src_dir.exists() or not src_dir.is_dir():
        raise SystemExit(f"Input directory not found: {src_dir}")

    images = collect_images(src_dir)
    if not images:
        raise SystemExit(f"No supported images found in {src_dir}")

    written = [resize_image(img, out_dir, args.width, args.suffix) for img in images]
    print("Resized files:")
    for path in written:
        print(f" - {path}")


if __name__ == "__main__":
    main()
