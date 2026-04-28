#!/usr/bin/env python3
"""Convert segmentation datasets to FFCV format for accelerated loading.

This script provides a command-line interface for converting common
segmentation datasets (Cityscapes, ADE20K, Pascal VOC, Pascal VOC Aug)
to FFCV format.

Datasets must be downloaded first using prepare_data.py.

Usage:
    # Convert Cityscapes
    python convert_to_ffcv.py cityscapes --root ./data --split train

    # Convert ADE20K
    python convert_to_ffcv.py ade20k --root ./data --split train

    # Convert Pascal VOC
    python convert_to_ffcv.py voc --root ./data --split train --year 2012

    # Convert Pascal VOC Aug
    python convert_to_ffcv.py voc-aug --root ./data --split train
"""

import argparse
import logging
from pathlib import Path

import rootutils

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)


def find_project_root() -> Path:
    """Find the project root directory."""
    return rootutils.find_root(search_from=__file__, indicator=".project-root")


def convert_cityscapes(
    root: str,
    split: str,
    output: str | None,
    target_size: tuple[int, int] | None,
    mode: str,
) -> None:
    """Convert Cityscapes dataset to FFCV format.

    Note: Cityscapes does not support automatic download. You must manually
    download the dataset from https://www.cityscapes-dataset.com/ and place
    it in the specified root directory.
    """
    from data.ffcv import convert_cityscapes_to_ffcv

    log.info(f"Converting Cityscapes {split} split (mode={mode})...")
    output_path = convert_cityscapes_to_ffcv(
        root=root,
        split=split,
        output_path=output,
        target_size=target_size,
        mode=mode,
    )
    log.info(f"Cityscapes FFCV file created: {output_path}")


def convert_ade20k(
    root: str,
    split: str,
    output: str | None,
    target_size: tuple[int, int] | None,
) -> None:
    """Convert ADE20K dataset to FFCV format."""
    from data.ffcv import convert_ade20k_to_ffcv

    log.info(f"Converting ADE20K {split} split...")
    output_path = convert_ade20k_to_ffcv(
        root=root,
        split=split,
        output_path=output,
        target_size=target_size,
    )
    log.info(f"ADE20K FFCV file created: {output_path}")


def convert_voc(
    root: str,
    split: str,
    output: str | None,
    target_size: tuple[int, int] | None,
    year: str,
) -> None:
    """Convert Pascal VOC dataset to FFCV format."""
    from data.ffcv import convert_voc_to_ffcv

    log.info(f"Converting Pascal VOC {year} {split} split...")
    output_path = convert_voc_to_ffcv(
        root=root,
        split=split,
        output_path=output,
        target_size=target_size,
        year=year,
    )
    log.info(f"Pascal VOC FFCV file created: {output_path}")


def convert_voc_aug(
    root: str,
    split: str,
    output: str | None,
    target_size: tuple[int, int] | None,
    year: str,
) -> None:
    """Convert Pascal VOC Augmented dataset to FFCV format."""
    from data.ffcv import convert_voc_aug_to_ffcv

    log.info(f"Converting Pascal VOC Aug {year} {split} split...")
    output_path = convert_voc_aug_to_ffcv(
        root=root,
        split=split,
        output_path=output,
        target_size=target_size,
        year=year,
    )
    log.info(f"Pascal VOC Aug FFCV file created: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert segmentation datasets to FFCV format"
    )
    subparsers = parser.add_subparsers(dest="dataset", help="Dataset to convert")

    # Cityscapes
    cityscapes_parser = subparsers.add_parser("cityscapes", help="Cityscapes dataset")
    cityscapes_parser.add_argument(
        "--root", type=str, required=True, help="Root directory of Cityscapes dataset"
    )
    cityscapes_parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Dataset split",
    )
    cityscapes_parser.add_argument(
        "--output", type=str, default=None, help="Output FFCV file path"
    )
    cityscapes_parser.add_argument(
        "--target-size",
        type=int,
        nargs=2,
        default=None,
        metavar=("WIDTH", "HEIGHT"),
        help="Target size for resizing (WIDTH HEIGHT)",
    )
    cityscapes_parser.add_argument(
        "--mode",
        type=str,
        default="fine",
        choices=["fine", "coarse"],
        help="Annotation mode",
    )

    # ADE20K
    ade20k_parser = subparsers.add_parser("ade20k", help="ADE20K dataset")
    ade20k_parser.add_argument(
        "--root", type=str, required=True, help="Root directory of ADE20K dataset"
    )
    ade20k_parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val"],
        help="Dataset split",
    )
    ade20k_parser.add_argument(
        "--output", type=str, default=None, help="Output FFCV file path"
    )
    ade20k_parser.add_argument(
        "--target-size",
        type=int,
        nargs=2,
        default=None,
        metavar=("WIDTH", "HEIGHT"),
        help="Target size for resizing (WIDTH HEIGHT)",
    )
    # Pascal VOC
    voc_parser = subparsers.add_parser("voc", help="Pascal VOC dataset")
    voc_parser.add_argument(
        "--root", type=str, required=True, help="Root directory of Pascal VOC dataset"
    )
    voc_parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "trainval", "val", "test"],
        help="Dataset split",
    )
    voc_parser.add_argument(
        "--output", type=str, default=None, help="Output FFCV file path"
    )
    voc_parser.add_argument(
        "--target-size",
        type=int,
        nargs=2,
        default=None,
        metavar=("WIDTH", "HEIGHT"),
        help="Target size for resizing (WIDTH HEIGHT)",
    )
    voc_parser.add_argument(
        "--year",
        type=str,
        default="2012",
        choices=["2007", "2012", "2007-test"],
        help="Dataset year",
    )
    # Pascal VOC Aug
    voc_aug_parser = subparsers.add_parser(
        "voc-aug", help="Pascal VOC Augmented dataset"
    )
    voc_aug_parser.add_argument(
        "--root", type=str, required=True, help="Root directory of Pascal VOC dataset"
    )
    voc_aug_parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "trainval", "val"],
        help="Dataset split",
    )
    voc_aug_parser.add_argument(
        "--output", type=str, default=None, help="Output FFCV file path"
    )
    voc_aug_parser.add_argument(
        "--target-size",
        type=int,
        nargs=2,
        default=None,
        metavar=("WIDTH", "HEIGHT"),
        help="Target size for resizing (WIDTH HEIGHT)",
    )
    voc_aug_parser.add_argument(
        "--year",
        type=str,
        default="2012",
        choices=["2012"],
        help="Dataset year",
    )
    args = parser.parse_args()

    if args.dataset is None:
        parser.print_help()
        return

    target_size = tuple(args.target_size) if args.target_size else None

    if args.dataset == "cityscapes":
        convert_cityscapes(
            root=args.root,
            split=args.split,
            output=args.output,
            target_size=target_size,
            mode=args.mode,
        )
    elif args.dataset == "ade20k":
        convert_ade20k(
            root=args.root,
            split=args.split,
            output=args.output,
            target_size=target_size,
        )
    elif args.dataset == "voc":
        convert_voc(
            root=args.root,
            split=args.split,
            output=args.output,
            target_size=target_size,
            year=args.year,
        )
    elif args.dataset == "voc-aug":
        convert_voc_aug(
            root=args.root,
            split=args.split,
            output=args.output,
            target_size=target_size,
            year=args.year,
        )


if __name__ == "__main__":
    main()
