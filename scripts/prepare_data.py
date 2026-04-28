"""Prepare segmentation datasets: Cityscapes, PASCAL VOC, ADE20K

Utility functions (download, check_sha1, mkdir) extracted from:
https://github.com/zhanghang1989/PyTorch-Encoding/blob/master/encoding/utils/files.py
"""

import argparse
import errno
import hashlib
import os
import subprocess
import tarfile
import zipfile

import requests
from tqdm import tqdm

DATASET_ALIASES = {
    "cityscapes": "cityscapes",
    "pascal": "pascal",
    "pascal-voc": "pascal-voc",
    "voc": "pascal-voc",
    "pascal-voc-aug": "pascal-voc-aug",
    "voc-aug": "pascal-voc-aug",
    "ade20k": "ade20k",
    "all": "all",
}


def check_sha1(filename, sha1_hash):
    """Check whether the sha1 hash of the file content matches the expected hash."""
    sha1 = hashlib.sha1()
    with open(filename, "rb") as f:
        while True:
            data = f.read(1048576)
            if not data:
                break
            sha1.update(data)
    return sha1.hexdigest() == sha1_hash


def download(url, path=None, overwrite=False, sha1_hash=None):
    """Download a given URL.

    Parameters
    ----------
    url : str
        URL to download.
    path : str, optional
        Destination path to store downloaded file.
    overwrite : bool, optional
        Whether to overwrite destination file if it already exists.
    sha1_hash : str, optional
        Expected sha1 hash in hexadecimal digits.

    Returns
    -------
    str
        The file path of the downloaded file.
    """
    if path is None:
        fname = url.split("/")[-1]
    else:
        path = os.path.expanduser(path)
        if os.path.isdir(path):
            fname = os.path.join(path, url.split("/")[-1])
        else:
            fname = path

    needs_download = (
        overwrite
        or not os.path.exists(fname)
        or (sha1_hash and not check_sha1(fname, sha1_hash))
    )
    if not needs_download:
        print("File already exists, skipping: %s" % fname)
        return fname

    dirname = os.path.dirname(os.path.abspath(os.path.expanduser(fname)))
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    print("Downloading %s from %s..." % (fname, url))
    r = requests.get(url, stream=True)
    if r.status_code != 200:
        raise RuntimeError("Failed downloading url %s" % url)
    total_length = r.headers.get("content-length")
    with open(fname, "wb") as f:
        if total_length is None:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        else:
            total_length = int(total_length)
            for chunk in tqdm(
                r.iter_content(chunk_size=1024),
                total=int(total_length / 1024.0 + 0.5),
                unit="KB",
                unit_scale=False,
                dynamic_ncols=True,
            ):
                f.write(chunk)

    if sha1_hash and not check_sha1(fname, sha1_hash):
        raise UserWarning(
            "File {} is downloaded but the content hash does not match. "
            "The repo may be outdated or download may be incomplete. "
            'If the "repo_url" is overridden, consider switching to '
            "the default repo.".format(fname)
        )

    return fname


def mkdir(path):
    """Make directory, ok if it already exists."""
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


# ---------------------------------------------------------------------------
# Dataset download functions
# ---------------------------------------------------------------------------


def download_city(path, overwrite=False):
    city_dir = os.path.join(path, "cityscapes")
    mkdir(city_dir)
    _CITY_PACKAGES = ["gtFine_trainvaltest.zip", "leftImg8bit_trainvaltest.zip"]
    for pkg in _CITY_PACKAGES:
        local_file = os.path.join(city_dir, pkg)
        if os.path.exists(local_file) and not overwrite:
            print("File already exists, skipping: %s" % local_file)
        else:
            if os.path.exists(local_file) and overwrite:
                os.remove(local_file)
            subprocess.check_call(["csDownload", "-d", city_dir, pkg])
        # Extract the downloaded zip
        print("Extracting %s..." % local_file)
        with zipfile.ZipFile(local_file, "r") as zip_ref:
            zip_ref.extractall(path=city_dir)

    # Convert Cityscapes label IDs to train IDs
    print("Converting Cityscapes labelIds to trainIds...")
    env = os.environ.copy()
    env["CITYSCAPES_DATASET"] = city_dir
    subprocess.check_call(["csCreateTrainIdLabelImgs"], env=env)


def download_voc(path, overwrite=False):
    voc_dir = os.path.join(path, "pascal_voc")
    mkdir(voc_dir)
    _DOWNLOAD_URLS = [
        (
            "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
            "4e443f8a2eca6b1dac8a6c57641b67dd40621a49",
        )
    ]
    for url, checksum in _DOWNLOAD_URLS:
        filename = download(url, path=voc_dir, overwrite=overwrite, sha1_hash=checksum)
        with tarfile.open(filename) as tar:
            tar.extractall(path=voc_dir)


def download_aug(path, overwrite=False):
    """Download SBD, convert .mat masks to PNG, and generate VOC Aug split files.

    After running, the VOC directory will contain:
      - pascal_voc/VOCdevkit/VOC2012/SegmentationClassAug/*.png
      - pascal_voc/VOCdevkit/VOC2012/ImageSets/Segmentation/train_aug.txt
      - pascal_voc/VOCdevkit/VOC2012/ImageSets/Segmentation/trainval_aug.txt
    """
    import numpy as np
    from PIL import Image
    from scipy.io import loadmat

    voc_dir = os.path.join(path, "pascal_voc")
    mkdir(voc_dir)

    _AUG_DOWNLOAD_URLS = [
        (
            "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz",
            "7129e0a480c2d6afb02b517bb18ac54283bfaa35",
        )
    ]

    # 1. Download SBD archive
    for url, checksum in _AUG_DOWNLOAD_URLS:
        filename = download(url, path=voc_dir, overwrite=overwrite, sha1_hash=checksum)

    # 2. Extract SBD to a temporary location
    sbd_dir = os.path.join(voc_dir, "benchmark_RELEASE", "dataset")
    if not os.path.isdir(sbd_dir) or overwrite:
        with tarfile.open(filename) as tar:
            tar.extractall(path=voc_dir)

    # 3. Resolve VOC root
    voc_root = os.path.join(voc_dir, "VOCdevkit", "VOC2012")
    if not os.path.isdir(voc_root):
        raise FileNotFoundError(
            "VOC2012 not found at %s. Run with --dataset pascal to download VOC first."
            % voc_root
        )

    target_dir = os.path.join(voc_root, "SegmentationClassAug")
    imagesets_dir = os.path.join(voc_root, "ImageSets", "Segmentation")

    # 4. Convert .mat to .png
    cls_dir = os.path.join(sbd_dir, "cls")
    mat_files = sorted(f for f in os.listdir(cls_dir) if f.endswith(".mat"))

    if os.path.isdir(target_dir) and not overwrite:
        existing = [f for f in os.listdir(target_dir) if f.endswith(".png")]
        if len(existing) >= len(mat_files):
            print("File already exists, skipping: %s" % target_dir)
            return

    mkdir(target_dir)
    print("Converting %d SBD .mat masks to PNG..." % len(mat_files))
    for mat_file in tqdm(mat_files, unit="file"):
        img_id = os.path.splitext(mat_file)[0]
        png_path = os.path.join(target_dir, img_id + ".png")
        if os.path.exists(png_path) and not overwrite:
            continue
        mat_data = loadmat(os.path.join(cls_dir, mat_file))
        segmentation = mat_data["GTcls"][0, 0]["Segmentation"]
        seg_img = Image.fromarray(segmentation.astype(np.uint8))
        seg_img.save(png_path)

    # 5. Generate split files (train_aug.txt, trainval_aug.txt)
    sbd_train_f = os.path.join(sbd_dir, "train.txt")
    sbd_val_f = os.path.join(sbd_dir, "val.txt")
    with open(sbd_train_f) as f:
        sbd_train_ids = set(line.strip() for line in f if line.strip())
    with open(sbd_val_f) as f:
        sbd_val_ids = set(line.strip() for line in f if line.strip())

    # Read original VOC train/val splits
    voc_train_f = os.path.join(imagesets_dir, "train.txt")
    voc_val_f = os.path.join(imagesets_dir, "val.txt")
    with open(voc_train_f) as f:
        voc_train_ids = set(line.strip() for line in f if line.strip())
    with open(voc_val_f) as f:
        voc_val_ids = set(line.strip() for line in f if line.strip())

    # train_aug = (SBD train + SBD val + VOC train) minus VOC val
    train_aug_ids = sorted((sbd_train_ids | sbd_val_ids | voc_train_ids) - voc_val_ids)
    trainval_aug_ids = sorted(sbd_train_ids | sbd_val_ids | voc_train_ids | voc_val_ids)

    with open(os.path.join(imagesets_dir, "train_aug.txt"), "w") as f:
        f.write("\n".join(train_aug_ids) + "\n")
    with open(os.path.join(imagesets_dir, "trainval_aug.txt"), "w") as f:
        f.write("\n".join(trainval_aug_ids) + "\n")

    print(
        "Created train_aug.txt (%d images), trainval_aug.txt (%d images)"
        % (len(train_aug_ids), len(trainval_aug_ids))
    )


def download_ade(path, overwrite=False):
    ade_dir = os.path.join(path, "ade20k")
    mkdir(ade_dir)
    _ADE_DOWNLOAD_URLS = [
        (
            "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip",
            "219e1696abb36c8ba3a3afe7fb2f4b4606a897c7",
        ),
        (
            "http://data.csail.mit.edu/places/ADEchallenge/release_test.zip",
            "e05747892219d10e9243933371a497e905a4860c",
        ),
    ]
    for url, checksum in _ADE_DOWNLOAD_URLS:
        filename = download(url, path=ade_dir, overwrite=overwrite, sha1_hash=checksum)
        with zipfile.ZipFile(filename, "r") as zip_ref:
            zip_ref.extractall(path=ade_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare segmentation datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(DATASET_ALIASES),
        help="dataset to prepare",
    )
    parser.add_argument(
        "--root", required=True, help="directory where datasets will be stored"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="overwrite downloaded files if set"
    )
    args = parser.parse_args()

    dataset = DATASET_ALIASES[args.dataset]
    target = os.path.expanduser(args.root)
    mkdir(target)

    if dataset in ("cityscapes", "all"):
        print("Preparing Cityscapes...")
        download_city(target, overwrite=args.overwrite)

    if dataset in ("pascal", "pascal-voc", "all"):
        print("Preparing PASCAL VOC...")
        download_voc(target, overwrite=args.overwrite)

    if dataset in ("pascal", "pascal-voc-aug", "all"):
        print("Preparing PASCAL VOC Aug...")
        if dataset == "pascal-voc-aug":
            download_voc(target, overwrite=args.overwrite)
        download_aug(target, overwrite=args.overwrite)

    if dataset in ("ade20k", "all"):
        print("Preparing ADE20K...")
        download_ade(target, overwrite=args.overwrite)
