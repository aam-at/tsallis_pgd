import os
from collections import namedtuple

from torchvision.datasets.utils import download_and_extract_archive
from torchvision.datasets.vision import VisionDataset
from utils.pylogger import RankedLogger

from .configs import ADE20KInfo, DatasetInfo
from .mixins import (
    ClassInfoMixin,
    CV2SegmentationLoaderMixin,
    PILSegmentationLoaderMixin,
)

log = RankedLogger(__name__, rank_zero_only=True)

ARCHIVE_DICT = {
    "trainval": {
        "url": (
            "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"
        ),
        "md5": "7328b3957e407ddae1d3cbf487f149ef",
        "base_dir": "ADEChallengeData2016",
    }
}


def get_ade_weights(num_classes: int) -> list[float]:
    """Get ADE20K weights for the specified number of classes."""
    if num_classes != 150:
        raise ValueError("ADE20K weights are only available for 150 classes.")
    ADE_WTS = [
        2.5511e-05,
        3.6983e-05,
        4.7570e-05,
        6.6522e-05,
        1.1128e-04,
        1.0635e-04,
        1.6683e-04,
        1.7398e-04,
        2.2595e-04,
        2.6104e-04,
        3.4525e-04,
        3.2680e-04,
        4.6724e-04,
        2.7333e-04,
        3.8664e-04,
        5.1101e-04,
        4.6147e-04,
        2.8707e-04,
        4.8777e-04,
        5.6734e-04,
        5.2263e-04,
        5.7571e-04,
        7.9284e-04,
        7.1656e-04,
        9.8619e-04,
        7.3393e-04,
        6.0752e-04,
        6.0696e-04,
        1.0648e-03,
        1.5916e-03,
        7.4704e-04,
        1.3956e-03,
        1.0427e-03,
        1.6245e-03,
        1.3812e-03,
        1.2803e-03,
        1.5659e-03,
        2.3384e-03,
        2.6498e-03,
        2.1948e-03,
        1.9984e-03,
        2.1434e-03,
        2.2654e-03,
        2.3339e-03,
        2.6016e-03,
        2.9368e-03,
        2.4439e-03,
        2.5844e-03,
        2.3346e-03,
        1.0170e-03,
        2.7078e-03,
        3.7222e-03,
        3.0739e-03,
        3.0697e-03,
        5.0181e-03,
        4.7774e-03,
        2.0477e-03,
        3.1477e-03,
        2.8421e-03,
        3.7206e-03,
        2.5296e-03,
        2.1699e-03,
        2.8066e-03,
        2.8080e-03,
        5.5795e-03,
        4.0186e-03,
        4.8758e-03,
        3.5471e-03,
        3.1513e-03,
        3.0316e-03,
        3.9002e-03,
        5.0847e-03,
        4.8401e-03,
        5.9311e-03,
        5.3158e-03,
        5.0188e-03,
        4.0362e-03,
        4.4585e-03,
        5.2076e-03,
        4.4833e-03,
        5.5491e-03,
        5.7523e-03,
        5.5545e-03,
        8.7588e-03,
        5.0301e-03,
        5.4497e-03,
        7.6726e-03,
        5.1451e-03,
        7.9943e-03,
        4.4696e-03,
        7.4416e-03,
        6.7389e-03,
        7.8750e-03,
        5.5496e-03,
        1.2515e-02,
        5.1635e-03,
        8.1806e-03,
        9.9495e-03,
        1.0522e-02,
        6.0337e-03,
        1.1848e-02,
        1.0531e-02,
        6.0837e-03,
        8.0876e-03,
        1.1750e-02,
        8.2409e-03,
        6.8528e-03,
        8.1382e-03,
        8.7929e-03,
        7.6437e-03,
        5.7786e-03,
        1.3009e-02,
        1.8844e-02,
        1.0949e-02,
        4.2059e-03,
        5.7906e-03,
        1.2998e-02,
        1.4171e-02,
        7.0287e-03,
        9.0963e-03,
        1.0115e-02,
        1.0510e-02,
        1.3813e-02,
        1.2319e-02,
        1.4154e-02,
        1.5693e-02,
        1.5035e-02,
        1.1120e-02,
        1.6888e-02,
        7.3436e-03,
        1.4521e-02,
        9.3029e-03,
        1.4782e-02,
        1.1918e-02,
        1.7509e-02,
        2.0762e-02,
        1.4547e-02,
        2.0312e-02,
        1.0543e-02,
        1.8876e-02,
        3.6659e-02,
        2.0046e-02,
        2.2035e-02,
        1.4011e-02,
        1.5645e-02,
        1.1985e-02,
        1.0001e-02,
        2.7073e-02,
        2.1668e-02,
        1.8419e-02,
        2.1877e-02,
    ]
    return ADE_WTS


class _BaseAde20kSegmentation(ClassInfoMixin, VisionDataset):
    """`ADE20K Dataset.

    ADE20K <https://groups.csail.mit.edu/vision/datasets/ADE20K/>`_

    Args:
        root (string): Root directory of the ADE20K dataset
        split (string, optional): The image split to use, ``train`` or ``val``
        download (bool, optional): If true, downloads the dataset from the
            internet and puts it in root directory. If dataset is already
            downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that takes in a
            PIL image and returns a transformed version. E.g,
            ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes
            in the PIL image target and transforms it.
        transforms (callable, optional): A function/transform that takes input
            sample and its target as entry and returns a transformed version.

    Examples:
        Get dataset for training and download from internet

        .. code-block:: python

            dataset = ADE20K('./data/ade20k', split='train', download=True)

            img, target = dataset[0]

        Get dataset for validation and download from internet

        .. code-block:: python

            dataset = ADE20K('./data/ade20k', split='val', download=True)

            img, target = dataset[0]
    """

    ADE20KClass = namedtuple("ADE20KClass", ["name", "id", "color"])
    info: DatasetInfo = ADE20KInfo()

    classes = [
        ADE20KClass("wall", 1, (120, 120, 120)),
        ADE20KClass("building;edifice", 2, (180, 120, 120)),
        ADE20KClass("sky", 3, (6, 230, 230)),
        ADE20KClass("floor;flooring", 4, (80, 50, 50)),
        ADE20KClass("tree", 5, (4, 200, 3)),
        ADE20KClass("ceiling", 6, (120, 120, 80)),
        ADE20KClass("road;route", 7, (140, 140, 140)),
        ADE20KClass("bed", 8, (204, 5, 255)),
        ADE20KClass("windowpane;window", 9, (230, 230, 230)),
        ADE20KClass("grass", 10, (4, 250, 7)),
        ADE20KClass("cabinet", 11, (224, 5, 255)),
        ADE20KClass("sidewalk;pavement", 12, (235, 255, 7)),
        ADE20KClass("person", 13, (150, 5, 61)),
        ADE20KClass("earth;ground", 14, (120, 120, 70)),
        ADE20KClass("door;double;door", 15, (8, 255, 51)),
        ADE20KClass("table", 16, (255, 6, 82)),
        ADE20KClass("mountain;mount", 17, (143, 255, 140)),
        ADE20KClass("plant;flora;plant;life", 18, (204, 255, 4)),
        ADE20KClass("curtain;drape;drapery;mantle;pall", 19, (255, 51, 7)),
        ADE20KClass("chair", 20, (204, 70, 3)),
        ADE20KClass("car;auto;automobile;machine;motorcar", 21, (0, 102, 200)),
        ADE20KClass("water", 22, (61, 230, 250)),
        ADE20KClass("painting;picture", 23, (255, 6, 51)),
        ADE20KClass("sofa;couch;lounge", 24, (11, 102, 255)),
        ADE20KClass("shelf", 25, (255, 7, 71)),
        ADE20KClass("house", 26, (255, 9, 224)),
        ADE20KClass("sea", 27, (9, 7, 230)),
        ADE20KClass("mirror", 28, (220, 220, 220)),
        ADE20KClass("rug;carpet;carpeting", 29, (255, 9, 92)),
        ADE20KClass("field", 30, (112, 9, 255)),
        ADE20KClass("armchair", 31, (8, 255, 214)),
        ADE20KClass("seat", 32, (7, 255, 224)),
        ADE20KClass("fence;fencing", 33, (255, 184, 6)),
        ADE20KClass("desk", 34, (10, 255, 71)),
        ADE20KClass("rock;stone", 35, (255, 41, 10)),
        ADE20KClass("wardrobe;closet;press", 36, (7, 255, 255)),
        ADE20KClass("lamp", 37, (224, 255, 8)),
        ADE20KClass("bathtub;bathing;tub;bath;tub", 38, (102, 8, 255)),
        ADE20KClass("railing;rail", 39, (255, 61, 6)),
        ADE20KClass("cushion", 40, (255, 194, 7)),
        ADE20KClass("base;pedestal;stand", 41, (255, 122, 8)),
        ADE20KClass("box", 42, (0, 255, 20)),
        ADE20KClass("column;pillar", 43, (255, 8, 41)),
        ADE20KClass("signboard;sign", 44, (255, 5, 153)),
        ADE20KClass("chest;of;drawers;chest;bureau;dresser", 45, (6, 51, 255)),
        ADE20KClass("counter", 46, (235, 12, 255)),
        ADE20KClass("sand", 47, (160, 150, 20)),
        ADE20KClass("sink", 48, (0, 163, 255)),
        ADE20KClass("skyscraper", 49, (140, 140, 140)),
        ADE20KClass("fireplace;hearth;open;fireplace", 50, (250, 10, 15)),
        ADE20KClass("refrigerator;icebox", 51, (20, 255, 0)),
        ADE20KClass("grandstand;covered;stand", 52, (31, 255, 0)),
        ADE20KClass("path", 53, (255, 31, 0)),
        ADE20KClass("stairs;steps", 54, (255, 224, 0)),
        ADE20KClass("runway", 55, (153, 255, 0)),
        ADE20KClass("case;display;case;showcase;vitrine", 56, (0, 0, 255)),
        ADE20KClass("pool;table;billiard;table;snooker;table", 57, (255, 71, 0)),
        ADE20KClass("pillow", 58, (0, 235, 255)),
        ADE20KClass("screen;door;screen", 59, (0, 173, 255)),
        ADE20KClass("stairway;staircase", 60, (31, 0, 255)),
        ADE20KClass("river", 61, (11, 200, 200)),
        ADE20KClass("bridge;span", 62, (255, 82, 0)),
        ADE20KClass("bookcase", 63, (0, 255, 245)),
        ADE20KClass("blind;screen", 64, (0, 61, 255)),
        ADE20KClass("coffee;table;cocktail;table", 65, (0, 255, 112)),
        ADE20KClass("toilet;can;commode;crapper;pot;potty;stool", 66, (0, 255, 133)),
        ADE20KClass("flower", 67, (255, 0, 0)),
        ADE20KClass("book", 68, (255, 163, 0)),
        ADE20KClass("hill", 69, (255, 102, 0)),
        ADE20KClass("bench", 70, (194, 255, 0)),
        ADE20KClass("countertop", 71, (0, 143, 255)),
        ADE20KClass(
            "stove;kitchen;stove;range;kitchen;cooking;stove", 72, (51, 255, 0)
        ),
        ADE20KClass("palm;palm;tree", 73, (0, 82, 255)),
        ADE20KClass("kitchen;island", 74, (0, 255, 41)),
        ADE20KClass("computer", 75, (0, 255, 173)),
        ADE20KClass("swivel;chair", 76, (10, 0, 255)),
        ADE20KClass("boat", 77, (173, 255, 0)),
        ADE20KClass("bar", 78, (0, 255, 153)),
        ADE20KClass("arcade;machine", 79, (255, 92, 0)),
        ADE20KClass("hovel;hut;hutch;shack;shanty", 80, (255, 0, 255)),
        ADE20KClass("bus;coach;double-decker;passenger;vehicle", 81, (255, 0, 245)),
        ADE20KClass("towel", 82, (255, 0, 102)),
        ADE20KClass("light;light;source", 83, (255, 173, 0)),
        ADE20KClass("truck;motortruck", 84, (255, 0, 20)),
        ADE20KClass("tower", 85, (255, 184, 184)),
        ADE20KClass("chandelier;pendant;pendent", 86, (0, 31, 255)),
        ADE20KClass("awning;sunshade;sunblind", 87, (0, 255, 61)),
        ADE20KClass("streetlight;street;lamp", 88, (0, 71, 255)),
        ADE20KClass("booth;cubicle;stall;kiosk", 89, (255, 0, 204)),
        ADE20KClass("television", 90, (0, 255, 194)),
        ADE20KClass("airplane;aeroplane;plane", 91, (0, 255, 82)),
        ADE20KClass("dirt;track", 92, (0, 10, 255)),
        ADE20KClass("apparel;wearing;apparel;dress;clothes", 93, (0, 112, 255)),
        ADE20KClass("pole", 94, (51, 0, 255)),
        ADE20KClass("land;ground;soil", 95, (0, 194, 255)),
        ADE20KClass(
            "bannister;banister;balustrade;balusters;handrail",
            96,
            (0, 122, 255),
        ),
        ADE20KClass("escalator;moving;staircase;moving;stairway", 97, (0, 255, 163)),
        ADE20KClass("ottoman;pouf;pouffe;puff;hassock", 98, (255, 153, 0)),
        ADE20KClass("bottle", 99, (0, 255, 10)),
        ADE20KClass("buffet;counter;sideboard", 100, (255, 112, 0)),
        ADE20KClass("poster;posting;placard;notice;bill;card", 101, (143, 255, 0)),
        ADE20KClass("stage", 102, (82, 0, 255)),
        ADE20KClass("van", 103, (163, 255, 0)),
        ADE20KClass("ship", 104, (255, 235, 0)),
        ADE20KClass("fountain", 105, (8, 184, 170)),
        ADE20KClass(
            "conveyer;belt;conveyor;belt;conveyor;transporter",
            106,
            (133, 0, 255),
        ),
        ADE20KClass("canopy", 107, (0, 255, 92)),
        ADE20KClass("washer;automatic;washer;washing;machine", 108, (184, 0, 255)),
        ADE20KClass("plaything;toy", 109, (255, 0, 31)),
        ADE20KClass("swimming;pool;swimming;bath;natatorium", 110, (0, 184, 255)),
        ADE20KClass("stool", 111, (0, 214, 255)),
        ADE20KClass("barrel;cask", 112, (255, 0, 112)),
        ADE20KClass("basket;handbasket", 113, (92, 255, 0)),
        ADE20KClass("waterfall;falls", 114, (0, 224, 255)),
        ADE20KClass("tent;collapsible;shelter", 115, (112, 224, 255)),
        ADE20KClass("bag", 116, (70, 184, 160)),
        ADE20KClass("minibike;motorbike", 117, (163, 0, 255)),
        ADE20KClass("cradle", 118, (153, 0, 255)),
        ADE20KClass("oven", 119, (71, 255, 0)),
        ADE20KClass("ball", 120, (255, 0, 163)),
        ADE20KClass("food;solid;food", 121, (255, 204, 0)),
        ADE20KClass("step;stair", 122, (255, 0, 143)),
        ADE20KClass("tank;storage;tank", 123, (0, 255, 235)),
        ADE20KClass("trade;name;brand;name;brand;marque", 124, (133, 255, 0)),
        ADE20KClass("microwave;microwave;oven", 125, (255, 0, 235)),
        ADE20KClass("pot;flowerpot", 126, (245, 0, 255)),
        ADE20KClass(
            "animal;animate;being;beast;brute;creature;fauna",
            127,
            (255, 0, 122),
        ),
        ADE20KClass("bicycle;bike;wheel;cycle", 128, (255, 245, 0)),
        ADE20KClass("lake", 129, (10, 190, 212)),
        ADE20KClass("dishwasher;dish;washer;dishwashing;machine", 130, (214, 255, 0)),
        ADE20KClass("screen;silver;screen;projection;screen", 131, (0, 204, 255)),
        ADE20KClass("blanket;cover", 132, (20, 0, 255)),
        ADE20KClass("sculpture", 133, (255, 255, 0)),
        ADE20KClass("hood;exhaust;hood", 134, (0, 153, 255)),
        ADE20KClass("sconce", 135, (0, 41, 255)),
        ADE20KClass("vase", 136, (0, 255, 204)),
        ADE20KClass("traffic;light;traffic;signal;stoplight", 137, (41, 0, 255)),
        ADE20KClass("tray", 138, (41, 255, 0)),
        ADE20KClass(
            "trash;can;garbage;wastebin;bin;ashbin;dustbin;barrel;bin",
            139,
            (173, 0, 255),
        ),
        ADE20KClass("fan", 140, (0, 245, 255)),
        ADE20KClass("pier;wharf;wharfage;dock", 141, (71, 0, 255)),
        ADE20KClass("crt;screen", 142, (122, 0, 255)),
        ADE20KClass("plate", 143, (0, 255, 184)),
        ADE20KClass("monitor;monitoring;device", 144, (0, 92, 255)),
        ADE20KClass("bulletin;board;notice;board", 145, (184, 255, 0)),
        ADE20KClass("shower", 146, (0, 133, 255)),
        ADE20KClass("radiator", 147, (255, 214, 0)),
        ADE20KClass("glass;drinking;glass", 148, (25, 194, 194)),
        ADE20KClass("clock", 149, (102, 255, 0)),
        ADE20KClass("flag", 150, (92, 0, 255)),
    ]

    def __init__(
        self,
        root,
        split="train",
        download=False,
        transform=None,
        target_transform=None,
        transforms=None,
        target_size=None,
    ):
        super(_BaseAde20kSegmentation, self).__init__(
            root, transforms, transform, target_transform
        )
        self.target_size = target_size

        base_dir = ARCHIVE_DICT["trainval"]["base_dir"]

        if split not in ["train", "val"]:
            raise ValueError('Invalid split! Please use split="train" or split="val"')

        if split == "train":
            self.images_dir = os.path.join(self.root, base_dir, "images", "training")
            self.targets_dir = os.path.join(
                self.root, base_dir, "annotations", "training"
            )
        elif split == "val":
            self.images_dir = os.path.join(self.root, base_dir, "images", "validation")
            self.targets_dir = os.path.join(
                self.root, base_dir, "annotations", "validation"
            )

        self.split = split

        if download:
            self.download()

        self.images = []
        self.masks = []

        for file_name in os.listdir(self.images_dir):
            self.images.append(os.path.join(self.images_dir, file_name))
            self.masks.append(
                os.path.join(self.targets_dir, file_name.replace("jpg", "png"))
            )

    def sort(self):
        # Sort file names
        def extract_sort_key(path):
            return int(path.split("_")[-1].split(".")[0])

        sort_indices = sorted(
            range(len(self)), key=lambda i: extract_sort_key(self.images[i])
        )
        self.images = [self.images[i] for i in sort_indices]
        self.masks = [self.masks[i] for i in sort_indices]

    def download(self):
        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):
            archive_dict = ARCHIVE_DICT["trainval"]
            download_and_extract_archive(
                archive_dict["url"],
                self.root,
                extract_root=self.root,
                md5=archive_dict["md5"],
            )

        else:
            msg = (
                "You set download=True, but a folder ADEChallengeData2016 "
                "already exist in the root directory. If you want to "
                "re-download or re-extract the archive, delete the folder."
            )
            log.info(msg)

    def __getitem__(self, index):
        image, target = self._load_input(index)
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target

    def __len__(self):
        return len(self.images)

    def get_weights(self) -> list[float]:
        """Get class weights for the ADE20K dataset."""
        return get_ade_weights(self.info.num_classes)

    def extra_repr(self):
        lines = ["Split: {split}"]
        return "\n".join(lines).format(**self.__dict__)


class CV2Ade20kSegmentation(CV2SegmentationLoaderMixin, _BaseAde20kSegmentation): ...


class PILAde20kSegmentation(PILSegmentationLoaderMixin, _BaseAde20kSegmentation): ...


class FFCVAde20kSegmentation(CV2SegmentationLoaderMixin, _BaseAde20kSegmentation):
    """ADE20K dataset for creating FFCV format files.

    This class is used during FFCV file creation (prepare_data phase).
    For actual training with FFCV, use FFCVSegmentationDataModule(...).
    """

    def __init__(
        self,
        root,
        split="train",
        download=False,
        transform=None,
        target_transform=None,
        transforms=None,
        target_size=None,
        ffcv_path=None,
    ):
        super().__init__(
            root=root,
            split=split,
            download=download,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
            target_size=target_size,
        )
        self.ffcv_path = ffcv_path
