import json
import os
from typing import Tuple

import numpy as np
from PIL import Image
import torchvision.datasets
import torchvision.transforms
import tqdm

import data_lib

BASE_CLASSES = json.load(open("data_lib/00_info/GTSRB_classnames.json", "r"))
PRIMER = 'A centered photo of a "{}" traffic sign.'
CLASSES = [PRIMER.format(x) for x in BASE_CLASSES]

PARAMS = {
    "base_classes": BASE_CLASSES,
    "classes": CLASSES,
    "num_classes": len(CLASSES),
    "mean": (0.3415, 0.3124, 0.3214),
    "std": (0.1663, 0.1662, 0.1762),
    # Default Imagesize
    "img_size": 224,
    # If it's computationally recommended to resize in advance
    "create_resized_variant_if_possible": False,
    "primer": PRIMER,
    "eval_only": False,
    "type": "classification",
}


class Dataset(data_lib.DatasetScaffold):
    def __init__(self, root, **kwargs) -> None:
        super(Dataset, self).__init__(root, **kwargs)

        self.root = os.path.join(root, "GTSRB")
        self.PARAMS = PARAMS

        self._run_default_setup()

    def _download_and_setup(self):
        if (
            not (os.path.isdir(self.root) and len(os.listdir(self.root)) > 0)
            and self.download
        ):
            print("Downloading dataset...")
            base_url = (
                "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/"
            )
            torchvision.datasets.utils.download_and_extract_archive(
                f"{base_url}GTSRB-Training_fixed.zip",
                download_root=self.root,
                md5="513f3c79a4c5141765e10e952eaa2478",
            )
            # torchvision.datasets.utils.download_and_extract_archive(
            #     f"{base_url}GTSRB_Final_Test_Images.zip",
            #     download_root=self.root,
            #     md5="c7e4e6327067d32654124b0fe9e82185",
            # )
            # torchvision.datasets.utils.download_and_extract_archive(
            #     f"{base_url}GTSRB_Final_Test_GT.zip",
            #     download_root=self.root,
            #     md5="fe31e9c9270bbcd7b84b7f21a9d9d9e5",
            # )

        # train_val_split = 0.7
        # data = [x[0] for x in torchvision.datasets.folder.make_dataset(self.root, extensions=(".ppm"))]
        # classes = [sample.split('GTSRB/Training/')[-1].split('/')[0] for sample in data]
        # data_dict = {}
        # for path, classname in zip(data, classes):
        #     if classname not in data_dict:
        #         data_dict[classname] = []
        #     data_dict[classname].append(path)

        # if self.train:
        #     data_dict = {key: item[:int(len(item) * train_val_split)] for key, item in data_dict.items()}
        # else:
        #     data_dict = {key: item[int(len(item) * train_val_split):] for key, item in data_dict.items()}

        # class_to_idx = {class_name: idx for idx, class_name in enumerate(sorted(np.unique(classes)))}

        # self.data = []
        # self.targets = []
        # for class_name, item in data_dict.items():
        #     for path in item:
        #         self.data.append(path)
        #         self.targets.append(class_to_idx[class_name])
