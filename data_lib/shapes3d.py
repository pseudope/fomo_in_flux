import json
import os

import h5py
import numpy as np
from PIL import Image
import tqdm

import data_lib

BASE_CLASSES = json.load(open("data_lib/00_info/SHAPES3D_classnames.json", "r"))
PRIMER = "A photo of a synthetic {}."
CLASSES = [PRIMER.format(x) for x in BASE_CLASSES]

PARAMS = {
    "base_classes": BASE_CLASSES,
    "classes": CLASSES,
    "num_classes": len(CLASSES),
    "mean": (0.5066, 0.5805, 0.6005),
    "std": (0.2921, 0.3447, 0.3696),
    # Default Imagesize
    "img_size": 224,
    # If it's computationally recommended to resize in advance
    "create_resized_variant_if_possible": False,
    "primer": PRIMER,
    "eval_only": False,
    "type": "classification",
}

lat_names = ("floorCol", "wallCol", "objCol", "objSize", "objType", "objAzimuth")
lat_sizes = np.array([10, 10, 10, 8, 4, 15])


class Dataset(data_lib.DatasetScaffold):
    def __init__(self, root, **kwargs) -> None:
        super(Dataset, self).__init__(root, **kwargs)

        self.root = os.path.join(root, "SHAPES3D")
        self.PARAMS = PARAMS

        self._run_default_setup()

    def _download_and_setup(self):
        image_folder = f"{self.root}/shapes3d-images-{self.split}"
        if not os.path.exists(image_folder):
            if (
                not (os.path.isdir(self.root) and len(os.listdir(self.root)) > 0)
                and self.download
            ):
                os.makedirs(self.root, exist_ok=True)
                base_url = "https://storage.googleapis.com/3d-shapes/3dshapes.h5"
                os.system(f"wget -O {self.root}/shapes3d.h5 {base_url}")

            with h5py.File(f"{self.root}/shapes3d.h5", "r") as shapes3d_data:
                imgs = shapes3d_data["images"][()]
                lat_values = shapes3d_data["labels"][()]

            # np.random.seed(0)
            # avail_latents = np.array(list(it.product(range(10), range(10), range(10), range(8), range(4), range(15))))
            # train_indices = list(np.random.choice(len(avail_latents), 75000, replace=False))
            # test_indices = list(np.random.choice(list(set(range(len(avail_latents))) - set(train_indices)), 25000, replace=False))
            # train_latents = avail_latents[sorted(train_indices)]
            # test_latents = avail_latents[sorted(test_indices)]
            # mul = np.array([[48000, 4800, 480, 60, 15, 1]])
            # train_indices = np.sum(train_latents * mul, axis=-1)
            # test_indices = np.sum(test_latents * mul, axis=-1)

            # ll = []
            # num = {
            #     0: 0, 0.1: 1, 0.2: 2, 0.3: 3, 0.4: 3,
            #     0.5: 4, 0.6: 4, 0.7: 4, 0.8: 5, 0.9: 5
            # }
            # for lab in tqdm.tqdm(lat_values, desc='Generating label data...'):
            #     ll.append(lab[4] * 6 + num[np.round(lab[2],1)])
            # ll = np.array(ll).astype(int)

            # np.save(f'{self.root}/shapes3d_data_train.npy', imgs[train_indices])
            # np.save(f'{self.root}/shapes3d_data_test.npy', imgs[test_indices])
            # np.save(f'{self.root}/shapes3d_labels_train.npy', ll[train_indices])
            # np.save(f'{self.root}/shapes3d_labels_test.npy', ll[test_indices])

            # os.system(f'rm {self.root}/shapes3d.h5')

            files = json.load(open(f"data_lib/00_info/SHAPES3D_{self.split}.json", "r"))
            files = list(files.keys())
            indices = [int(x.split("/")[-1].split(".")[0]) for x in files]

            os.makedirs(image_folder, exist_ok=True)
            for i in tqdm.tqdm(indices, desc="Creating Shapes3D images..."):
                savepath = f"{image_folder}/{i}.png"
                _ = Image.fromarray(imgs[i]).convert("RGB").save(savepath)
