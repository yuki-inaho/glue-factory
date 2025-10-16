""" """

from pathlib import Path

import numpy as np
import torch

from .. import settings
from ..utils import misc, preprocess
from .base_dataset import BaseDataset


class DoppelgangersDataset(BaseDataset):

    default_conf = {
        "root": "???",
        "preprocessing": preprocess.ImagePreprocessor.default_conf,
        "load_images": True,
        "subset": None,
    }

    def _init(self, conf):
        pass

    def get_dataset(self, split: str, epoch: int = 0):
        return DoppelgangersSplit(self.conf, split)

    def download(self):
        raise NotImplementedError(
            "Please download the doppelgangers dataset from the official repository."
        )


class DoppelgangersSplit(torch.utils.data.Dataset):

    def __init__(self, conf, split: str):
        self.conf = conf
        pairs_name = {
            "test": "pairs_metadata/test_pairs.npy",
        }[split]
        pair_f = settings.DATA_PATH / conf.root / pairs_name
        self.items = list(np.load(pair_f, allow_pickle=True))

        self.items = [
            x for x in self.items if ".gif" not in x[0] and ".gif" not in x[1]
        ]
        if conf.subset is not None:
            self.items = np.random.default_rng(32).choice(
                self.items, self.conf.subset, replace=False
            )
        self.preprocessor = preprocess.ImagePreprocessor(conf.preprocessing)
        self.split = split

        image_dir = {
            "test": "doppelgangers/images/test_set/",
        }[split]
        self.image_dir = settings.DATA_PATH / conf.root / image_dir

    def _read_view(self, name):
        path = self.image_dir / name
        img = preprocess.load_image(path)
        data = self.preprocessor(img)
        data["name"] = name
        return data

    def __getitem__(self, idx):
        name0, name1, has_overlap, num_matches = self.items[idx].tolist()
        data = {
            "name": "/".join([name0, name1]),
            "scene": name0.split("/")[0],
            "nviews": 2,
            "has_overlap": has_overlap,
            "num_matches": num_matches,
        }

        if self.conf.load_images:
            data["view0"] = self._read_view(name0)
            data["view1"] = self._read_view(name1)
        return data

    def __len__(self):
        return len(self.items)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    from ..visualization.viz2d import plot_image_grid

    conf = {
        "root": "doppelgangerspp",
        "preprocessing": {
            "resize": 512,
            "side": "long",
            "interpolation": "area",
            "antialias": False,
        },
        "num_workers": 1,
    }

    dataset = DoppelgangersDataset(conf)

    loader = dataset.get_data_loader("test")

    images = []
    for i, data in tqdm(enumerate(loader)):
        images.append(
            [
                data[f"view{i}"]["image"][0].permute(1, 2, 0)
                for i in range(data["nviews"][0])
            ]
        )
        if i > 3:
            print(misc.print_summary(data))
            break

    axes = plot_image_grid(images, dpi=200)
    plt.savefig("doppelgangers.png")
    plt.show()
