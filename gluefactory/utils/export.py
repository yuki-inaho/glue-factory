"""
Export the predictions of a model for a given dataloader (e.g. ImageFolder).
Use a standalone script with `python3 -m dsfm.scipts.export_predictions dir`
or call from another script.
"""

from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

from . import misc


@torch.no_grad()
def export_predictions(
    loader,
    model,
    output_file,
    as_half=False,
    keys="*",
    callback_fn=None,
    optional_keys=[],
    mode: str = "w",
    store_directional: bool = False,
):
    assert keys == "*" or isinstance(keys, (tuple, list))
    Path(output_file).parent.mkdir(exist_ok=True, parents=True)
    hfile = h5py.File(str(output_file), mode)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    for data_ in tqdm(loader):
        data = misc.batch_to_device(data_, device, non_blocking=True)
        pred = model(data)
        if callback_fn is not None:
            pred = {**callback_fn(pred, data), **pred}
        all_keys = set(pred.keys())
        if keys != "*":
            matched_keys = []
            for pattern in keys:
                found = False
                for key in all_keys - set(matched_keys):
                    if pattern in key:
                        matched_keys.append(key)
                        found = True
                assert found, f"Pattern {pattern} not found in prediction keys."
        else:
            matched_keys = list(all_keys)
        for pattern in optional_keys:
            for key in all_keys - set(matched_keys):
                if pattern in key:
                    matched_keys.append(key)

        pred = {k: v for k, v in pred.items() if k in matched_keys}
        assert len(pred) > 0

        # renormalization
        for k in pred.keys():
            if k.startswith("keypoints"):
                idx = k.replace("keypoints", "")
                scales = 1.0 / (
                    data["scales"] if len(idx) == 0 else data[f"view{idx}"]["scales"]
                )
                pred[k] = pred[k] * scales[None]
            if k.startswith("lines"):
                idx = k.replace("lines", "")
                scales = 1.0 / (
                    data["scales"] if len(idx) == 0 else data[f"view{idx}"]["scales"]
                )
                pred[k] = pred[k] * scales[None]
            if k.startswith("orig_lines"):
                idx = k.replace("orig_lines", "")
                scales = 1.0 / (
                    data["scales"] if len(idx) == 0 else data[f"view{idx}"]["scales"]
                )
                pred[k] = pred[k] * scales[None]

        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}

        if as_half:
            for k in pred:
                dt = pred[k].dtype
                if (dt == np.float32) and (dt != np.float16):
                    pred[k] = pred[k].astype(np.float16)
        try:
            name = data["name"][0]
            if store_directional:
                view_names = [x["name"][0] for x in misc.iterelements(data, "view")]
                assert (
                    len(view_names) == 2
                ), "Can only store directional data for 2-view inputs."
                pairs = [(0, 1), (1, 0)]
                for k, (i, j) in enumerate(pairs):
                    grpi = hfile.require_group(view_names[i])
                    grpi_j = grpi.create_group(view_names[j])
                    dict_to_h5group(grpi_j, misc.get_view(pred, str(k)))
            else:
                grp = hfile.create_group(name)
                dict_to_h5group(grp, pred)
        except RuntimeError:
            print(f"Skipping {name} (already in file?)")
            continue

        del pred
    hfile.close()
    return output_file


def dict_to_h5group(h5grp, data):
    """Write a nested dictionary to an h5 file."""
    for k, v in data.items():
        if isinstance(v, dict):
            dict_to_h5group(h5grp.create_group(k), v)
        elif isinstance(v, np.ndarray):
            h5grp.create_dataset(k, data=v)
        elif isinstance(v, torch.Tensor):
            h5grp.create_dataset(k, data=v.cpu().numpy())
        else:
            h5grp.attrs[k] = v


def dict_to_h5(file_path, data, modfe="w"):
    """Write a nested dictionary to an h5 file."""
    with h5py.File(file_path, modfe) as hfile:
        dict_to_h5group(hfile, data)
    return file_path
