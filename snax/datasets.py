import requests
import gzip
import h5py
import json
import os
import pickle
import tarfile
import numpy as np
from pathlib import Path
from .data import Dataset

from typing import Dict

DEST = Path("/snax/datasets")
POINTERS = Path(__file__).parent / "datasets"
BINDINGS = {
    "mnist": lambda: load_mnistlike("mnist"),
    "fashion-mnist": lambda: load_mnistlike("fashion-mnist"),
    "k-mnist": lambda: load_mnistlike("k-mnist"),
    "cifar10": lambda: load_cifar10(),
}


def download_mnistlike(destination, info):
    dataset_path = destination / info["folder"]

    def read_idx(filepath, skipbytes):
        data = []
        with gzip.open(filepath, "r") as file:
            for i, line in enumerate(file):
                if i == 0:
                    data.append(np.array([pixel for pixel in line[skipbytes:]]))
                else:
                    data.append(np.array([pixel for pixel in line]))
        return np.concatenate(data)

    _ = os.makedirs(dataset_path)
    results = {}
    for k, i in info["data"].items():
        _ = open(dataset_path / i["filename"], "wb").write(
            requests.get(i["url"]).content
        )
        data = read_idx(dataset_path / i["filename"], i["skip"])
        results.update({k: np.reshape(data, i["size"])})

    with h5py.File(dataset_path / f"{info['folder']}.h5", "w") as hf:
        hf.create_dataset("train_images", data=results["train_images"])
        hf.create_dataset("test_images", data=results["test_images"])
        hf.create_dataset("test_labels", data=results["test_labels"])
        hf.create_dataset("train_labels", data=results["train_labels"])


def load_mnistlike(version):
    with open(POINTERS / f"{version}.json", "r") as file:
        pointer = json.load(file)

    dataset_path = DEST / pointer["folder"]
    if not (dataset_path.is_dir()):
        download_mnistlike(DEST, pointer)

    with h5py.File(dataset_path / f"{pointer['folder']}.h5", "r") as hf:
        train = {"image": hf["train_images"][:], "label": hf["train_labels"][:]}
        test = {"image": hf["test_images"][:], "label": hf["test_labels"][:]}

    return {
        "train": Dataset.from_tensor_slices(train),
        "test": Dataset.from_tensor_slices(test),
    }


def download_cifar10(destination, info):
    dataset_path = destination / info["folder"]
    _ = os.makedirs(dataset_path)

    filename = info["data"]["filename"]
    url = info["data"]["url"]
    _ = open(dataset_path / filename, "wb").write(requests.get(url).content)
    file = tarfile.open(dataset_path / filename, mode="r:gz")

    train_images = []
    train_labels = []
    test_images = []
    test_labels = []
    for batch in file:
        content = file.extractfile(batch)
        if hasattr(content, "read"):
            try:
                data = pickle.load(content, encoding="bytes")

                if data[b"batch_label"] == b"testing batch 1 of 1":
                    test_images.append(
                        np.transpose(
                            np.reshape(data[b"data"], (10000, 3, 32, 32)), (0, 2, 3, 1)
                        )
                    )
                    test_labels += data[b"labels"]
                else:
                    train_images.append(
                        np.transpose(
                            np.reshape(data[b"data"], (10000, 3, 32, 32)), (0, 2, 3, 1)
                        )
                    )
                    train_labels += data[b"labels"]
            except:
                pass

    train_images = np.concatenate(train_images)
    test_images = np.concatenate(test_images)

    train_labels = np.array(train_labels)[:, np.newaxis]
    test_labels = np.array(test_labels)[:, np.newaxis]

    with h5py.File(dataset_path / f"{info['folder']}.h5", "w") as hf:
        hf.create_dataset("train_images", data=train_images)
        hf.create_dataset("test_images", data=test_images)
        hf.create_dataset("test_labels", data=test_labels)
        hf.create_dataset("train_labels", data=train_labels)


def load_cifar10():
    with open(POINTERS / "cifar10.json", "r") as file:
        pointer = json.load(file)

    dataset_path = DEST / pointer["folder"]
    if not (dataset_path.is_dir()):
        download_cifar10(DEST, pointer)

    with h5py.File(dataset_path / f"{pointer['folder']}.h5", "r") as hf:
        train = {"image": hf["train_images"][:], "label": hf["train_labels"][:]}
        test = {"image": hf["test_images"][:], "label": hf["test_labels"][:]}

    return {
        "train": Dataset.from_tensor_slices(train),
        "test": Dataset.from_tensor_slices(test),
    }


def load(dataset_name: str) -> Dict[str, Dataset]:
    assert (
        dataset_name in BINDINGS.keys()
    ), f"dataset_name must be one of: {list(BINDINGS.keys())}"
    return BINDINGS[dataset_name]()
