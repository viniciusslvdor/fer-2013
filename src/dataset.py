import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import imagehash
from PIL import Image
from sklearn.neighbors import NearestNeighbors

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DATASET_PATH = DATA_DIR / "external" / "icml_face_data.csv"
PROCESSED_DATASET_PATH = DATA_DIR / "processed" / "fer2013_clean.npz"

EMOTIONS = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}

USAGE_TO_SPLIT = {"Training": "train", "PublicTest": "val", "PrivateTest": "test"}

PHASH_HAMMING_THRESHOLD_BITS = 5


def load_raw_dataset(path=RAW_DATASET_PATH):
    df = pd.read_csv(path)
    return df.rename(columns={" pixels": "pixels_str", " Usage": "usage"})


def decode_pixels(df):
    df = df.copy()
    df["pixels"] = df["pixels_str"].apply(
        lambda pixels_str: np.array(list(map(int, pixels_str.split())), dtype=np.uint8).reshape(48, 48)
    )
    return df


def _hash_pixels(pixels_str):
    return hashlib.md5(pixels_str.encode()).hexdigest()


def remove_exact_duplicates(df):
    df = df.copy()
    df["hash"] = df["pixels_str"].apply(_hash_pixels)

    usage_combinations = df.groupby("hash")["usage"].apply(lambda s: tuple(sorted(s.unique())))
    hashes_to_drop_from_train = usage_combinations[
        usage_combinations.apply(lambda combo: "Training" in combo and len(combo) > 1)
    ].index

    df = df[~(df["hash"].isin(hashes_to_drop_from_train) & (df["usage"] == "Training"))]
    return df.drop(columns="hash")


def remove_near_duplicates(df, threshold_bits=PHASH_HAMMING_THRESHOLD_BITS):
    df = df.copy()
    phashes = df["pixels"].apply(lambda pixels: imagehash.phash(Image.fromarray(pixels)))
    hash_matrix = np.stack(phashes.apply(lambda h: h.hash.flatten()).values)

    n_bits = hash_matrix.shape[1]
    nn = NearestNeighbors(metric="hamming", radius=threshold_bits / n_bits)
    nn.fit(hash_matrix)
    neighbor_indices = nn.radius_neighbors(hash_matrix, return_distance=False)

    usage = df["usage"].values
    near_dup_train_positions = set()
    for i, neighbors in enumerate(neighbor_indices):
        for j in neighbors:
            if j > i and usage[i] != usage[j]:
                if usage[i] == "Training":
                    near_dup_train_positions.add(i)
                if usage[j] == "Training":
                    near_dup_train_positions.add(j)

    rows_to_drop = df.iloc[list(near_dup_train_positions)].index
    return df.drop(index=rows_to_drop)


def build_clean_dataset(path=RAW_DATASET_PATH):
    df = load_raw_dataset(path)
    df = decode_pixels(df)
    df = remove_exact_duplicates(df)
    df = remove_near_duplicates(df)
    return df


def split_dataset(df):
    arrays = {}
    for usage, split_name in USAGE_TO_SPLIT.items():
        subset = df[df["usage"] == usage]
        arrays[f"images_{split_name}"] = np.stack(subset["pixels"].values)
        arrays[f"labels_{split_name}"] = subset["emotion"].values
    return arrays


def save_dataset(arrays, path=PROCESSED_DATASET_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_processed_dataset(path=PROCESSED_DATASET_PATH):
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


if __name__ == "__main__":
    clean_df = build_clean_dataset()
    dataset_arrays = split_dataset(clean_df)
    save_dataset(dataset_arrays)

    print(f"Saved processed dataset to {PROCESSED_DATASET_PATH}")
    for key, value in dataset_arrays.items():
        print(f"{key}: {value.shape}")
