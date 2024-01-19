import logging
import argparse

import torch
import numpy as np

from sklearn.cluster import KMeans
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def cluster(args):
    with open(args.subset) as file:
        subset = [line.strip() for line in file]

    logger.info(f"Loading features from {args.in_dir}")
    features = []
    for path in tqdm(subset):
        in_path = args.in_dir / path
        features.append(np.load(in_path.with_suffix(".npy")))
    features = np.concatenate(features, axis=0).reshape(-1, 1)

    logger.info(f"Clustering features of shape: {features.shape}")
    kmeans = KMeans(n_clusters=args.n_clusters).fit(features)

    checkpoint_path = args.checkpoint_dir / f"kmeans_{args.n_clusters}.pt"
    checkpoint_path.parent.mkdir(exist_ok=True, parents=True)
    # torch.save(
    #     {
    #         "n_features_in_": kmeans.n_features_in_,
    #         "_n_threads": kmeans._n_threads,
    #         "cluster_centers_": kmeans.cluster_centers_,
    #     },
    #     str(checkpoint_path),
    # )
    torch.save(
        kmeans.get_params(),
        str(checkpoint_path),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster speech features.")
    parser.add_argument(
        "in_dir",
        metavar="in-dir",
        help="path to the encoded dataset",
        type=Path,
    )
    parser.add_argument(
        "subset",
        metavar="subset",
        help="path to the .txt file containing the list of files to cluster",
        type=Path,
    )
    parser.add_argument(
        "checkpoint_dir",
        metavar="checkpoint-dir",
        help="path to the checkpoint directory",
        type=Path,
    )
    parser.add_argument(
        "--n-clusters",
        help="number of clusters",
        type=int,
        default=100,
    )
    args = parser.parse_args()
    cluster(args)
