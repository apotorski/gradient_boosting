from typing import NamedTuple

from jax import Array


class Dataset(NamedTuple):
    feature_collections: Array
    labels: Array
    weights: Array


class QuantizedDataset(NamedTuple):
    feature_collections: Array
    bin_edge_collections: Array
    labels: Array
    weights: Array