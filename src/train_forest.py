#!/usr/bin/env python3
import argparse
import logging

import jax
from jax import Array
from jax.nn import log_sigmoid
import jax.numpy as jnp

from dataset_wrappers import Dataset
from forest import Forest, train_forest
from sample_generators import generate_moons


def per_sample_loss_fn(logits: Array, labels: Array) -> Array:
    return (1 - labels)*logits - log_sigmoid(logits)


def create_dataset(key: Array) -> Dataset:
    feature_collections, labels = generate_moons(key)

    sample_number = labels.size
    negative_sample_number, positive_sample_number = jnp.bincount(labels)

    negative_sample_weight = sample_number/negative_sample_number/2
    positive_sample_weight = sample_number/positive_sample_number/2

    weights = jnp.empty(sample_number) \
        .at[labels == 0].set(negative_sample_weight) \
        .at[labels == 1].set(positive_sample_weight)

    dataset = Dataset(feature_collections, labels, weights)

    return dataset


def split_dataset(
        dataset: Dataset,
        test_size: float,
        key: Array
        ) -> tuple[Dataset, Dataset]:
    feature_collections, labels, weights = dataset

    negative_sample_idxs, = jnp.nonzero(labels == 0)
    positive_sample_idxs, = jnp.nonzero(labels == 1)

    key, subkey = jax.random.split(key)
    negative_sample_idxs = jax.random.permutation(subkey, negative_sample_idxs)

    key, subkey = jax.random.split(key)
    positive_sample_idxs = jax.random.permutation(subkey, positive_sample_idxs)

    negative_sample_split_idx = round(test_size*negative_sample_idxs.size)
    positive_sample_split_idx = round(test_size*positive_sample_idxs.size)

    test_sample_idxs = jnp.concatenate([
        negative_sample_idxs[:negative_sample_split_idx],
        positive_sample_idxs[:positive_sample_split_idx]
    ])

    training_sample_idxs = jnp.concatenate([
        negative_sample_idxs[negative_sample_split_idx:],
        positive_sample_idxs[positive_sample_split_idx:]
    ])

    key, subkey = jax.random.split(key)
    test_sample_idxs = jax.random.permutation(subkey, test_sample_idxs)

    key, subkey = jax.random.split(key)
    training_sample_idxs = jax.random.permutation(subkey, training_sample_idxs)

    test_feature_collections = feature_collections[test_sample_idxs]
    test_labels = labels[test_sample_idxs]
    test_weights = weights[test_sample_idxs]

    training_feature_collections = feature_collections[training_sample_idxs]
    training_labels = labels[training_sample_idxs]
    training_weights = weights[training_sample_idxs]

    test_dataset = Dataset(
        test_feature_collections,
        test_labels,
        test_weights
    )

    training_dataset = Dataset(
        training_feature_collections,
        training_labels,
        training_weights
    )

    return test_dataset, training_dataset


def train_model(
        iteration_number: int,
        height: int,
        regularization_coefficient: float,
        leaf_weight_update_number: int,
        learning_rate: float,
        feature_bin_number: int,
        dataset: Dataset,
        validation_size: float,
        key: Array
        ) -> Forest:
    validation_dataset, training_dataset = \
        split_dataset(dataset, validation_size, key)

    forest = train_forest(
        per_sample_loss_fn,
        iteration_number,
        height,
        regularization_coefficient,
        leaf_weight_update_number,
        learning_rate,
        feature_bin_number,
        training_dataset,
        validation_dataset
    )

    return forest


def evaluate_model(forest: Forest, dataset: Dataset) -> None:
    feature_collections, labels, weights = dataset

    predictions = forest(feature_collections)

    per_sample_losses = per_sample_loss_fn(predictions, labels)
    loss = jnp.average(per_sample_losses, weights=weights)

    logging.info(f'Model is evaluated - test loss = {loss:.6f}')


def main(
        iteration_number: int,
        height: int,
        regularization_coefficient: float,
        leaf_weight_update_number: int,
        learning_rate: float,
        feature_bin_number: int,
        test_size: float,
        validation_size: float
        ) -> None:
    key = jax.random.key(seed=0)

    key, subkey = jax.random.split(key)
    dataset = create_dataset(subkey)

    key, subkey = jax.random.split(key)
    test_dataset, training_dataset = \
        split_dataset(dataset, test_size, subkey)

    key, subkey = jax.random.split(key)
    model = train_model(
        iteration_number,
        height,
        regularization_coefficient,
        leaf_weight_update_number,
        learning_rate,
        feature_bin_number,
        training_dataset,
        validation_size,
        subkey
    )

    evaluate_model(model, test_dataset)


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

    parser = argparse.ArgumentParser(description='Train the decision forest.')
    parser.add_argument('--iteration_number', type=int)
    parser.add_argument('--height', type=int)
    parser.add_argument('--regularization_coefficient', type=float)
    parser.add_argument('--leaf_weight_update_number', type=int)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--feature_bin_number', type=int)
    parser.add_argument('--test_size', type=float)
    parser.add_argument('--validation_size', type=float)

    args = parser.parse_args()

    main(**vars(args))
