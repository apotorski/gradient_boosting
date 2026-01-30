#!/usr/bin/env python3
import argparse
import logging

import jax
import jax.numpy as jnp
from jax import Array

from dataset_wrappers import Dataset
from forest import Forest, train_forest


def per_sample_loss_fn(predictions: Array, labels: Array) -> Array:
    return jnp.square(labels - predictions)


def generate_dataset() -> Dataset:
    feature_collections = jnp.column_stack(
        tuple(map(
            jnp.ravel,
            jnp.meshgrid(
                jnp.linspace(-2.0, 2.0, 2**10),
                jnp.linspace(-2.0, 2.0, 2**10)
            )
        ))
    )
    labels = jnp.square(feature_collections).sum(axis=1)
    weights = jnp.ones_like(labels)

    dataset = Dataset(feature_collections, labels, weights)

    return dataset


def split_dataset(
        dataset: Dataset,
        test_size: float,
        key: Array
        ) -> tuple[Dataset, Dataset]:
    feature_collections, labels, weights = dataset

    sample_number = len(labels)

    sample_indexes = jnp.arange(sample_number)
    shuffled_sample_indexes = jax.random.permutation(key, sample_indexes)

    split_index = round(test_size*sample_number)
    test_indexes = shuffled_sample_indexes[:split_index]
    training_indexes = shuffled_sample_indexes[split_index:]

    test_feature_collections = feature_collections[test_indexes]
    test_labels = labels[test_indexes]
    test_weights = weights[test_indexes]

    training_feature_collections = feature_collections[training_indexes]
    training_labels = labels[training_indexes]
    training_weights = weights[training_indexes]

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
        bin_number: int,
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
        bin_number,
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
        bin_number: int,
        test_size: float,
        validation_size: float
        ) -> None:
    key = jax.random.key(seed=0)

    dataset = generate_dataset()

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
        bin_number,
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

    parser = argparse.ArgumentParser(description='Train the regressor.')
    parser.add_argument('--iteration_number', type=int)
    parser.add_argument('--height', type=int)
    parser.add_argument('--regularization_coefficient', type=float)
    parser.add_argument('--leaf_weight_update_number', type=int)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--bin_number', type=int)
    parser.add_argument('--test_size', type=float)
    parser.add_argument('--validation_size', type=float)

    args = parser.parse_args()

    main(**vars(args))