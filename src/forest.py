import logging
from pathlib import Path
from typing import Callable, NamedTuple, Self

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from dataset_wrappers import Dataset, QuantizedDataset
from tree import evaluate_tree, train_tree


class Forest:

    _height: int
    _split_feature_index_collections: Array
    _split_threshold_collections: Array
    _leaf_weight_collections: Array

    def __init__(
            self,
            height: int,
            split_feature_index_collections: Array,
            split_threshold_collections: Array,
            leaf_weight_collections: Array
            ) -> None:
        self._height = height
        self._split_feature_index_collections = split_feature_index_collections
        self._split_threshold_collections = split_threshold_collections
        self._leaf_weight_collections = leaf_weight_collections

    def __call__(self, feature_collections: Array) -> Array:
        def update_predictions(iteration: Array, predictions: Array) -> Array:
            return predictions + evaluate_tree(
                self._height,
                self._split_feature_index_collections[iteration],
                self._split_threshold_collections[iteration],
                self._leaf_weight_collections[iteration],
                feature_collections
            )

        tree_number = len(self._leaf_weight_collections)

        sample_number = len(feature_collections)

        predictions = jnp.zeros(sample_number)
        predictions: Array = jax.lax.fori_loop(
            0, tree_number, update_predictions, predictions
        )

        return predictions

    def save(self, forest_save_path: Path) -> None:
        buffer = np.empty(1, dtype=np.dtype([
            ('height', np.int64),
            (
                'split_feature_index_collections',
                self._split_feature_index_collections.dtype,
                self._split_feature_index_collections.shape
            ),
            (
                'split_threshold_collections',
                self._split_threshold_collections.dtype,
                self._split_threshold_collections.shape
            ),
            (
                'leaf_weight_collections',
                self._leaf_weight_collections.dtype,
                self._leaf_weight_collections.shape
            )
        ]))

        buffer['height'] = np.int64(self._height)
        buffer['split_feature_index_collections'] = np.asarray(
            self._split_feature_index_collections
        )
        buffer['split_threshold_collections'] = np.asarray(
            self._split_threshold_collections
        )
        buffer['leaf_weight_collections'] = np.asarray(
            self._leaf_weight_collections
        )

        np.save(forest_save_path, buffer)

    @classmethod
    def load(cls, forest_save_path: Path) -> Self:
        buffer, = np.load(forest_save_path)

        height = int(buffer['height'])
        split_feature_index_collections = jnp.asarray(
            buffer['split_feature_index_collections']
        )
        split_threshold_collections = jnp.asarray(
            buffer['split_threshold_collections']
        )
        leaf_weight_collections = jnp.asarray(
            buffer['leaf_weight_collections']
        )

        return cls(
            height,
            split_feature_index_collections,
            split_threshold_collections,
            leaf_weight_collections
        )

    @property
    def split_feature_index_collections(self) -> Array:
        return self._split_feature_index_collections

    @property
    def split_threshold_collections(self) -> Array:
        return self._split_threshold_collections

    @property
    def leaf_weight_collections(self) -> Array:
        return self._leaf_weight_collections


class ForestUpdateState(NamedTuple):

    split_feature_index_collections: Array
    split_threshold_collections: Array
    leaf_weight_collections: Array
    running_training_predictions: Array
    running_validation_predictions: Array
    training_loss: Array
    best_validation_loss: Array
    best_iteration: Array


def train_forest(
        per_sample_loss_fn: Callable[
            [Array, Array], Array
        ],
        iteration_number: int,
        height: int,
        regularization_coefficient: float,
        leaf_weight_update_number: int,
        learning_rate: float,
        bin_number: float,
        training_dataset: QuantizedDataset,
        validation_dataset: Dataset
        ) -> Forest:
    @jax.jit
    def loss_fn(predictions: Array, labels: Array, weights: Array) -> Array:
        per_sample_losses = per_sample_loss_fn(predictions, labels)

        loss = jnp.average(per_sample_losses, weights=weights)

        return loss

    per_sample_derivative_fn = jax.jit(jax.vmap(
        jax.value_and_grad(jax.grad(per_sample_loss_fn))
    ))

    quantized_training_dataset = quantize_dataset(training_dataset, bin_number)

    def update_forest(
            iteration: Array,
            state: ForestUpdateState
            ) -> ForestUpdateState:
        split_feature_indexes, split_thresholds, \
            leaf_weights, training_predictions = \
                train_tree(
                    loss_fn,
                    per_sample_derivative_fn,
                    quantized_training_dataset,
                    state.running_training_predictions,
                    height,
                    bin_number,
                    regularization_coefficient,
                    leaf_weight_update_number,
                    learning_rate
                )

        split_feature_index_collections = \
            state.split_feature_index_collections \
                .at[iteration].set(split_feature_indexes)

        split_threshold_collections = \
            state.split_threshold_collections \
                .at[iteration].set(split_thresholds)

        leaf_weight_collections = \
            state.leaf_weight_collections \
                .at[iteration].set(leaf_weights)

        running_training_predictions = jnp.add(
            state.running_training_predictions,
            training_predictions
        )

        validation_predictions = evaluate_tree(
            height,
            split_feature_indexes,
            split_thresholds,
            leaf_weights,
            validation_dataset.feature_collections
        )

        running_validation_predictions = jnp.add(
            state.running_validation_predictions,
            validation_predictions
        )

        training_loss = loss_fn(
            running_training_predictions,
            quantized_training_dataset.labels,
            quantized_training_dataset.weights
        )

        validation_loss = loss_fn(
            running_validation_predictions,
            validation_dataset.labels,
            validation_dataset.weights
        )

        flag = validation_loss < state.best_validation_loss
        best_validation_loss = jnp.where(
            flag, validation_loss, state.best_validation_loss
        )
        best_iteration = jnp.where(
            flag, iteration, state.best_iteration
        )

        return ForestUpdateState(
            split_feature_index_collections,
            split_threshold_collections,
            leaf_weight_collections,
            running_training_predictions,
            running_validation_predictions,
            training_loss,
            best_validation_loss,
            best_iteration
        )

    leaf_number = 2**height
    split_number = leaf_number - 1

    split_feature_index_collections = jnp.empty(
        shape=(iteration_number, split_number),
        dtype=jnp.uint32
    )
    split_threshold_collections = jnp.empty(
        shape=(iteration_number, split_number),
        dtype=jnp.float32
    )
    leaf_weight_collections = jnp.empty(
        shape=(iteration_number, leaf_number),
        dtype=jnp.float32
    )

    training_sample_number = len(training_dataset.feature_collections)
    validation_sample_number = len(validation_dataset.feature_collections)

    running_training_predictions = jnp.zeros(training_sample_number)
    running_validation_predictions = jnp.zeros(validation_sample_number)

    training_loss = jnp.inf
    best_validation_loss = jnp.inf
    best_iteration = 0

    forest_update_state = ForestUpdateState(
        split_feature_index_collections,
        split_threshold_collections,
        leaf_weight_collections,
        running_training_predictions,
        running_validation_predictions,
        training_loss,
        best_validation_loss,
        best_iteration
    )

    forest_update_state: ForestUpdateState = jax.lax.fori_loop(
        0, iteration_number, update_forest, forest_update_state
    )

    training_loss = forest_update_state.training_loss
    best_validation_loss = forest_update_state.best_validation_loss
    best_iteration = forest_update_state.best_iteration

    tree_number = best_iteration + 1

    split_feature_index_collections = forest_update_state \
        .split_feature_index_collections.at[:tree_number].get()

    split_threshold_collections = forest_update_state \
        .split_threshold_collections.at[:tree_number].get()

    leaf_weight_collections = forest_update_state \
        .leaf_weight_collections.at[:tree_number].get()

    forest = Forest(
        height,
        split_feature_index_collections,
        split_threshold_collections,
        leaf_weight_collections
    )

    logging.info(
        f'Model is fitted - tree number = {tree_number:,} '
        f'- training loss = {training_loss:.6f} '
        f'- best validation loss = {best_validation_loss:.6f}'
    )

    return forest


def quantize_dataset(dataset: Dataset, bin_number: int) -> QuantizedDataset:
    feature_collections, labels, weights = dataset

    quantiles = jnp.linspace(0.0, 1.0, bin_number + 1)

    def quantize_features(features: Array) -> tuple[Array, Array]:
        bin_edges = jnp.quantile(features, quantiles)[1:-1]

        quantized_features = jnp.digitize(
            features, bin_edges, right=True
        ).astype(jnp.uint8)

        return quantized_features, bin_edges

    transposed_quantized_feature_collections, bin_edge_collections = \
        jax.lax.map(quantize_features, feature_collections.transpose())

    quantized_feature_collections = \
        transposed_quantized_feature_collections.transpose()

    quantized_dataset = QuantizedDataset(
        quantized_feature_collections,
        bin_edge_collections,
        labels,
        weights
    )

    return quantized_dataset