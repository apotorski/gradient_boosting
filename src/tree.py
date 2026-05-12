from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array

from dataset_wrappers import QuantizedDataset

EPSILON = jnp.float32(1e-8)


def evaluate_tree(
        height: int,
        split_feature_idxs: Array,
        split_thresholds: Array,
        leaf_weights: Array,
        feature_collections: Array
        ) -> Array:
    sample_number = len(feature_collections)
    sample_idxs = jnp.arange(sample_number)

    def update_node_idxs(_: Array, node_idxs: Array) -> Array:
        features = feature_collections[
            sample_idxs,
            split_feature_idxs[
                node_idxs
            ]
        ]

        outcomes = jnp.greater(
            features,
            split_thresholds[
                node_idxs
            ]
        )

        node_idxs = 2*node_idxs + jnp.where(outcomes, 2, 1)

        return node_idxs

    node_idxs = jnp.zeros(sample_number, dtype=jnp.uint32)
    node_idxs: Array = jax.lax.fori_loop(
        0, height, update_node_idxs, node_idxs
    )

    split_number = 2**height - 1
    leaf_idxs = node_idxs - split_number

    predictions = leaf_weights[leaf_idxs]

    return predictions


def train_tree(
        loss_fn: Callable[
            [Array, Array, Array], Array
        ],
        per_sample_derivative_fn: Callable[
            [Array, Array], tuple[Array, Array]
        ],
        dataset: QuantizedDataset,
        running_predictions: Array,
        height: int,
        feature_bin_number: int,
        regularization_coefficient: float,
        leaf_weight_update_number: int,
        learning_rate: float
        ) -> tuple[Array, Array, Array, Array]:
    feature_collections, feature_bin_collections, labels, weights = dataset

    sample_number, feature_number = feature_collections.shape

    sample_idxs = jnp.arange(sample_number, dtype=jnp.uint32)
    feature_idxs = jnp.arange(feature_number, dtype=jnp.uint32)

    leaf_idxs = jnp.zeros(sample_number, dtype=jnp.uint32)

    addend_pairs = _compute_addend_pairs(
        per_sample_derivative_fn, running_predictions, labels, weights
    )

    leaf_number = 2**height
    split_number = leaf_number - 1

    split_feature_idxs = jnp.empty(split_number, dtype=jnp.uint32)
    quantized_split_thresholds = jnp.empty(split_number, dtype=jnp.uint32)

    start_node_idx = end_node_idx = 0
    for level in range(height):
        level_leaf_number = 2**level

        start_node_idx = end_node_idx
        end_node_idx = start_node_idx + level_leaf_number

        level_split_feature_idxs, level_quantized_split_thresholds = \
            _compute_split(
                level_leaf_number, feature_number, feature_bin_number,
                leaf_idxs, feature_idxs, feature_collections,
                addend_pairs, regularization_coefficient
            )

        split_feature_idxs = split_feature_idxs \
            .at[start_node_idx:end_node_idx] \
                .set(level_split_feature_idxs)

        quantized_split_thresholds = quantized_split_thresholds \
            .at[start_node_idx:end_node_idx] \
                .set(level_quantized_split_thresholds)

        features = feature_collections[
            sample_idxs,
            level_split_feature_idxs[
                leaf_idxs
            ]
        ]

        level_outcomes = jnp.greater(
            features,
            level_quantized_split_thresholds[
                leaf_idxs
            ]
        )

        leaf_idxs = 2*leaf_idxs + jnp.where(level_outcomes, 1, 0)

    leaf_weights = _compute_leaf_weights(
        leaf_number, leaf_idxs, addend_pairs,
        regularization_coefficient,
        loss_fn, running_predictions, labels, weights,
        leaf_weight_update_number, per_sample_derivative_fn
    )*learning_rate

    split_thresholds = feature_bin_collections[
        split_feature_idxs, quantized_split_thresholds
    ]

    predictions = leaf_weights[leaf_idxs]

    return split_feature_idxs, split_thresholds, leaf_weights, predictions


def _compute_addend_pairs(
        per_sample_derivative_fn: Callable[
            [Array, Array], tuple[Array, Array]
        ],
        predictions: Array,
        labels: Array,
        weights: Array
        ) -> Array:
    return jnp.column_stack(
        per_sample_derivative_fn(predictions, labels)
    )*jnp.expand_dims(weights, axis=1)


def _compute_split(
        leaf_number: int,
        feature_number: int,
        feature_bin_number: int,
        leaf_idxs: Array,
        feature_idxs: Array,
        feature_collections: Array,
        addend_pairs: Array,
        regularization_coefficient: float
        ) -> tuple[Array, Array]:
    sum_pairs = jnp.zeros(
        shape=(
            leaf_number,
            feature_number,
            feature_bin_number,
            addend_pairs.shape[-1]
        )
    ).at[
        (
            leaf_idxs[:, jnp.newaxis],
            feature_idxs[jnp.newaxis, :],
            feature_collections
        )
    ].add(
        addend_pairs[:, jnp.newaxis]
    )

    cumulative_sum_pairs = jnp.cumsum(sum_pairs, axis=2)

    left_sum_pairs = cumulative_sum_pairs[:, :, :-1]
    total_sum_pairs = cumulative_sum_pairs[:, :, [-1]]
    right_sum_pairs = total_sum_pairs - left_sum_pairs

    left_proxy_scores = _compute_proxy_scores(
        left_sum_pairs, regularization_coefficient
    )
    right_proxy_scores = _compute_proxy_scores(
        right_sum_pairs, regularization_coefficient
    )

    proxy_scores = left_proxy_scores + right_proxy_scores

    feature_idxs, quantized_thresholds = jnp.unravel_index(
        proxy_scores.reshape(proxy_scores.shape[0], -1).argmax(axis=1),
        proxy_scores.shape[1:]
    )

    return feature_idxs.astype(jnp.uint32), \
        quantized_thresholds.astype(jnp.uint32)


def _compute_leaf_weights(
        leaf_number: int,
        leaf_idxs: Array,
        addend_pairs: Array,
        regularization_coefficient: float,
        loss_fn: Callable[
            [Array, Array, Array], Array
        ],
        predictions: Array,
        labels: Array,
        weights: Array,
        leaf_weight_update_number: int,
        per_sample_derivative_fn: Callable[
            [Array, Array], tuple[Array, Array]
        ]
        ) -> Array:
    def update_leaf_weights(_: Array, leaf_weights: Array) -> Array:
        updated_predictions = predictions + leaf_weights[leaf_idxs]

        updated_addend_pairs = _compute_addend_pairs(
            per_sample_derivative_fn, updated_predictions, labels, weights
        )

        delta_leaf_weights = _compute_delta_leaf_weights(
            leaf_number, leaf_idxs, updated_addend_pairs,
            regularization_coefficient,
            loss_fn, updated_predictions, labels, weights
        )

        return leaf_weights + delta_leaf_weights

    leaf_weights = _compute_delta_leaf_weights(
        leaf_number, leaf_idxs, addend_pairs,
        regularization_coefficient,
        loss_fn, predictions, labels, weights
    )

    leaf_weights = jax.lax.fori_loop(
        1, leaf_weight_update_number, update_leaf_weights, leaf_weights
    )

    return leaf_weights


def _compute_proxy_scores(
        sum_pairs: Array,
        regularization_coefficient: float
        ) -> Array:
    gradients, hessians = sum_pairs[..., 0], sum_pairs[..., 1]

    proxy_scores = jnp.square(gradients) \
        / (hessians + regularization_coefficient + EPSILON)

    return proxy_scores


def _compute_delta_leaf_weights(
        leaf_number: int,
        leaf_idxs: Array,
        addend_pairs: Array,
        regularization_coefficient: float,
        loss_fn: Callable[
            [Array, Array, Array], Array
        ],
        predictions: Array,
        labels: Array,
        weights: Array
        ) -> Array:
    sum_pairs = jnp.zeros(shape=(leaf_number, addend_pairs.shape[-1])) \
        .at[leaf_idxs].add(addend_pairs)

    gradients, hessians = sum_pairs[..., 0], sum_pairs[..., 1]

    delta_leaf_weights = jnp.negative(gradients) \
        / (hessians + regularization_coefficient + EPSILON)

    previous_loss = loss_fn(predictions, labels, weights)

    def check_decrease_condition(step_length: Array) -> Array:
        delta_predictions = (step_length*delta_leaf_weights)[leaf_idxs]

        loss = loss_fn(predictions + delta_predictions, labels, weights)

        return (loss > previous_loss) & ~jnp.isclose(step_length, 0.0)

    def decrease_step_length(step_length: Array) -> Array:
        return jnp.multiply(0.5, step_length)

    step_length = jax.lax.while_loop(
        check_decrease_condition, decrease_step_length, 1.0
    )

    delta_leaf_weights *= step_length

    return delta_leaf_weights
